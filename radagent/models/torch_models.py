"""PyTorch segmenters.

* ``ViTUNETR`` - the hybrid ViT encoder + U-Net decoder from the
  ``ViTUNETR25D_LiverSegmentation`` notebook (ImageNet ViT-Small via timm, 2.5D
  three-slice input, CNN side branch for sharp edges). Changes for this project:
  ``n_classes=3`` (background / liver / tumour) and a ``Dropout2d`` before the head so
  that Monte-Carlo dropout can be used as an uncertainty estimate. The dropout layer
  has no parameters, so 2-class notebook checkpoints still load (the head is
  re-initialised for 3 classes - see :func:`load_warm_start`).
* ``UNet25D`` - a ~1.9 M-parameter 2.5D U-Net with decoder dropout; trains on a CPU.

``TorchSegmenter`` wraps either one behind the :class:`Segmenter` interface with
flip test-time augmentation x MC-dropout passes.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from radagent.models.base import N_CLASSES, Segmenter
from radagent.preprocess import to_model_range


# --------------------------------------------------------------------------- blocks
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def up_chain(in_channels: int, out_channels: int, num_upsamples: int) -> nn.Sequential:
    layers, ch = [], in_channels
    for _ in range(num_upsamples):
        layers += [nn.ConvTranspose2d(ch, out_channels, kernel_size=2, stride=2),
                   nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        ch = out_channels
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- ViT-UNETR
class ViTUNETR(nn.Module):
    """ViT-UNETR 2.5D (architecture of the liver-segmentation notebook, 3-class head)."""

    def __init__(self, vit_name: str = "vit_small_patch16_224.augreg_in21k_ft_in1k", img_size: int = 256,
                 in_ch: int = 3, n_classes: int = N_CLASSES, pretrained: bool = True, decoder_dropout: float = 0.1):
        super().__init__()
        import timm

        self.vit = timm.create_model(vit_name, pretrained=pretrained, img_size=img_size, in_chans=in_ch, num_classes=0)
        embed_dim = self.vit.embed_dim
        depth = len(self.vit.blocks)
        self.taps = [depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]

        self.stem = ConvBlock(in_ch, 16)
        self.cnn1 = ConvBlock(16, 32)
        self.cnn2 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(embed_dim, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.skip2 = up_chain(embed_dim, 128, num_upsamples=1)
        self.dec3 = ConvBlock(128 + 128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.skip1 = up_chain(embed_dim, 64, num_upsamples=2)
        self.dec2 = ConvBlock(64 + 64 + 64, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.skip0 = up_chain(embed_dim, 32, num_upsamples=3)
        self.dec1 = ConvBlock(32 + 32 + 32, 32)
        self.up0 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(16 + 16, 16)
        self.drop = nn.Dropout2d(decoder_dropout)           # NEW: MC-dropout uncertainty
        self.head = nn.Conv2d(16, n_classes, kernel_size=1)

    def encoder_parameters(self):
        return self.vit.parameters()

    def forward(self, x):
        f0, f1, f2, f3 = self.vit.get_intermediate_layers(x, n=self.taps, reshape=True)
        s0 = self.stem(x)
        s1 = self.cnn1(self.pool(s0))
        s2 = self.cnn2(self.pool(s1))
        y = self.bottleneck(f3)
        y = self.dec3(torch.cat([self.up3(y), self.skip2(f2)], dim=1))
        y = self.dec2(torch.cat([self.up2(y), self.skip1(f1), s2], dim=1))
        y = self.dec1(torch.cat([self.up1(y), self.skip0(f0), s1], dim=1))
        y = self.dec0(torch.cat([self.up0(y), s0], dim=1))
        return self.head(self.drop(y))


# --------------------------------------------------------------------------- UNet25D
class UNet25D(nn.Module):
    """Small 2.5D U-Net (3 input slices) with dropout in the bottleneck and decoder."""

    def __init__(self, in_ch: int = 3, n_classes: int = N_CLASSES, base: int = 16, dropout: float = 0.1):
        super().__init__()
        b = base
        self.enc0, self.enc1, self.enc2, self.enc3 = ConvBlock(in_ch, b), ConvBlock(b, 2 * b), ConvBlock(2 * b, 4 * b), ConvBlock(4 * b, 8 * b)
        self.pool = nn.MaxPool2d(2)
        self.bott = ConvBlock(8 * b, 16 * b)
        self.drop = nn.Dropout2d(dropout)
        self.up3, self.dec3 = nn.ConvTranspose2d(16 * b, 8 * b, 2, 2), ConvBlock(16 * b, 8 * b)
        self.up2, self.dec2 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2), ConvBlock(8 * b, 4 * b)
        self.up1, self.dec1 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), ConvBlock(4 * b, 2 * b)
        self.up0, self.dec0 = nn.ConvTranspose2d(2 * b, b, 2, 2), ConvBlock(2 * b, b)
        self.head = nn.Conv2d(b, n_classes, 1)

    def encoder_parameters(self):
        return []

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        y = self.drop(self.bott(self.pool(e3)))
        y = self.drop(self.dec3(torch.cat([self.up3(y), e3], 1)))
        y = self.drop(self.dec2(torch.cat([self.up2(y), e2], 1)))
        y = self.dec1(torch.cat([self.up1(y), e1], 1))
        y = self.dec0(torch.cat([self.up0(y), e0], 1))
        return self.head(y)


# --------------------------------------------------------------------------- helpers
def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_network(backend: str, seg_cfg: dict, img_size: int, pretrained: bool | None = None) -> nn.Module:
    if backend == "vitunetr":
        return ViTUNETR(vit_name=seg_cfg["vit_name"], img_size=img_size, n_classes=N_CLASSES,
                        pretrained=seg_cfg["pretrained"] if pretrained is None else pretrained,
                        decoder_dropout=seg_cfg["decoder_dropout"])
    if backend == "unet25d":
        return UNet25D(n_classes=N_CLASSES, base=seg_cfg["unet_base"], dropout=seg_cfg["decoder_dropout"])
    raise ValueError(f"not a torch backend: {backend}")


def _extract_state_dict(obj) -> dict:
    if isinstance(obj, nn.Module):
        obj = obj.state_dict()
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    clean = {}
    for k, v in obj.items():
        for prefix in ("module.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        clean[k] = v
    return clean


def load_warm_start(model: nn.Module, path: str | Path, verbose: bool = True) -> dict:
    """Load every tensor whose name and shape match (e.g. a 2-class liver checkpoint).

    Mismatching tensors - typically ``head.weight``/``head.bias`` going from 2 to 3
    classes - keep their fresh initialisation.
    """
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only
        obj = torch.load(path, map_location="cpu")
    sd = _extract_state_dict(obj)
    own = model.state_dict()
    loadable = {k: v for k, v in sd.items() if k in own and tuple(own[k].shape) == tuple(v.shape)}
    skipped = sorted(k for k in sd if k not in loadable)
    missing = sorted(k for k in own if k not in loadable)
    model.load_state_dict(loadable, strict=False)
    report = {"loaded": len(loadable), "skipped_from_checkpoint": skipped, "left_initialised": missing}
    if verbose:
        print(f"[warm-start] loaded {len(loadable)}/{len(own)} tensors from {path}; "
              f"re-initialised: {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    return report


def amp_context(device: torch.device):
    """Mixed precision on CUDA only; a no-op context elsewhere (CPU/MPS autocast support varies by version)."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _set_mc_dropout(model: nn.Module, on: bool) -> None:
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train(on)


class TorchSegmenter(Segmenter):
    def __init__(self, model: nn.Module, backend: str, device: torch.device, tta_flip: bool = True,
                 mc_samples: int = 2, version: str = "1.0"):
        self.model = model.to(device)
        self.name = backend
        self.version = version
        self.device = device
        self.tta_flip = tta_flip
        self.mc_samples = max(1, int(mc_samples))

    @property
    def n_passes(self) -> int:
        return self.mc_samples * (2 if self.tta_flip else 1)

    @torch.no_grad()
    def predict_batch(self, stacks_u8: np.ndarray, passes: int | None = None, roi: tuple | None = None,
                      coarse: bool = False) -> np.ndarray:
        x = torch.from_numpy(to_model_range(stacks_u8)).to(self.device)
        quick = passes is not None and passes <= 1
        mc = 1 if quick else self.mc_samples
        _set_mc_dropout(self.model, on=mc > 1)
        outs = []
        with amp_context(self.device):
            for _ in range(mc):
                outs.append(F.softmax(self.model(x).float(), dim=1))
                if self.tta_flip and not quick:
                    outs.append(torch.flip(F.softmax(self.model(torch.flip(x, [3])).float(), dim=1), [3]))
        _set_mc_dropout(self.model, on=False)
        return torch.stack(outs).float().cpu().numpy()

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "backend": self.name, "version": self.version,
                    **(extra or {})}, path)

    @classmethod
    def load(cls, path: str | Path, seg_cfg: dict, img_size: int, inf_cfg: dict) -> "TorchSegmenter":
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        backend = ckpt.get("backend", seg_cfg["backend"])
        model = build_network(backend, seg_cfg, img_size, pretrained=False)   # weights come from the checkpoint
        model.load_state_dict(_extract_state_dict(ckpt), strict=True)
        return cls(model, backend, pick_device(seg_cfg.get("device", "auto")), tta_flip=inf_cfg["tta_flip"],
                   mc_samples=inf_cfg["mc_samples"], version=str(ckpt.get("version", "1.0")))
