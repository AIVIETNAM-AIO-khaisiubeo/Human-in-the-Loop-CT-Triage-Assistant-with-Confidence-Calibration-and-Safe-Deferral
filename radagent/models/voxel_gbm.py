"""Classical baseline segmenter: per-pixel gradient-boosted trees on 2.5D image features.

Purpose: a CPU-only, dependency-light tool so the whole agentic workflow can be run and
tested without a GPU or PyTorch. It is deliberately *not* the headline model - that is
the ViT-UNETR - but it is a legitimate probabilistic segmenter, and its bootstrap
ensemble gives an epistemic-uncertainty signal analogous to MC dropout.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier

from radagent.models.base import N_CLASSES, Segmenter

FEATURE_NAMES = [
    "prev", "center", "next", "mean_s1", "mean_s2", "mean_s4", "mean_s8", "std_s2",
    "grad_s1", "contrast_s4", "dz_abs", "row", "col",
    "median_21", "center_minus_median", "mean_s2_minus_median", "min_5", "mean_s2_minus_s8", "mean_3d",
    "log_s1.5", "log_s3", "log_s5", "log3d_s2.5",
]


def slice_features(stack_u8: np.ndarray) -> np.ndarray:
    """(3, S, S) uint8 -> (S*S, F) float32 feature matrix."""
    prev, cur, nxt = (stack_u8[i].astype(np.float32) for i in range(3))
    S0, S1 = cur.shape
    m1 = ndi.gaussian_filter(cur, 1.0)
    m2 = ndi.gaussian_filter(cur, 2.0)
    m4 = ndi.gaussian_filter(cur, 4.0)
    m8 = ndi.gaussian_filter(cur, 8.0)
    sq2 = ndi.gaussian_filter(cur * cur, 2.0)
    std2 = np.sqrt(np.maximum(sq2 - m2 * m2, 0.0))
    gx = ndi.sobel(m1, axis=0)
    gy = ndi.sobel(m1, axis=1)
    grad = np.hypot(gx, gy)
    dz = np.abs(prev - nxt)
    rows, cols = np.meshgrid(np.linspace(0, 1, S0, dtype=np.float32), np.linspace(0, 1, S1, dtype=np.float32),
                             indexing="ij")
    k = max(5, int(round(21 * S0 / 128)) | 1)                     # ~55 mm neighbourhood
    med = ndi.median_filter(cur, size=k)                            # robust organ-level background
    mn = ndi.minimum_filter(m1, size=5)
    m3d = ndi.gaussian_filter((prev + cur + nxt) / 3.0, 1.5)        # through-plane averaging
    # scale-normalised Laplacian-of-Gaussian: blob detectors for round lesions of ~3-15 px
    logs = [s_ * s_ * ndi.gaussian_laplace(cur, s_) for s_ in (1.5, 3.0, 5.0)]
    log3d = 6.25 * ndi.gaussian_laplace((prev + cur + nxt) / 3.0, 2.5)
    feats = np.stack([prev, cur, nxt, m1, m2, m4, m8, std2, grad, cur - m4, dz, rows, cols,
                      med, cur - med, m2 - med, mn, m2 - m8, m3d, *logs, log3d], axis=-1)
    return feats.reshape(-1, len(FEATURE_NAMES))


class VoxelGBMSegmenter(Segmenter):
    name = "voxel_gbm"
    version = "1.0"

    def __init__(self, members: list | None = None, img_size: int = 128, tumour_prior_scale: float = 1.0):
        self.members = members or []
        self.img_size = img_size
        # prior-shift correction: tumour pixels are over-sampled for training, so their
        # posterior is scaled back (chosen on the validation split) and renormalised
        self.tumour_prior_scale = tumour_prior_scale

    @property
    def n_passes(self) -> int:
        return max(1, len(self.members))

    # ------------------------------------------------------------------ training
    @staticmethod
    def sample_pixels(img_u8: np.ndarray, lab: np.ndarray, rng: np.random.Generator, budget: int,
                      z_keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stratified pixel sample of one case: tumour-, liver- and background-rich."""
        from radagent.preprocess import stacks_25d

        zs, rs, cs = [], [], []
        quotas = {2: int(0.25 * budget), 1: int(0.35 * budget)}
        for cls, q in quotas.items():
            idx = np.argwhere(lab[z_keep] == cls)
            if len(idx):
                pick = idx[rng.integers(len(idx), size=min(q, len(idx) * 3))]
                zs.append(z_keep[pick[:, 0]]); rs.append(pick[:, 1]); cs.append(pick[:, 2])
        body = np.argwhere((lab[z_keep] == 0) & (img_u8[z_keep] > 0))
        n_bg = budget - sum(len(z) for z in zs)
        if len(body) and n_bg > 0:
            pick = body[rng.integers(len(body), size=n_bg)]
            zs.append(z_keep[pick[:, 0]]); rs.append(pick[:, 1]); cs.append(pick[:, 2])
        Z = np.concatenate(zs); R = np.concatenate(rs); C = np.concatenate(cs)
        X, y = [], []
        S = img_u8.shape[1]
        for z in np.unique(Z):
            sel = Z == z
            f = slice_features(stacks_25d(img_u8, [int(z)])[0])
            flat = R[sel] * S + C[sel]
            X.append(f[flat]); y.append(lab[z][R[sel], C[sel]])
        return np.concatenate(X), np.concatenate(y).astype(np.int64)

    def fit(self, cases: list[dict], gbm_cfg: dict, seed: int = 0, verbose: bool = True) -> "VoxelGBMSegmenter":
        """cases: list of {img: (Z,S,S) uint8, lab: (Z,S,S) uint8, z_keep: idx array}."""
        rng = np.random.default_rng(seed)
        per_case = []
        t0 = time.time()
        for case in cases:
            per_case.append(self.sample_pixels(case["img"], case["lab"], rng, gbm_cfg["pixels_per_case"], case["z_keep"]))
        if verbose:
            print(f"[voxel_gbm] sampled {sum(len(y) for _, y in per_case):,} pixels from {len(cases)} cases "
                  f"in {time.time() - t0:.1f}s")
        self.members = []
        for m in range(gbm_cfg["n_members"]):
            boot = rng.integers(len(per_case), size=len(per_case))      # bootstrap over *patients*
            X = np.concatenate([per_case[i][0] for i in boot])
            y = np.concatenate([per_case[i][1] for i in boot])
            clf = HistGradientBoostingClassifier(max_iter=gbm_cfg["max_iter"], learning_rate=gbm_cfg["learning_rate"],
                                                 max_leaf_nodes=gbm_cfg["max_leaf_nodes"], random_state=seed + m,
                                                 early_stopping=False)
            t0 = time.time()
            clf.fit(X, y)
            self.members.append(clf)
            if verbose:
                print(f"[voxel_gbm] member {m + 1}/{gbm_cfg['n_members']} trained on {len(y):,} pixels "
                      f"in {time.time() - t0:.1f}s")
        return self

    # ----------------------------------------------------------------- inference
    def predict_batch(self, stacks_u8: np.ndarray, passes: int | None = None, roi: tuple | None = None,
                      coarse: bool = False) -> np.ndarray:
        P = self.n_passes if passes is None else max(1, min(passes, self.n_passes))
        B, _, S0, S1 = stacks_u8.shape
        out = np.zeros((P, B, N_CLASSES, S0, S1), dtype=np.float32)
        out[:, :, 0] = 1.0                                     # outside the body / ROI = background
        region = np.zeros((S0, S1), dtype=bool)
        if roi is None:
            region[:] = True
        else:
            r0, r1, c0, c1 = roi
            region[r0:r1, c0:c1] = True
        if coarse:
            grid = np.zeros((S0, S1), dtype=bool)
            grid[::2, ::2] = True
            region &= grid
        for b in range(B):
            sel = (region & (ndi.maximum_filter(stacks_u8[b, 1], size=3) > 0)).ravel()
            if not sel.any():
                continue
            f = slice_features(stacks_u8[b])[sel]
            for p in range(P):
                proba = self.members[p].predict_proba(f)
                full = np.zeros((S0 * S1, N_CLASSES), dtype=np.float32)
                full[:, 0] = 1.0
                full[sel] = 0.0
                for k, c in enumerate(self.members[p].classes_):
                    full[sel, int(c)] = proba[:, k]
                if self.tumour_prior_scale != 1.0:
                    full[sel, 2] *= self.tumour_prior_scale
                    full[sel] /= full[sel].sum(axis=1, keepdims=True)
                img = full.T.reshape(N_CLASSES, S0, S1)
                if coarse:   # nearest-neighbour fill of the skipped pixels
                    img = np.repeat(np.repeat(img[:, ::2, ::2], 2, axis=1), 2, axis=2)[:, :S0, :S1]
                out[p, b] = img
        return out

    # -------------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"members": self.members, "img_size": self.img_size, "version": self.version,
                         "tumour_prior_scale": self.tumour_prior_scale}, f)

    @classmethod
    def load(cls, path: str | Path) -> "VoxelGBMSegmenter":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(members=d["members"], img_size=d["img_size"], tumour_prior_scale=d.get("tumour_prior_scale", 1.0))
