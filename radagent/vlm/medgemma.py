"""MedGemma 1.5 4B (google/medgemma-1.5-4b-it) as a second reader of the CT.

Follows Google's ``high_dimensional_ct_hugging_face`` notebook: three-window RGB slices,
interleaved ``SLICE n`` labels, ``AutoProcessor`` + ``AutoModelForImageTextToText``.
Two uses per case:

* **score** - one forward step on "is there a liver tumour? answer yes or no"; the
  probability of the yes-tokens against the no-tokens at the first answer position is the
  model's score ``p_yes`` (a number we can calibrate and fuse, unlike free text);
* **describe** (optional) - a short greedy generation ending in "Final Answer: yes/no",
  shown on the website and in the report as unverified model text.

Requirements: ``pip install -r requirements-vlm.txt``, a GPU, and access to the model on
Hugging Face (accept the Health AI Developer Foundations terms on the model page and set
``HF_TOKEN``). Precision: bf16 on Ampere or newer GPUs (L4, A100); on older GPUs such as
the free-Colab T4 the weights are loaded in 4-bit with float32 compute, which avoids the
float16 overflow Gemma 3 models are known for.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from radagent.vlm.base import VLMReader
from radagent.vlm.prompts import DESCRIBE_QUERY, SCORE_QUERY, build_content

YES_WORDS = ("yes",)
NO_WORDS = ("no",)


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.exp(x - m).sum()))


def yes_probability(logits: np.ndarray, yes_ids: list[int], no_ids: list[int]) -> tuple[float, float]:
    """P(yes | yes or no) from next-token logits, and the probability mass on those tokens."""
    logits = np.asarray(logits, dtype=np.float64)
    ly, ln = _logsumexp(logits[yes_ids]), _logsumexp(logits[no_ids])
    p_yes = 1.0 / (1.0 + math.exp(ln - ly))
    mass = math.exp(_logsumexp(np.concatenate([logits[yes_ids], logits[no_ids]])) - _logsumexp(logits))
    return p_yes, mass


def single_token_ids(tokenizer, words) -> list[int]:
    """First-token ids of the word variants that encode to exactly ONE token.

    Multi-token variants are skipped: their first piece (e.g. a bare space) would be shared
    between "yes" and "no" and blur the comparison.
    """
    ids = set()
    for w in words:
        for v in (w, w.capitalize(), w.upper(), " " + w, " " + w.capitalize()):
            enc = tokenizer.encode(v, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(int(enc[0]))
    if not ids:
        raise RuntimeError(f"none of {words} is a single token for this tokenizer")
    return sorted(ids)


class MedGemmaReader(VLMReader):
    name = "medgemma"

    def __init__(self, cfg: dict):
        v = cfg["vlm"]
        self.model_id = v["model_id"]
        self.device_pref = v.get("device", "auto")
        self.dtype_pref = v.get("dtype", "auto")
        self.max_new_tokens = int(v.get("max_new_tokens", 320))
        self._model = None
        self._processor = None
        self._compute = None
        self._precision = None
        self._yes = self._no = None

    # ------------------------------------------------------------------ loading
    def _load(self):
        if self._model is not None:
            return
        import torch
        import transformers
        from packaging.version import Version

        cuda = torch.cuda.is_available() and self.device_pref != "cpu"
        mode = self.dtype_pref
        if mode == "auto":
            if cuda and torch.cuda.get_device_capability()[0] >= 8:
                mode = "bfloat16"
            elif cuda:
                try:
                    import bitsandbytes  # noqa: F401

                    mode = "4bit"
                except ImportError:
                    mode = "float16"
                    print("[medgemma] WARNING: bitsandbytes missing on a pre-Ampere GPU - using float16, "
                          "which can overflow for Gemma 3 models. pip install bitsandbytes")
            else:
                mode = "float32"
                print("[medgemma] WARNING: no GPU - MedGemma on CPU is very slow (minutes per case)")
        kwargs = {}
        if mode == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float32)
            compute = torch.float32
        else:
            compute = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[mode]
        # transformers >= 4.56 renamed torch_dtype -> dtype
        key = "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"
        kwargs[key] = compute
        if cuda:
            kwargs["device_map"] = "auto"
        token = os.environ.get("HF_TOKEN") or None
        t0 = time.time()
        try:
            self._processor = transformers.AutoProcessor.from_pretrained(self.model_id, use_fast=True, token=token)
        except Exception:  # fast image processor needs torchvision
            self._processor = transformers.AutoProcessor.from_pretrained(self.model_id, token=token)
        try:
            self._model = transformers.AutoModelForImageTextToText.from_pretrained(self.model_id, token=token, **kwargs)
        except OSError as e:
            raise RuntimeError(
                f"could not load {self.model_id}: {e}\nAccept the terms on https://huggingface.co/{self.model_id} "
                "with your Hugging Face account and set HF_TOKEN (Colab: Secrets -> HF_TOKEN).") from e
        if not cuda:
            self._model.to("cpu")
        self._model.eval()
        self._compute = compute
        self._precision = mode
        tok = self._processor.tokenizer
        self._yes, self._no = single_token_ids(tok, YES_WORDS), single_token_ids(tok, NO_WORDS)
        print(f"[medgemma] loaded {self.model_id} ({mode}) in {time.time() - t0:.0f}s; "
              f"yes ids {self._yes}, no ids {self._no}")

    def describe_model(self) -> dict:
        return {"backend": self.name, "model_id": self.model_id, "precision": self._precision or self.dtype_pref}

    # ------------------------------------------------------------------ inference
    def _inputs(self, images: list[np.ndarray], query: str):
        from PIL import Image

        items = [{"type": "image", "image": Image.fromarray(im)} for im in images]
        messages = [{"role": "user", "content": build_content(len(images), query, items)}]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        return inputs.to(self._model.device, dtype=self._compute)

    def read(self, images: list[np.ndarray], describe: bool = True) -> dict:
        import torch

        self._load()
        t0 = time.time()
        inputs = self._inputs(images, SCORE_QUERY)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=1, do_sample=False,
                                       output_logits=True, return_dict_in_generate=True)
        logits = out.logits[0][0].float().cpu().numpy()
        p_yes, mass = yes_probability(logits, self._yes, self._no)
        text = ""
        if describe:
            inputs = self._inputs(images, DESCRIBE_QUERY)
            n_in = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                gen = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            text = self._processor.decode(gen[0][n_in:], skip_special_tokens=True).strip()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"p_yes": float(p_yes), "answer_mass": float(mass), "text": text,
                "n_images": len(images), "seconds": round(time.time() - t0, 2)}
