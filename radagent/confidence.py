"""Case-level confidence and uncertainty.

``tumour_probability``  calibrated probability that the case contains a tumour.
``uncertainty``         predictive entropy of that calibrated probability, in bits (0 = certain,
                        1 = p = 0.5). For a calibrated predictor the expected error of the
                        suggestion is min(p, 1 - p), a monotone function of this entropy, which
                        makes it the natural score for selective prediction / deferral.

The stochastic passes (flip-TTA x MC-dropout, or ensemble members) give the usual
decomposition of the total uncertainty of the pass-averaged prediction
(Depeweg et al. 2018, "BALD"):

    H[mean_k p_k]  =  mean_k H[p_k]   +   I                (bits)
     total            expected entropy     mutual information
                      (aleatoric)           = model disagreement (epistemic)

Both parts are reported so that a reviewer can see *why* a case is uncertain.
"""
from __future__ import annotations

import numpy as np

from radagent.calibration import Calibrator, binary_entropy


def uncertainty_level(u: float) -> str:
    return "low" if u < 0.5 else ("medium" if u < 0.85 else "high")


def case_confidence(p_raw: float, pass_scores: list, calibrator: Calibrator, cfg: dict) -> dict:
    p_cal = float(calibrator(p_raw))
    pass_cal = np.asarray(calibrator(np.asarray(pass_scores, dtype=np.float64))) if len(pass_scores) else np.array([p_cal])
    total = float(binary_entropy(p_cal))
    expected = float(np.mean(binary_entropy(pass_cal)))
    mi = float(max(0.0, binary_entropy(float(pass_cal.mean())) - expected))
    return {
        "tumour_probability_raw": round(float(p_raw), 5),
        "tumour_probability": round(p_cal, 5),
        "pass_probabilities": [round(float(x), 4) for x in pass_cal.tolist()],
        "pass_std": round(float(pass_cal.std()), 5),
        "uncertainty": round(total, 5),
        "uncertainty_level": uncertainty_level(total),
        "uncertainty_components": {"predictive_entropy": round(total, 5),
                                   "expected_entropy": round(expected, 5),
                                   "mutual_information": round(mi, 5)},
        "calibrator": calibrator.describe(),
    }
