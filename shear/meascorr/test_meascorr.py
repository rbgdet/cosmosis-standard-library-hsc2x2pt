import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose


MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))

from meascorr import dSigma_meascorr_class, sigcrit_prefactor


def make_correction():
    correction = dSigma_meascorr_class({
        "Omm": 0.3,
        "w0": -1.0,
        "name": "test",
        "bin0": 0,
        "bin1": 0,
    })
    correction.set_zl(np.array([0.2, 0.35, 0.5]))
    correction.set_zs(np.array([0.1, 0.4, 0.8, 1.2]))
    correction.set_sumwlssigcritinvPz(np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
        [0.2, 0.4, 0.1, 0.3],
    ]))
    return correction


def reference_accumulation(correction, chi_zl, chi_zs):
    result = 0.0
    for j, source_distance in enumerate(chi_zs):
        if source_distance <= 0.0:
            continue
        sigma_cr_inv = ((1.0 + correction.zl) * chi_zl
                        * (1.0 - chi_zl / source_distance)
                        / sigcrit_prefactor)
        sigma_cr_inv[sigma_cr_inv < 0.0] = 0.0
        result += np.sum(
            sigma_cr_inv * correction.sumwlssigcritinvPz[:, j])
    return result


def test_numba_accumulation_matches_original_loop():
    correction = make_correction()
    chi_zl, chi_zs = correction._get_distances(0.03, 0.27, -0.9)
    expected = reference_accumulation(correction, chi_zl, chi_zs)
    actual = correction._get_dSigma_corr_numerator(
        0.03, 0.27, -0.9, chi_zl=chi_zl, chi_zs=chi_zs)
    assert_allclose(actual, expected, rtol=2.0e-15, atol=0.0)


def test_reference_is_cached_and_fiducial_correction_is_unity():
    correction = make_correction()
    denominator = correction.prepare_reference()
    assert denominator == correction._reference_denominator
    with patch.object(
            correction,
            "_get_dSigma_corr_numerator",
            wraps=correction._get_dSigma_corr_numerator) as numerator:
        assert_allclose(
            correction.get_corr_factor(0.0, 0.3, -1.0),
            1.0,
            rtol=2.0e-15,
            atol=0.0,
        )
        # One sampled numerator call; the fixed denominator was not repeated.
        assert numerator.call_count == 1


def test_mutating_input_invalidates_reference_cache():
    correction = make_correction()
    correction.prepare_reference()
    correction.set_zs(correction.zs.copy())
    assert correction._reference_denominator is None
