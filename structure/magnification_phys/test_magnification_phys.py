import sys
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy.interpolate import InterpolatedUnivariateSpline as ius


MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))

from magnification_phys import get_Sigmacr_Cl_window


def get_window_reference(zs_in, nzs_in, zl_in, nzl_in, z2chi):
    """Original implementation retained as a numerical test reference."""
    sel = nzl_in > 0
    zl = zl_in[sel]
    nzl = nzl_in[sel]
    dzl = np.diff(zl)[0]
    assert np.all(np.isclose(np.diff(zl), dzl))

    sel = nzs_in > 0
    zs = zs_in[sel]
    nzs = nzs_in[sel]
    dzs = np.diff(zs)[0]

    nzs_normed = nzs / (np.sum(nzs) * dzs)
    nzl_normed = nzl / (np.sum(nzl) * dzl)

    chil = z2chi(zl)
    chis = z2chi(zs)

    c0, c1, c2 = [], [], []
    for _zs, _nzs, _chis in zip(zs, nzs_normed, chis):
        _c0, _c1, _c2 = [], [], []
        for _zl, _nzl, _chil in zip(zl, nzl_normed, chil):
            if _zl >= _zs or np.abs(_chis - _chil) < 1.0e-3:
                _c0.append(0.0)
                _c1.append(0.0)
                _c2.append(0.0)
                continue

            common = _nzs * _nzl / (1.0 + _zl) / _chil**2 / (_chis - _chil)
            _c0.append(common * _chil * _chis)
            _c1.append(common * (_chil + _chis))
            _c2.append(common)

        c0.append(_c0)
        c1.append(_c1)
        c2.append(_c2)

    c0 = np.asarray(c0)
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    zl_mat, zs_mat = np.meshgrid(zl, zs)

    z = np.linspace(0.0, zl[zl > 0].max() * 1.01, 100)
    chi = z2chi(z)
    c0_chi, c1_chi, c2_chi = [], [], []
    for _z in z:
        mask = np.logical_and(zl_mat > _z, np.logical_and(zs_mat > _z, zs_mat > zl_mat))
        c0_chi.append(np.sum(c0[mask]) * dzl * dzs)
        c1_chi.append(np.sum(c1[mask]) * dzl * dzs)
        c2_chi.append(np.sum(c2[mask]) * dzl * dzs)

    c0_chi = np.asarray(c0_chi)
    c1_chi = np.asarray(c1_chi)
    c2_chi = np.asarray(c2_chi)
    window = (c0_chi - c1_chi * chi + c2_chi * chi**2) * (1.0 + z)**2
    return ius(chi, window, ext=1)


def make_z2chi():
    z = np.linspace(0.0, 5.0, 1001)
    chi = 3000.0 * z / (1.0 + 0.2 * z)
    return ius(z, chi)


def compare_windows(zs, nzs, zl, nzl):
    z2chi = make_z2chi()
    reference = get_window_reference(zs, nzs, zl, nzl, z2chi)
    optimized = get_Sigmacr_Cl_window(zs, nzs, zl, nzl, z2chi)
    chi_test = np.linspace(0.0, z2chi(zl[nzl > 0]).max() * 1.02, 300)
    assert_allclose(optimized(chi_test), reference(chi_test), rtol=1.0e-12, atol=1.0e-14)


def test_numba_window_matches_reference_for_smooth_distributions():
    zl = np.linspace(0.0, 0.9, 181)
    zs = np.linspace(0.0, 3.0, 301)
    nzl = np.exp(-0.5 * ((zl - 0.48) / 0.09)**2)
    nzs = np.exp(-0.5 * ((zs - 1.0) / 0.28)**2)
    nzl[[0, -1]] = 0.0
    nzs[[0, -1]] = 0.0
    compare_windows(zs, nzs, zl, nzl)


def test_numba_window_matches_reference_with_zero_tails_and_overlap():
    zl = np.linspace(0.0, 0.8, 161)
    zs = np.linspace(0.0, 1.5, 301)
    nzl = np.where((zl >= 0.2) & (zl <= 0.7), 1.0 + zl, 0.0)
    nzs = np.where((zs >= 0.1) & (zs <= 1.4), np.exp(-zs), 0.0)
    compare_windows(zs, nzs, zl, nzl)
