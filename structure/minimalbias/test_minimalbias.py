import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.special import eval_legendre


MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))

import minimalbias
from minimalbias import minimalbias_class, _get_wp_aniso, _simpson_weights


def get_wp_aniso_reference(r, xi0, xi2, xi4, beta, rp_in, pimax):
    dlnrp_min = 0.01
    if np.log10(rp_in[1] / rp_in[0]) < dlnrp_min:
        rp = 10**np.arange(np.log10(rp_in.min()), np.log10(rp_in.max()),
                           dlnrp_min)
        interpolate = True
    else:
        rp = rp_in
        interpolate = False

    rpi = np.logspace(-3, np.log10(pimax), 300)
    rp2, rpi2 = np.meshgrid(rp, rpi)
    s = np.sqrt(rp2**2 + rpi2**2)
    mu = rpi2 / s
    xi0s = (1 + 2/3*beta + 1/5*beta**2)*ius(r, xi0)(s)
    xi2s = (4/3*beta + 4/7*beta**2)*ius(r, xi2)(s)
    xi4s = 8/35*beta**2*ius(r, xi4)(s)
    integrand = 2*(xi0s + xi2s*eval_legendre(2, mu)
                   + xi4s*eval_legendre(4, mu))
    result = integrate.simpson(integrand, x=rpi, axis=0)
    if interpolate:
        result = ius(rp, result)(rp_in)
    return result


def make_multipoles():
    r = np.logspace(-3, 3, 500)
    xi0 = np.exp(-r/80.0)/(1.0 + r)**1.1
    xi2 = -0.3*np.exp(-r/60.0)/(1.0 + r)**0.9
    xi4 = 0.08*np.exp(-r/45.0)/(1.0 + r)**0.7
    return r, xi0, xi2, xi4


def make_model():
    z = np.array([0.0, 0.3, 0.6, 1.0])
    k = np.logspace(-4, 2, 256)
    shape = k**0.96*np.exp(-k/8.0)
    pklin = np.array([(1.0 + zi)**-2*shape for zi in z])
    pknlin = pklin*(1.0 + (k[None, :]/0.7)**1.2)
    model = minimalbias_class({"do_Kaiser": True, "verbose": False})
    model.set_pk_lin_data(z, k, pklin)
    model.set_pk_nlin_data(z, k, pknlin)
    model.set_z2f(z, np.array([0.51, 0.63, 0.72, 0.82]))
    model.set_param(0.3, 1.8)
    return model


def test_simpson_weights_match_scipy_to_numerical_precision():
    x = np.logspace(-3, 2, 300)
    y = np.sin(x)/(1.0 + x)
    assert_allclose(
        np.dot(_simpson_weights(x), y),
        integrate.simpson(y, x=x),
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_numba_isotropic_projection_matches_reference():
    model = make_model()
    r, xi0, _, _ = make_multipoles()
    rpi = np.logspace(-3, np.log10(87.0), 300)
    rp2, rpi2 = np.meshgrid(r, rpi)
    s = np.sqrt(rp2**2 + rpi2**2)
    expected = 2*integrate.simpson(ius(r, xi0, ext=3)(s), x=rpi, axis=0)
    actual = model._project_xi_pimax(r, xi0, 87.0)
    assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-14)


def test_numba_anisotropic_projection_matches_dense_reference():
    r, xi0, xi2, xi4 = make_multipoles()
    rp = np.logspace(-2, 2, 500)
    expected = get_wp_aniso_reference(r, xi0, xi2, xi4, 0.42, rp, 100.0)
    actual = _get_wp_aniso(r, xi0, xi2, xi4, 0.42, rp, 100.0)
    assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-14)


def test_numba_anisotropic_projection_matches_sparse_reference():
    r, xi0, xi2, xi4 = make_multipoles()
    rp = np.logspace(-1, 1.7, 80)
    expected = get_wp_aniso_reference(r, xi0, xi2, xi4, 0.78, rp, 73.0)
    actual = _get_wp_aniso(r, xi0, xi2, xi4, 0.78, rp, 73.0)
    assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-14)


def test_get_wp_matches_original_projection_path():
    model = make_model()
    rp = np.logspace(-0.5, 1.6, 24)
    with patch.object(minimalbias, "_get_wp_aniso",
                      side_effect=get_wp_aniso_reference):
        expected = model.get_wp(0.45, rp, 0.08, 92.0)
    actual = model.get_wp(0.45, rp, 0.08, 92.0)
    assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-14)
