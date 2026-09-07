"""
Author: Sunao Sugiyama
Last edit: 2023.05.17

This computes the magnification bias on g-g lensing in physical scale, i.e. dSigma.

Code reviewed by Tianqing Zhang in Oct 2023
"""
import numpy as np
from numba import njit
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from fftlog import hankel, fftlog

class magnification_class:
    def __init__(self, config = None):
        self.config = {'verbose':True}
        self.config.update(config or {})
    
    def set_param(self, alpha, Omm):
        """
        alpha (float): magnification bias parameter
        """
        self.param = {'alpha': alpha, 'Omm':Omm}
        
    def set_pk_nlin_data(self, z, k, pknonlin):
        """
        set nonlinear matter power spectrum
        """
        self.pk_nlin_data = [z, k, pknonlin]
        
    def set_z2chi(self, z, chi):
        """
        set relation between redshift z and comoving distance chi.
        """
        self.z2chi = ius(z, chi)
        
    def set_nz_source(self, z, nz):
        """
        Here, input redshfit distribution should already include redshift bias effect.
        """
        dtype = np.dtype([('z', z.dtype), ('nz', nz.dtype)])
        self.nzs = np.empty(z.size, dtype)
        self.nzs['z']  = z
        self.nzs['nz'] = nz
        
    def set_nz_lens(self, z, nz):
        dtype = np.dtype([('z', z.dtype), ('nz', nz.dtype)])
        self.nzl = np.empty(z.size, dtype)
        self.nzl['z']  = z
        self.nzl['nz'] = nz
    
    def _kzpktable2pltable(self, z, k, pktable, l):
        """
        Convert P(k_i, z_j) -> P(l_i/chi_j, z_j)
        """
        pltable = np.empty((l.size, z.size))
        chi = self.z2chi(z) # [Mpc/h]
        for i,_z in enumerate(z):
            pltable[:,i] = log_extrap_func(k, pktable[i, :])((l+0.5)/chi[i])
        return pltable
    
    def _get_clSigmacrit_mag(self):
                
        window = get_Sigmacr_Cl_window(self.nzs['z'], self.nzs['nz'], 
                                       self.nzl['z'], self.nzl['nz'], 
                                       self.z2chi) # [h/Mpc]
        H0 = 100/299792.4580 # [h/Mpc]
        Om= self.param['Omm']
        rhocrit = 2.77536627e11 # h^2 M_\odot/Mpc^3
        prefactor = (2.0/3.0*rhocrit/H0**2)*(3.0/2.0*H0**2*Om)**2 # [h^4Msun/Mpc^5]
        
        l = np.logspace(0, 5, 1000)
        sel = self.pk_nlin_data[0] > 0
        pltable = self._kzpktable2pltable(self.pk_nlin_data[0][sel], 
                                          self.pk_nlin_data[1], 
                                          self.pk_nlin_data[2][sel, :], l)
        
        chi= self.z2chi(self.pk_nlin_data[0][sel]) # [Mpc/h]
        integrand   = window(chi)*pltable
        clSigmacrit = integrate.simpson(integrand.T, chi, axis=0)
        clSigmacrit*= prefactor/1e12 # hMsun/pc^2
        return l, clSigmacrit
    
    def get_ds_mag(self, z, rp, dlnrp, N_extrap_low=None, N_extrap_high=None):
        """
        rp     (array): array of radial bins in unit of Mpc/h, need to be corrected for the measurement effect.
        """
        l, clSigmacrit = self._get_clSigmacrit_mag()
        # print(self.nzs['z'])
        # print(clSigmacrit)
        if N_extrap_low is None:
            N_extrap_low = l.size
        if N_extrap_high is None:
            N_extrap_high = l.size
        if np.any(clSigmacrit < 0):
            raise "some of the clSigmacrit in magnification bias have negative (nonphysical) values."
        myhankel = hankel(l, l**2*clSigmacrit, 1.01, 
                          N_extrap_low=N_extrap_low,
                          N_extrap_high=N_extrap_high,
                          c_window_width=0.25)
    
        if dlnrp > 0:
            D = 2 # dimension
            t, ds_mag = myhankel.hankel_binave(2, dlnrp, D)
        else:
            t, ds_mag = myhankel.hankel(2)
        ds_mag /= 2.0*np.pi

        chil_rep = self.z2chi(z) # [Mpc/h]
        ds_mag = ius(t, ds_mag, ext=3)(rp/chil_rep)
        ds_mag*= 2*(self.param['alpha']-1)
        
        return ds_mag


@njit(cache=True)
def _compute_window_coefficients_numba(zs, nzs, zl, nzl, chis, chil):
    """Integrate the three window coefficients over source redshift."""
    nlens = zl.size
    c0_by_lens = np.zeros(nlens)
    c1_by_lens = np.zeros(nlens)
    c2_by_lens = np.zeros(nlens)

    for ilens in range(nlens):
        if chil[ilens] == 0.0 or 1.0 + zl[ilens] == 0.0:
            continue

        for isource in range(zs.size):
            if zl[ilens] >= zs[isource]:
                continue

            delta_chi = chis[isource] - chil[ilens]
            if np.abs(delta_chi) < 1.0e-3:
                continue

            common = (
                nzs[isource]
                * nzl[ilens]
                / (1.0 + zl[ilens])
                / chil[ilens]**2
                / delta_chi
            )
            c0_by_lens[ilens] += common * chil[ilens] * chis[isource]
            c1_by_lens[ilens] += common * (chil[ilens] + chis[isource])
            c2_by_lens[ilens] += common

    return c0_by_lens, c1_by_lens, c2_by_lens


@njit(cache=True)
def _evaluate_window_numba(
    z,
    chi,
    zl,
    c0_by_lens,
    c1_by_lens,
    c2_by_lens,
    integration_weight,
):
    """Evaluate the magnification window on its integration grid."""
    window = np.zeros(z.size)

    for iz in range(z.size):
        c0 = 0.0
        c1 = 0.0
        c2 = 0.0

        for ilens in range(zl.size):
            if zl[ilens] > z[iz]:
                c0 += c0_by_lens[ilens]
                c1 += c1_by_lens[ilens]
                c2 += c2_by_lens[ilens]

        window[iz] = (
            c0 - c1 * chi[iz] + c2 * chi[iz]**2
        ) * integration_weight * (1.0 + z[iz])**2

    return window


def _prepare_redshift_distribution(z_in, nz_in, name, check_spacing):
    """Filter and normalize an n(z) for the compiled window kernels."""
    if np.isscalar(z_in) and np.isscalar(nz_in):
        return (
            np.ascontiguousarray([float(z_in)], dtype=np.float64),
            np.ascontiguousarray([1.0], dtype=np.float64),
            1.0,
        )

    z_in = np.asarray(z_in)
    nz_in = np.asarray(nz_in)
    if z_in.ndim != 1 or nz_in.ndim != 1 or z_in.size != nz_in.size:
        raise ValueError(f"{name} z and n(z) must be one-dimensional arrays of equal length")

    sel = nz_in > 0
    z = np.ascontiguousarray(z_in[sel], dtype=np.float64)
    nz = np.ascontiguousarray(nz_in[sel], dtype=np.float64)
    if z.size < 2:
        raise ValueError(f"{name} n(z) needs at least two positive samples")

    dz = np.diff(z)[0]
    if check_spacing and not np.all(np.isclose(np.diff(z), dz)):
        raise ValueError(f"{name} redshift grid must be uniformly spaced")

    normalization = np.sum(nz) * dz
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError(f"{name} n(z) has invalid normalization")

    return z, np.ascontiguousarray(nz / normalization), float(dz)


def get_Sigmacr_Cl_window(zs_in, nzs_in, zl_in, nzl_in, z2chi):
    """Construct the magnification window with compiled numerical kernels."""
    zl, nzl, dzl = _prepare_redshift_distribution(
        zl_in, nzl_in, "lens", check_spacing=True
    )
    zs, nzs, dzs = _prepare_redshift_distribution(
        zs_in, nzs_in, "source", check_spacing=False
    )

    chil = np.ascontiguousarray(z2chi(zl), dtype=np.float64)
    chis = np.ascontiguousarray(z2chi(zs), dtype=np.float64)
    if not np.all(np.isfinite(chil)) or not np.all(np.isfinite(chis)):
        raise ValueError("Comoving-distance arrays contain non-finite values")

    c0_by_lens, c1_by_lens, c2_by_lens = (
        _compute_window_coefficients_numba(zs, nzs, zl, nzl, chis, chil)
    )

    positive_lenses = zl > 0
    if not np.any(positive_lenses):
        raise ValueError("Lens n(z) has no positive-redshift samples")

    z = np.linspace(0.0, zl[positive_lenses].max() * 1.01, 100)
    chi = np.ascontiguousarray(z2chi(z), dtype=np.float64)
    if not np.all(np.isfinite(chi)):
        raise ValueError("Window comoving-distance grid contains non-finite values")

    window = _evaluate_window_numba(
        z,
        chi,
        zl,
        c0_by_lens,
        c1_by_lens,
        c2_by_lens,
        dzl * dzs,
    )
    return ius(chi, window, ext=1)  # [h/Mpc]
    
def log_extrap_func(x,y):
    def func(x_new):
        if isinstance(x_new, (int, float)):
            x_new = np.atleast_1d(x_new)
        ans = np.zeros(x_new.size)
        sel = x_new < x.min()
        if np.sum(sel):
            ans[sel] = np.exp( np.log(y[1]/y[0])/np.log(x[1]/x[0]) * np.log(x_new[sel]/x[0]) + np.log(y[0]) )
        sel = x.max() < x_new
        if np.sum(sel):
            ans[sel] = np.exp( np.log(y[-2]/y[-1])/np.log(x[-2]/x[-1]) * np.log(x_new[sel]/x[-1]) + np.log(y[-1]) )
        sel = np.logical_and(x.min()<= x_new, x_new <= x.max())
        ans[sel] = 10**ius(np.log10(x),np.log10(y))(np.log10(x_new[sel]))
        return ans
    return func
    