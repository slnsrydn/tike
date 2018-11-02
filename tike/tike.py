#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2017-2018, UChicago Argonne, LLC. All rights reserved.    #
#                                                                         #
# Copyright 2018. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

"""Define the highest level functions for solving ptycho-tomography problem."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import logging
import tike.tomo
import tike.ptycho
from tike.constants import *
import dxchange
import tomopy
import matplotlib.pyplot as plt

folder = 'tmp/lego-joint-test1'
print (folder)

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['admm',
           ]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _combined_interface(obj, obj_min,
                        data, data_min,
                        probe, theta, v, h,
                        **kwargs):
    """Define an interface that all functions in this module match."""
    assert np.all(obj_size > 0), "Detector dimensions must be > 0."
    assert np.all(probe_size > 0), "Probe dimensions must be > 0."
    assert np.all(detector_size > 0), "Detector dimensions must be > 0."
    assert theta.size == h.size == v.size == \
        detector_grid.shape[0] == probe_grid.shape[0], \
        "The size of theta, h, v must be the same as the number of probes."
    logger.info(" _ptycho_interface says {}".format("Hello, World!"))
    return None


def admm(obj=None, voxelsize=None,
         data=None,
         probe=None, theta=None, h=None, v=None, energy=None,
         niter=1, rho=0.5, gamma=0.25,
         **kwargs):
    """Solve using the Alternating Direction Method of Multipliers (ADMM).

    Parameters
    ----------
    obj : (Z, X, Y, P) :py:class:`numpy.array` float
        The initial guess for the reconstruction.
    obj_min : (3, ) float
        The min corner (z, x, y) of the `obj`.
    data : (M, H, V) :py:class:`numpy.array` float
        An array of detector intensities for each of the `M` probes. The
        grid of each detector is `H` pixels wide (the horizontal
        direction) and `V` pixels tall (the vertical direction).
    data_min : (2, ) float [p]
        The min corner (h, v) of the data in the global coordinate system.
    probe : (H, V) :py:class:`numpy.array` complex
        A single illumination function for the all probes.
    energy : float [keV]
        The energy of the probe
    algorithms : (2, ) string
        The names of the pytchography and tomography reconstruction algorithms.
    kwargs :
        Any keyword arguments for the pytchography and tomography
        reconstruction algorithms.

    """
    Z, X, Y = obj.shape[0:3]
    T = theta.size
    x = obj
    psi = np.ones([T, Z, Y], dtype=obj.dtype)
    hobj = 1 + 0j  # np.ones([T, Z, Y], dtype=obj.dtype)
    lamda = 0j
    cp = np.zeros((niter,))
    cl = np.zeros((niter,))
    co = np.zeros((niter,))
    
    res_admm = list()
    dualres1_admm = list()
    dualres2_admm = list()
    dualres3_admm = list()
    convpsi_admm = list()
    convallx_admm = list()
    for i in range(niter):
        # Ptychography.
        psi, convpsi, dualres3 = tike.ptycho.reconstruct(data=data,
                                      probe=probe, v=v, h=h,
                                      psi=psi,
                                      algorithm='grad',
                                      niter=3, rho=rho, gamma=gamma, reg=hobj,
                                      lamda=lamda, **kwargs)
        dxchange.write_tiff(np.real(psi[0]).astype('float32'), folder + '/psi-amplitude/psi-amplitude')
        dxchange.write_tiff(np.imag(psi[0]).astype('float32'), folder + '/psi-phase/psi-phase')
        dxchange.write_tiff(np.abs((psi + lamda/rho)[0]).astype('float32'), folder + '/psilamd-amplitude/psilamd-amplitude')
        dxchange.write_tiff(np.angle((psi + lamda/rho)[0]).astype('float32'), folder + '/psilamd-phase/psilamd-phase')
        cp[i] = np.sqrt(np.sum(np.power(np.abs(hobj-psi), 2)))
        convpsi_admm.append(convpsi)
        dualres3_admm.append(dualres3)
        # Tomography.
        phi = -1j / wavenumber(energy) * np.log(psi + lamda / rho) / voxelsize
        new_x, convx = tike.tomo.reconstruct(obj=x,
                                  theta=theta,
                                  line_integrals=phi,
                                  algorithm='grad', reg_par=-1,
                                  niter=1, **kwargs)
        np.save("x-vals/x{:03d}.npy".format(i), new_x)
        co[i] = np.sqrt(np.sum(np.power(np.abs(x- new_x), 2)))
        convallx_admm.extend(convx)
        dxchange.write_tiff(new_x.imag[obj.shape[0] // 2 - 1].astype('float32'), folder + '/beta/beta')
        dxchange.write_tiff(new_x.real[obj.shape[0] // 2 - 1].astype('float32'), folder + '/delta/delta')
        dxchange.write_tiff(new_x.imag.astype('float32'), folder + '/beta-full/beta')
        dxchange.write_tiff(new_x.real.astype('float32'), folder + '/delta-full/delta')
    
        # Lambda update.
        line_integrals = tike.tomo.forward(obj=new_x, theta=theta) * voxelsize
        np.save("line_int-vals/line_integrals{:03d}.npy".format(i), line_integrals)
        new_hobj = np.exp(1j * wavenumber(energy) * line_integrals)
        dxchange.write_tiff(np.abs(new_hobj[0]).astype('float32'), folder +'/hobj-amplitude/hobj-amplitude')
        dxchange.write_tiff(np.angle(new_hobj[0]).astype('float32'), folder +'/hobj-phase/hobj-phase') 
        dualres1 = rho * np.sqrt(np.sum(np.power(np.abs(hobj- new_hobj), 2)))
        dualres1_admm.append(dualres1)
    
        new_lamda = lamda + rho * (psi - hobj)
        np.save("lamda-vals/lamda{:03d}.npy".format(i), new_lamda)
        res = np.sqrt(np.sum(np.power(np.abs(np.mean(psi - new_hobj, axis = 0)), 2)))
        res_admm.append(res)
        cl[i] = np.sqrt(np.sum(np.power(np.abs(lamda-new_lamda), 2)))
        lamda = new_lamda.copy() 
        dualres2 =  np.zeros(obj.shape, dtype='complex')
        for m in range(0, T):
            gradhobj = 1j * wavenumber(energy) * (tomopy.recon(np.cos(wavenumber(energy) * line_integrals), theta[m], algorithm="fbp") +
                             1j * tomopy.recon(np.sin(wavenumber(energy) * line_integrals), theta[m], algorithm="fbp"))
            dualres2 += -gradhobj * new_lamda[m].reshape([*new_lamda[m].shape, 1]) 
           
        dualres2_admm.append(np.sqrt(np.sum(np.power(np.abs(dualres2), 2))))
        x = new_x.copy()
        hobj = new_hobj.copy()
        print (i, cp[i], co[i], cl[i])
        
    np.save('res_admmPR1T1theta360prb6rho1gamma01.npy', res_admm)
    np.save('dualres_admmPR1T1theta360prb6rho1gamma01.npy', dualres1_admm)
    np.save('dualres_admmPR1T1theta360prb6rho1gamma01.npy', dualres3_admm)
    np.save('convallx_admmPR1T1theta360prb6rho1gamma01.npy', convallx_admm)
    np.save('convallpsi_admmPR1T1theta360prb6rho1gamma01.npy', convpsi_admm)
    
    plt.figure()
    plt.semilogy(res_admm)
    plt.xlabel('ADMM iter')
    plt.ylabel('residual')
    plt.savefig('res_admmPR1T1theta360prb6rho1gamma01.png')

    plt.figure()
    plt.semilogy(dualres1_admm)
    plt.xlabel('ADMM iter')
    plt.ylabel('dual residual')
    plt.savefig('dualres_admmPR1T1theta360prb6rho1gamma01.png')
    
    plt.figure()
    plt.semilogy(dualres2_admm)
    plt.xlabel('ADMM iter')
    plt.ylabel('dual residual')
    plt.savefig('dualres_admmPR1T1theta360prb6rho1gamma01.png')    
    
    plt.figure()
    plt.semilogy(dualres3_admm)
    plt.xlabel('ADMM iter')
    plt.ylabel('dual residual')
    plt.savefig('dualres_admmPR1T1theta360prb6rho1gamma01.png')
    
    plt.figure()
    plt.semilogy(convallx_admm)
    plt.xlabel('SD and ADMM iter')
    plt.ylabel('Error of x')
    plt.savefig('convallx_admmPR1T1theta360prb6rho1gamma01.png')
    return x
