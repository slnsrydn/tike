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

"""
Module for defining scanning trajectories.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import logging
from math import sqrt

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['scantimes',
           'sinusoid',
           'triangle',
           'sawtooth',
           'square',
           'staircase',
           'lissajous',
           'raster',
           'spiral',
           'scan3',
           'avgspeed',
           'lengths',
           'distance']


logger = logging.getLogger(__name__)


def constant(t, C=0):
    """A constant function with time."""
    return np.full(t.size, C)


class FunctionBlock(object):
    """An object for building functions as a sum of other functions.

    FunctionBlocks are used to create scanning functions from pre-defined
    functions. For example, a Fourier series is a the sum of a number of
    sine and cosine functions.

    FunctionBlocks are callable. They return a position as a function of time.

    Parameters
    ----------
    ndim : int
        The dimensionality of the output
    func : function
        The function which computes the output
    dfunc : function
        The derivative of the function which computes the output
    """
    def __init__(self, function=None, dfunction=None):
        super(FunctionBlock, self).__init__()
        self.ndim = len(function)
        self.func = None
        self.dfunc = None

    def __call__(self, t):
        """Return position give a time, `t`."""
        raise NotImplementedError()

    def __add__(self, another_block):
        """Return the sum of this block and another_block."""
        raise NotImplementedError

    def __radd__(self, non_block):
        return self.__add__(non_block)

    def __sub__(self, another_block):
        """Return a new block that is this block minus another_block."""
        raise NotImplementedError()

    def __rsub__(self, non_block):
        return -self.__add__(non_block)

    def __mul__(self, another_block):
        """Return a new block that is this block times another_block."""
        raise NotImplementedError()

    def __rmul__(self, non_block):
        return self.__mul__(non_block)

    def __truediv__(self, another_block):
        """Return a new block that is this block divided by another_block."""
        raise NotImplementedError()

    def __rtruediv__(self, another_block):
        raise NotImplementedError()

    def __neg__():
        """Return a new block that is negated."""
        raise NotImplementedError()


def f2w(f):
    """Return the angular frequency from the given frequency"""
    return 2*np.pi*f


def period(f):
    """Return the period from the given frequency"""
    return 1 / f


def scantimes(t0, t1, hz):
    """An array of points in the range [t0, t1) at the given frequency (hz)
    """
    return np.linspace(t0, t1, (t1-t0)*hz, endpoint=False)


def sinusoid(A, f, p, t, slope=False):
    """Return the value of a sine function at time `t`.
    #continuous #1D

    Parameters
    A : float
        The amplitude of the function
    f : float
        The temporal frequency of the function
    p : float
        The phase shift of the function
    """
    w = f2w(f)
    p = np.mod(p, 2*np.pi)
    if slope:
        return A * np.cos(w*t - p)
    return A * np.sin(w*t - p)


def triangle(A, f, p, t, slope=False):
    """Return the value of a triangle function at time `t`.
    #continuous #1d

    Parameters
    A : float
        The amplitude of the function
    f : float
        The temporal frequency of the function
    p : float
        The phase shift of the function
    """
    a = 0.5 / f
    ts = t - p/(2*np.pi)/f
    q = np.floor(ts/a + 0.5)
    pos, vel = square(4*A*f, f, p-np.pi/2, t)
    return A * (2/a * (ts - a*q) * np.power(-1, q)), pos


def sawtooth(A, f, p, t, slope=False):
    """Return the value of a sawtooth function at time `t`.
    #discontinuous #1d

    Parameters
    A : float
        The amplitude of the function
    f : float
        The temporal frequency of the function
    p : float
        The phase shift of the function
    """
    ts = t*f - p/(2*np.pi)
    q = np.floor(ts + 0.5)
    if slope:
        return 2*A*f + 0*t
    return A * (2 * (ts - q))


def square(A, f, p, t, slope=False):
    """Return the value of a square function at time `t`.
    #discontinuous #1d

    Parameters
    A : float
        The amplitude of the function
    f : float
        The temporal frequency of the function
    p : float
        The phase shift of the function
    """
    ts = t - p/(2*np.pi)/f
    return A * (np.power(-1, np.floor(2*f*ts))), 0*t


def staircase(A, f, p, t, slope=False):
    """Return the value of a staircase function at time `t`.
    #discontinuous #1d

    Parameters
    A : float
        The amplitude of the function
    f : float
        The temporal frequency of the function
    p : float
        The phase shift of the function
    """
    ts = t*f - p/(2*np.pi)
    if slope:
        return 0*t
    return A * np.floor(ts)


def lissajous(A, B, fx, fy, px, py, t, slope=False):
    """Return the value of a lissajous function at time `t`.
    #continuous #2d

    The lissajous is centered on the origin.

    Parameters
    A, B : float
        The horizontal and vertical amplitudes of the function
    fx, fy : float
        The temporal frequencies of the function
    px, py : float
        The phase shifts of the x and y components of the function
    """
    x = sinusoid(A, fx, px, t, slope=slope)
    y = sinusoid(B, fy, py, t, slope=slope)
    return x, y


def raster(A, B, fx, fy, px, py, t, slope=False):
    """Return the value of a raster function at time `t`.
    #discontinuous #2d

    The raster starts at the origin and moves initially in the positive
    directions. `fy` should be `2*fx` to make a conventional raster.

    Parameters
    A : float
        The maximum horizontal displacement of the function at half the period.
        Every period the horizontal displacement is 0.
    B : float
        The maximum vertical displacement every period.
    fx, fy : float
        The temporal frequencies of the function.
    px, py : float
        The phase shifts of the x and y components of the function
    """
    if slope:
        x = triangle(A, fx, px+np.pi/2, t, slope=True)/2
        y = staircase(B, fy, py, t, slope=True)
        return x, y
    x = triangle(A, fx, px+np.pi/2, t) + A
    y = staircase(B, fy, py, t)
    return x/2, y


def spiral(A, B, fx, fy, px, py, t, slope=False):
    """Return the value of a spiral function at time `t`.
    #discontinuous #2d

    The spiral is centered on the origin.

    Parameters
    A, B : float
        The horizontal and vertical amplitudes of the function
    fx, fy : float
        The temporal frequencies of the function
    px, py : float
        The phase shifts of the x and y components of the function
    """
    x, dx = triangle(A, fx, px+np.pi/2, t, slope=slope)
    y, dy = triangle(B, fy, py+np.pi/2, t, slope=slope)
    return [x, y], [dx, dy]


def scan3(A, B, fx, fy, fz, px, py, time, hz):
    x, y, t = lissajous(A, B, fx, fy, px, py, time, hz)
    z = sawtooth(np.pi, 0.5*fz, 0.5*np.pi, t, hz)
    return z, x, y, t


def avgspeed(time, x, y=None, z=None):
    return distance(z, x, y) / time


def lengths(x, y=None, z=None):
    if y is None:
        y = np.zeros(x.shape)
    if z is None:
        z = np.zeros(x.shape)
    a = np.diff(x)
    b = np.diff(y)
    c = np.diff(z)
    return sqrt(a*a + b*b + c*c)


def distance(x, y=None, z=None):
    d = lengths(z, x, y)
    return np.sum(d)
