"""
All functions

"""

# defining some more functions

# Function for white noise

# Define important functions

import numpy as np
# Import all the modules
import os
import sys

import numpy as np
import pandas as pd
#import modin.pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import healpy as hp
from tqdm.auto import tqdm


import astropaint as ap
from astropaint import Catalog, Canvas, Painter
from astropaint.lib import utils, transform
from astropaint.profiles import NFW

sys.path.insert(0, "/Users/sanjay/2020/inpainting")
import flatsky, tools, inpaint

def set_zero(patch):

    return 0*patch


def amplify(patch, factor=1000):

    return factor*patch


def add_cmb( patch, pixres, L, Cl, B2l = None, seed =0):#(patch, pixres, L, Cl, B2l=None, seed=0):

    mapparams = [*patch.shape, pixres, pixres]
    print(seed)    
    np.random.seed(seed)
    #mapparams = [*patch.shape, pixres, pixres]

    #create CMB and convolve with beam
    cmb = flatsky.make_gaussian_realisation(mapparams, L, Cl, bl = B2l)

    patch_plus_cmb = cmb + patch

    #if B2l is not None:
     #   Bl2d = flatsky.cl_to_cl2d(L, B2l, mapparams)

      #  patch_plus_cmb = np.fft.ifft2( np.fft.fft2(patch_plus_cmb) * Bl2d ).real

    return patch_plus_cmb

def add_noise(patch, pixres, L, Nl, seed=0):

    """
    SP: why to have a noise seed to begin with
    """


    #np.random.seed(seed)

    mapparams = [*patch.shape, pixres, pixres]

    noise = flatsky.make_gaussian_realisation(mapparams, L, Nl)

    return patch + noise


def beam_smooth_patch(patch, pixres, L, B2l):

    mapparams = [*patch.shape, pixres, pixres]
    B2l_2d = flatsky.cl_to_cl2d(L, B2l, mapparams)

    smooth_patch = np.fft.ifft2( np.fft.fft2(patch) * B2l_2d ).real

    return smooth_patch

def high_pass_filter(patch, pixres, L, ell_filter=2000):
    mapparams = [*patch.shape, pixres, pixres]
    hpf = np.ones_like(L)
    hpf[:ell_filter] = 0
    hpf_2d = flatsky.cl_to_cl2d(L, hpf, mapparams)
    hpf_patch = np.fft.ifft2( np.fft.fft2(patch) * hpf_2d ).real

    return hpf_patch

def low_pass_filter(patch, pixres, L, ell_filter=2000):
    mapparams = [*patch.shape, pixres, pixres]
    lpf = np.ones_like(L)
    lpf[ell_filter:] = 0
    lpf_2d = flatsky.cl_to_cl2d(L, lpf, mapparams)
    lpf_patch = np.fft.ifft2( np.fft.fft2(patch) * lpf_2d ).real

    return lpf_patch

def taper_patch(patch):

    nx = patch.shape[0]
    han = np.hanning(nx)
    han2d = np.outer(han, han)

    patch*=han2d

    return patch

#wiener_2d = Cl_BG_2d*B2l_2d/((Cl_BG_2d+Cl_CMB_2d)*B2l_2d+Cl_N_2d)

def apply_filter_fft2(patch, filter_fft2 ):



    patch_filtered = np.fft.ifft2( np.fft.fft2(patch) * filter_fft2 ).real

    return patch_filtered



# White noise function
degrees2radians = np.pi / 180.
arcmins2radians = degrees2radians / 60.

def fn_get_noise(mapparams, expnoiselevel = None, seed = 0):
    nx, ny, dx, dy = mapparams
    dx *= arcmins2radians
    dy *= arcmins2radians
    npixels = nx * ny

    if 0:
        expnoiselevel = self.expnoiselevel
    else:
        expnoiselevel = np.asarray(expnoiselevel)


    DeltaT = expnoiselevel * (np.radians(1/60.)) #in uK now

    NOISE = np.zeros( (1,nx,ny) )
    np.random.seed(seed)
    DUMMY = np.random.standard_normal([nx,ny])
    NOISE[0] = DUMMY * DeltaT/dx
        #NOISE[nn] = np.random.normal(loc=0.0, scale=DeltaT[nn]/dx, size=[nx,ny])

    #imshow(NOISE[0]);colorbar();show()
        #show();quit()

    return NOISE

def get_R_vec_cart(patch, D_a, lon_range, th, ph):
    xpix = patch.shape[0]

    # angular coordinates on the side in radians
    x_rad = np.deg2rad(np.linspace(lon_range[1], lon_range[0], xpix))

    x = x_rad * D_a

    xx, yy = np.meshgrid(x, x, indexing="ij")

    R_vec_2d_sph = np.stack([0 * yy, xx, yy], axis=-1)

    #J_cart2sph = get_cart2sph_jacobian(th, ph)
    J_sph2cart = get_sph2cart_jacobian(th, ph)
    # J_sph2cart = transform.sph2cart(cat['co-lat'].values,cat['lon'].values)

    #R_vec_sph = np.einsum('ij,...i->...j', J_cart2sph, R_vec_cart)
    R_vec_cart = np.einsum('ij,...i->...j', J_sph2cart, R_vec_2d_sph)

    return R_vec_cart



def get_sph2cart_jacobian(th, ph):
    """calculate the transformation matrix (jacobian) for spherical to cartesian coordinates at
    line of sight (th, ph) [radians]
        see https://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates

        th is the polar angle with respect to z and ph is the azimuthal angle with respect to x

    example: see cart2sph2"""

    row1 = np.stack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))
    row2 = np.stack((np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)))
    row3 = np.stack((-np.sin(ph), np.cos(ph), 0.0 * np.cos(th)))

    return np.stack((row1, row2, row3))
