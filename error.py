"""
Plots to figure out the error
"""
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



# import matched filter
#sys.path.insert(0, os.path.join(os.path.realpath("../..")))
from lib.matchedfilter import MatchedFilter


# import flatsky scripts

sys.path.insert(0, "/Users/sanjaykumarp/usc_postdoc/2020/inpainting")
import flatsky, tools, inpaint

# defining some more functions

# Function for white noise

# Define important functions

def set_zero(patch):

    return 0*patch


def amplify(patch, factor=1000):

    return factor*patch


def add_cmb(mapparams, L, Cl, B2l = None, seed =0):#(patch, pixres, L, Cl, B2l=None, seed=0):
    np.random.seed(seed)
    #mapparams = [*patch.shape, pixres, pixres]

    #create CMB and convolve with beam
    cmb = flatsky.make_gaussian_realisation(mapparams, L, Cl, bl = B2l)

    #patch_plus_cmb = cmb + patch

    #if B2l is not None:
     #   Bl2d = flatsky.cl_to_cl2d(L, B2l, mapparams)

      #  patch_plus_cmb = np.fft.ifft2( np.fft.fft2(patch_plus_cmb) * Bl2d ).real

    return cmb#patch_plus_cmb

def add_noise(patch, pixres, L, Nl, seed=0):
    np.random.seed(seed)
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





catalog = Catalog("200_halos.csv")
canvas = Canvas(catalog, nside = 8192, R_times=10)

#painter = Painter(NFW.BG)
#painter.spray(canvas, n_cpus = 2)
canvas.load_map_from_file('200_halos.fits')

#df = catalog.data[['x','y' ,'z' ,'M_200c', 'redshift', 'D_c', 'theta', 'phi', 'R_200c', 'R_th_200c', 'v_th', 'v_ph',]]


#measures = ["v_th", "v_ph"]

# Define halo list here...

#halo_list = np.arange(0,200)
D_a = catalog.data.D_a
theta = catalog.data.theta
phi = catalog.data.phi



lmax = 9000

exp_name = "CMB-S4"
exp_frequency = [93, 145, 225]
#exp_frequency = [90, 150, 220]
exp_beam = 1.4

# CMB and noise power spectra in K
L, Cl = utils.get_CMB_Cl(lmax=lmax, return_ell=True, uK=False)
Nl_SO = utils.get_experiment_Nl(name=exp_name, frequency=exp_frequency, lmax=lmax, apply_beam=False, uK=False)


B2l = utils.get_custom_B2l(exp_beam, lmax=lmax)

#params or supply a params file
dx = .5
boxsize_am = 120. #boxsize in arcmins
xpix=nx = int(boxsize_am/dx)
mapparams = [nx, nx, dx, dx]
x1,x2 = -nx/2. * dx, nx/2. * dx
verbose = 0


lon_range = [x1/60, x2/60]
#xpix = nx
#sigma = 2

han = np.hanning(nx)
han2d = np.outer(han, han)


#lmax = 10000
#el = np.arange(lmax)

#beam and noise levels
#noiseval = 2.0 #uK-arcmin
#beamval = 1.4 #arcmins

#CMB power spectrum
#Cls_file = 'data/Cl_Planck2018_camb.npz'
#Tcmb = 2.73

#for inpainting
noofsims = 1000
mask_radius_inner = 20.0 #arcmins
mask_radius_outer = 60.0 #arcmins
mask_inner = 0  #If 1, the inner region is masked before the LPF. Might be useful in the presence of bright SZ signal at the centre.

#beam_dict = {"pixres": 1., "L" : [L], "B2l" : [B2l]}

cmb_dict = {"pixres": dx, "L" : [L], "Cl": [Cl], "B2l": [None]}
beam_dict = {"pixres": dx, "L" : [L], "B2l" : [B2l]}
noise_dict = {"pixres": dx, "L" : [L], "Nl" : [Nl_SO]}
#wiener_dict = { "filter_fft2" : [Wl_2d]}
lpf_dict = { "pixres": dx, "L" : [L], "ell_filter": [4800]}

amp_factor = 1
amplify_dict = { "factor" : [amp_factor]}

func_list = [
             #amplify,
             #set_zero,
             #add_cmb,
             #beam_smooth_patch,
             #add_noise,
             #low_pass_filter,
             #taper_patch,
            ]

func_kwargs_list = [
                    #{},
                    #amplify_dict,
                    #cmb_dict,
                    #beam_dict,
                    #noise_dict,
                    #lpf_dict,
                    #{},
                    ]

#func_list = [beam_smooth_patch, low_pass_filter, taper_patch]
#func_kwargs_list = [beam_dict,lpf_dict, {}]

tmp_func_list = [
                 #beam_smooth_patch,
                 #low_pass_filter,
                 #taper_patch,
                ]

tmp_func_kwargs_list = [
                        #{"pixres": dx, "L" : L, "B2l" : B2l},
                        #{"pixres": dx, "L" : L, "ell_filter": 4800},
                        #{},
                       ]



df = pd.read_csv('notebooks/results_200_halos.csv')

eth = df['error_th']


inds = np.where(eth>100)[0]
measures = ['v_th']


halo_list = np.where(eth>100)[0]
params = catalog.data.loc[halo_list][["c_200c", "R_200c", "M_200c", "theta", "phi", "v_th", "v_ph"]]
real_space_MF = False
for measure in measures:
    #print(measure)
    if measure=="v_th":
        #print(f"setting template using {measure}")
        params["v_th"] = 1
        params["v_ph"] = 0

        row = 0
    elif measure=="v_ph":
        #print(f"setting template using {measure}")
        params["v_th"] = 0
        params["v_ph"] = 1
        row = 1

    cutouts = canvas.cutouts(halo_list=halo_list,
                         xpix=xpix,
                         lon_range=lon_range,
                         #apply_func=smooth_taper_patch,
                         #apply_func=weiner_filter_taper,
                         apply_func=func_list,
                         func_kwargs=func_kwargs_list,
                         #apply_func=add_cmb_noise_weiner,
                         #sigma=sigma,
                         #apply_func = add_cmb,
                         )

    # loop over each cutout and infer the velocity
    for halo, cutout in zip(halo_list, cutouts):
        # add white noise here
        mapparams =[240,240, 0.5,0.5]

        #noise = fn_get_noise(mapparams,expnoiselevel =noise_level , seed = i)
        #cmb = add_cmb(mapparams, L, Cl, B2l = None, seed =i)
        #i +=1
        cutout = cutout #+ noise[0] + cmb
        # build the template
        rr_vec = get_R_vec_cart(cutout, D_a[halo], lon_range=lon_range,
                                  th=theta[halo], ph=phi[halo])

        template = NFW.BG(rr_vec,**params.loc[halo])

        for func, kwargs in zip(tmp_func_list, tmp_func_kwargs_list):
            template = func(template, **kwargs)
        #template = weiner_filter_taper(template)

        mf = MatchedFilter(template=template, Cl=(B2l*Cl+Nl_SO), lon_range=lon_range, xpix=xpix)

        if real_space_MF:

            bg_mf = mf.calc_MF(mf.template, mf.Cl_2D, mf.dx_arcmin/60)
            filtered_template = template*bg_mf
            filtered_cutout = cutout*bg_mf

            v_map = (np.sum((filtered_cutout)))/(np.sum((filtered_template)))

        else:

            filtered_template = mf.apply_MF(template, mf.dx_arcmin/60 )

            filtered_template = np.fft.fftshift(filtered_template).real

            filtered_cutout = mf.apply_MF(cutout, mf.dx_arcmin/60 )

            filtered_cutout = np.fft.fftshift(filtered_cutout).real

            final_cutout = filtered_cutout*filtered_template
            final_template = filtered_template*filtered_template

            v_map = np.sum(final_cutout)/np.sum(final_template)

            #v_map = (np.sum((filtered_cutout[nx//2-2:nx//2+2,nx//2-2:nx//2+2])))/(np.sum((filtered_template[nx//2-2:nx//2+2,nx//2-2:nx//2+2])))
        #filered_patch = convolve2d(template,bg_mf, mode="same")
        #filtered_template = template*bg_mf
        #filtered_cutout = cutout*bg_mf

        #filtered_template = convolve2d(template, bg_mf, mode="same")
        #filtered_cutout = convolve2d(cutout, bg_mf, mode="same")

        #
            #v_map /=amp_factor
            MF = ((np.fft.ifft2(np.fft.ifftshift(mf.MF_fft2D)))).real

        plt.clf()
        plt.subplot(121);plt.imshow(1E6*cutout, cmap = cm.RdBu)
        plt.subplot(122);plt.imshow(template, cmap = cm.RdBu)
        plt.title('%s err %s z %s M'%(round(eth[halo],1),round(df.loc[halo]['redshift'],2), round(df.loc[halo]['M_200c']/1e14,2)))
        plt.show()
