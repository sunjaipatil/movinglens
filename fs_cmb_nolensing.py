"""
plots to show the effect of redshift of velocity estimation

"""
# import all modules
import time
st_time = time.time()
import numpy as np
import os
import sys

import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import healpy as hp
from tqdm.auto import tqdm

sys.path.insert(0, "/home1/sanjayku/2020/AstroPaint/")
import astropaint as ap
from astropaint import Catalog, Canvas, Painter
from astropaint.lib import utils, transform
from astropaint.profiles import NFW

from lib.matchedfilter import MatchedFilter
#
from functions import *
import getkappa as gp
import pickle,gzip
# import flatsky scripts

sys.path.insert(0, "/Users/sanjay/2020/inpainting")
import flatsky, tools, inpaint


data = pd.read_csv('../AstroPaint/astropaint/data/websky_lite_40x40_1E14.csv')
#data = pd.read_csv('/project/pierpaol_135/websky_mass_1e14.csv')

zbins = np.arange(0.0, 3.0, 0.2)#0.2
mbins = np.arange(1.0, 30, 1)*1E14
deltaz = 0.1
deltam = 0.5*1e14
total_no_samples = 100


# Define all the parameters
lmax = 9000
noise_level = 1E-8
cmb_power = True
unit_filter = False
beam_size = 1.2

# generate CMB and noise
L, Cl= utils.get_CMB_Cl(lmax=lmax, return_ell=True, uK=False)
Nl = utils._compute_Nl(noise_level, lmax)[0]
B2l = utils.get_custom_B2l(beam_size, lmax=lmax)




# Box parameters
reso = dx = .5
boxsize_am = 120. #boxsize in arcmins
xpix=nx = int(boxsize_am/dx)
mapparams = [nx, nx, dx, dx]
x1,x2 = -nx/2. * dx, nx/2. * dx
lon_range = [x1/60, x2/60]


result = {}
result['noise_level'] = noise_level
result['cmb_power'] = cmb_power
result['reso'] = dx
result['boxsize'] = boxsize_am
result['mapparams'] = mapparams

measures = ['v_th','v_ph']

beam_dict = {"pixres": dx, "L" : [L], "B2l" : [B2l]}
halo_list = np.arange(0,1)
noise_dict = {"pixres": dx, "L" : [L], "Nl" : [Nl], "seed": list(2*halo_list+1)}
cmb_dict = {"pixres": dx, "L" : [L], "Cl": [Cl], "B2l": [None], "seed": list(2*halo_list)}
# effects to be added
#kappa_dict = {"mapparms": [nx,nx,dx,dx], "mass": catalog.data["M_200c"], "redshift": catalog.data["redshift"]}
func_list = [
             #amplify,
             #set_zero,
             add_cmb,
            #gp.getkappa_imports,
             #beam_smooth_patch,
             #add_noise,
             #low_pass_filter,
             taper_patch,
            ]

func_kwargs_list = [
                    #amplify_dict,
                    #{},
                    cmb_dict,
                    #kappa_dict,
                    #beam_dict,
                    #noise_dict,

                    #lpf_dict,
                    {},
                    ]

tmp_func_list = [
                 #beam_smooth_patch,
                 #low_pass_filter,
                 taper_patch,

                ]

tmp_func_kwargs_list = [
                        #{"pixres": dx, "L" : L, "B2l" : B2l},
                        #{"pixres": dx, "L" : L, "ell_filter": 4800},
                        {},

                       ]


randomseed = 0
for z in zbins:
    dfz = data[(data['redshift']>= z-deltaz)&(data['redshift']<z+deltaz)]
    result[z] = {}
    for mm in mbins:
        dfzm = dfz[(dfz['M_200c']>= mm-deltam)&(dfz['M_200c']<mm+deltam)]
        # 1000 to extract all the statistics
        no_of_halos = max(total_no_samples,len(dfzm['M_200c']))
        result[z][round(mm/1e14, 2)] = {}
        v_th,v_thm,v_ph,v_phm = np.zeros(no_of_halos), np.zeros(no_of_halos), np.zeros(no_of_halos), np.zeros(no_of_halos)
        data_dict = dfzm.to_dict('records')
        for i,df_dict in enumerate(data_dict):
            if i == total_no_samples:
                break
            # spray BG effect on the catalog
            dataframe = pd.DataFrame([df_dict])
            catalog = Catalog(dataframe)
            canvas = Canvas(catalog, nside = 8192, R_times=10)
            painter = Painter(NFW.BG)
            painter.spray(canvas, n_cpus = 2)
            # Pass the cluster theta, phi, and angular distance information

            D_a, theta, phi =catalog.data.D_a, catalog.data.theta, catalog.data.phi


            ## weiner filter related stuff

            # CMB and noise power spectra in K

            if not cmb_power:
                Cl = Cl*0
            params = catalog.data.loc[np.arange(0, len(dataframe['x']))][["c_200c", "R_200c", "M_200c", "theta", "phi", "v_th", "v_ph"]]

            cmb_dict = {"pixres": dx, "L" : [L], "Cl": [Cl], "B2l": [None], "seed": randomseed}
            func_kwargs_list = [
                    #amplify_dict,
                    #{},
                    cmb_dict,
                    #kappa_dict,
                    #beam_dict,
                    #noise_dict,

                    #lpf_dict,
                    {},
                    ]
            for measure in measures:
                if measure=="v_th":
                    params["v_th"] = 1
                    params["v_ph"] = 0
                elif measure=="v_ph":
                    params["v_th"] = 0
                    params["v_ph"] = 1

                cutouts = canvas.cutouts(halo_list=halo_list,
                             xpix=xpix,
                             lon_range=lon_range,
                             #apply_func=smooth_taper_patch,
                             #apply_func=weiner_filter_taper,
                             apply_func=func_list,
                             func_kwargs=func_kwargs_list
                             #apply_func=add_cmb_noise_weiner,
                             #sigma=sigma,
                             #apply_func = add_cmb,
                             )

                for halo,cutout in enumerate(cutouts):
                    cluster_mass = catalog.data["M_200c"].values[0]
                    cluster_redshift = catalog.data["redshift"].values[0]
                    #cutout = gp.getkappa_imports(cutout,[nx,nx,dx,dx], cluster_mass, cluster_redshift)


                    if halo==1:
                        print("!! the code logic is wrong !!!")
                    rr_vec = get_R_vec_cart(cutout, D_a[halo], lon_range=lon_range,
                                      th=theta[halo], ph=phi[halo])
                    template = NFW.BG(rr_vec,**params.loc[halo])
                    for func, kwargs in zip(tmp_func_list, tmp_func_kwargs_list):
                        template = func(template, **kwargs)


                    mf = MatchedFilter(template=template, Cl=(B2l*Cl+Nl), lon_range=lon_range, xpix=xpix)
                    filtered_template = mf.apply_MF(template, mf.dx_arcmin/60 )
                    filtered_template = np.fft.fftshift(filtered_template).real
                    filtered_cutout = mf.apply_MF(cutout, mf.dx_arcmin/60 )
                    filtered_cutout = np.fft.fftshift(filtered_cutout).real
                    final_cutout = filtered_cutout*filtered_template
                    final_template = filtered_template*filtered_template
                    v_map = np.sum(final_cutout)/np.sum(final_template)

                    from IPython import embed;embed()
                    exit()
                if measure == "v_th":
                    v_th[i] = catalog.data[measure]
                    v_thm[i] = v_map
                elif measure == "v_ph":
                    v_ph[i] = catalog.data[measure]
                    v_phm[i] = v_map

            randomseed +=1
        result[z][round(mm/1e14, 2)]['v_th'] = v_th
        result[z][round(mm/1e14, 2)]['v_thm'] = v_thm
        result[z][round(mm/1e14, 2)]['v_ph'] = v_ph
        result[z][round(mm/1e14, 2)]['v_phm'] = v_phm

pickle.dump(result,gzip.open('mf_results_fs/%s_halos_%s_cmb_lensing_noise_%s.pkl.gz'%(total_no_samples,cmb_power,noise_level),'w'))

end_time =time.time()
total_time = (end_time-st_time)/60
print('total time taken %s minutes'%(total_time))
