"""
plots to show the effect of redshift of velocity estimation

"""
import time
st_time = time.time()
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

sys.path.insert(0, "/home1/sanjayku/2020/AstroPaint/")
import astropaint as ap
from astropaint import Catalog, Canvas, Painter
from astropaint.lib import utils, transform
from astropaint.profiles import NFW

from lib.matchedfilter import MatchedFilter
#
from functions import *
import pickle,gzip
# import flatsky scripts

sys.path.insert(0, "/home1/sanjayku/2020/inpainting")
import flatsky, tools, inpaint


#data = pd.read_csv('/project/pierpaol_135/websky_mass_1e14.csv')
data = pd.read_csv('../AstroPaint/astropaint/data/websky_lite_40x40_1E14.csv')
#data = pd.read_csv('/project/pierpaol_135/dummy.csv')
#
zbins = np.arange(0.0, 3.0, 0.2)#0.2
mbins = np.arange(1.0, 30, 1)*1E14
deltaz = 0.1
deltam = 0.5*1e14

result = {}
for z in zbins:

    dfz = data[(data['redshift']>= z-deltaz)&(data['redshift']<z+deltaz)]
    result[z] = {}
    for mm in mbins:
        print("MASS!!!! %s"%(mm))
        dfzm = dfz[(dfz['M_200c']>= mm-deltam)&(dfz['M_200c']<mm+deltam)]
        no_of_halos = len(dfzm['M_200c'])
        result[z][round(mm/1e14, 2)] = {}
        v_th,v_thm,v_ph,v_phm = np.zeros(no_of_halos), np.zeros(no_of_halos), np.zeros(no_of_halos), np.zeros(no_of_halos)
        
        data_dict = dfzm.to_dict('records')
        for i,df_dict in enumerate(data_dict):
            
            if not abs(df_dict['v_th'])>=10*abs(df_dict['v_ph']):
                continue
            dataframe = pd.DataFrame([df_dict])
            catalog = Catalog(dataframe)
            canvas = Canvas(catalog, nside = 8192, R_times=10)
        
            painter = Painter(NFW.BG)
            painter.spray(canvas, n_cpus = 2)
            dx = .5
            boxsize_am = 120. #boxsize in arcmins
            xpix=nx = int(boxsize_am/dx)
            mapparams = [nx, nx, dx, dx]
            x1,x2 = -nx/2. * dx, nx/2. * dx
            lon_range = [x1/60, x2/60]
            halo_list = np.arange(0,1)
            D_a, theta, phi =catalog.data.D_a, catalog.data.theta, catalog.data.phi

            if len(D_a)>1:
                print('More than one cluster in the data frame');quit()
            ## filter related stuff
            lmax = 9000

            exp_name = "CMB-S4"
            exp_beam = 1.4
            sigma_n = 1
            exp_frequency = [93, 145, 225]
            # CMB and noise power spectra in K
            L, Cl= utils.get_CMB_Cl(lmax=lmax, return_ell=True, uK=False)
            #Nl = utils._compute_Nl(sigma_n, lmax)[0]
            Cl = Cl
            exp_fsrequency = [93, 145, 225]
            Nl = utils.get_experiment_Nl(name=exp_name, frequency=exp_frequency, lmax=lmax, apply_beam=False, uK=False)
            B2l = utils.get_custom_B2l(exp_beam, lmax=lmax)

            params = catalog.data.loc[np.arange(0, len(dataframe['x']))][["c_200c", "R_200c", "M_200c", "theta", "phi", "v_th", "v_ph"]]

            measures = ['v_th', 'v_ph']
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
                             apply_func=[],
                             func_kwargs=[]
                             #apply_func=add_cmb_noise_weiner,
                             #sigma=sigma,
                             #apply_func = add_cmb,
                             )

                for j,cutout in enumerate(cutouts):
                    halo = j
                    rr_vec = get_R_vec_cart(cutout, D_a[halo], lon_range=lon_range,
                                      th=theta[halo], ph=phi[halo])
                    template = NFW.BG(rr_vec,**params.loc[halo])
                    cutout = taper_patch(cutout)
                    template = taper_patch(template)
                    mf = MatchedFilter(template=template, Cl=(B2l*Cl+Nl), lon_range=lon_range, xpix=xpix)
                    filtered_template = mf.apply_MF(template, mf.dx_arcmin/60 )
                    filtered_template = np.fft.fftshift(filtered_template).real
                    filtered_cutout = mf.apply_MF(cutout, mf.dx_arcmin/60 )
                    filtered_cutout = np.fft.fftshift(filtered_cutout).real
                    final_cutout = filtered_cutout*filtered_template
                    final_template = filtered_template*filtered_template
                    v_map = np.sum(final_cutout)/np.sum(final_template)

                if measure == "v_th":
                    v_th[i] = catalog.data.iloc[0]['v_th']
                    v_thm[i] = v_map
                elif measure == "v_ph":
                    v_ph[i] = catalog.data.iloc[0]['v_ph']
                    v_phm[i] = v_map
        inds = np.where(v_th!=0)[0]
        
        result[z][round(mm/1e14, 2)]['v_th'] = v_th[inds]
        result[z][round(mm/1e14, 2)]['v_thm'] = v_thm[inds]
        result[z][round(mm/1e14, 2)]['v_ph'] = v_ph[inds]
        result[z][round(mm/1e14, 2)]['v_phm'] = v_phm[inds]
pickle.dump(result,gzip.open('mf_results_websky_lite_vph.pkl.gz','w'))

