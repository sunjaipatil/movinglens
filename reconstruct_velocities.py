import os

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import healpy as hp
from tqdm.auto import tqdm
import astropaint as ap
from astropaint import Catalog, Canvas
from astropaint.lib import profile, utilities

from lib.matchedfilter import MatchedFilter
from lib import transform as util

from scipy.ndimage import gaussian_filter


astropaint_path = ap.__path__[0]
print(astropaint_path)
map_path = os.path.join(astropaint_path, "examples")
mask_path = os.path.join(".", "data", "masks")
data_path = os.path.join(".", "data")
mask_fname = "SO_mask_16000_Gal_bool_Nside1024_apod_taperFWHM1deg.fits"
#map_fname = "websky_BG_NFW_NSIDE=8192_ray.fits"

nside = 128
mass_cut = 1E15

map_out_fname = f"websky_BG_CBN_NSIDE={nside}.fits"
df_out_name = f"websky_BG_CBN_SO_mask_M>{mass_cut:.2E}.csv"

# cutout specs
lon_range = [-1, 1]
xpix = 200
sigma = 1

# filter params
lmax = 7000

exp_name = "SO"
exp_frequency = 145
exp_beam = 1.4

plot_results = False



if __name__ == "__main__":
    # load the catalog
    catalog = Catalog("websky_lite_redshift")
    so_mask = hp.read_map(os.path.join(mask_path, mask_fname))

    # cut the catalog

    catalog.cut_M_200c(mass_cut)
    catalog.cut_mask(so_mask)

    # load the map
    canvas = Canvas(catalog, nside=nside, R_times=10)
    canvas.load_map_from_file(os.path.join(map_path, map_out_fname))
    canvas.cut_alm(lmin=1000)
    # Matched filter pipeline
    df = catalog.data[['x', 'y', 'z', 'M_200c',
                       'redshift', 'D_c', 'theta', 'phi', 'R_200c',
                       'R_th_200c', 'v_th', 'v_ph']]

    measures = ["v_th", "v_ph"]


    halo_list = catalog.data.index

    # template params
    D_a = catalog.data.D_a
    theta = catalog.data.theta
    phi = catalog.data.phi
    params = catalog.data.loc[halo_list][["c_200c", "R_200c", "M_200c", "theta", "phi", "v_th", "v_ph"]]

    # filter params
    L, Cl = utilities.get_CMB_Cl(lmax=lmax, return_ell=True, uK=False)
    Nl = utilities.get_experiment_Nl(name=exp_name, frequency=exp_frequency, lmax=lmax, apply_beam=False, uK=False)
    B2l = utilities.get_custom_B2l(exp_beam, lmax=lmax)



    # loop over theta and phi components
    for measure in measures:
        print(measure)
        if measure == "v_th":
            print(f"setting template using {measure}")
            params["v_th"] = 1
            params["v_ph"] = 0

        elif measure == "v_ph":
            print(f"setting template using {measure}")
            params["v_th"] = 0
            params["v_ph"] = 1

        cutouts = canvas.cutouts(halo_list=halo_list,
                                 xpix=xpix,
                                 lon_range=lon_range,
                                 apply_func=gaussian_filter,
                                 sigma=sigma,
                                 )

        # loop over each cutout and infer the velocity
        for halo, cutout in tqdm(enumerate(cutouts), total=len(halo_list)):

            # build the template
            rr_vec = util.get_R_vec_cart(cutout, D_a[halo], lon_range=lon_range,
                                         th=theta[halo], ph=phi[halo])

            template = profile.NFW.BG(rr_vec, **params.loc[halo])

            template = gaussian_filter(template, sigma)

            # build the matched filter
            mf = MatchedFilter(template=template, Cl=(B2l * Cl + Nl), lon_range=lon_range, xpix=xpix)
            bg_mf = mf.calc_MF(mf.template, mf.Cl_2D, mf.dx_arcmin / 60)

            # filered_patch = convolve2d(template,bg_mf, mode="same")
            filtered_template = template * bg_mf
            filtered_cutout = cutout * bg_mf

            # filtered_template = convolve2d(template, bg_mf, mode="same")
            # filtered_cutout = convolve2d(cutout, bg_mf, mode="same")

            if plot_results:
                fig, ax = plt.subplots(1, 4, figsize=(15, 3))

                cutout_plot = ax[0].imshow(cutout, cmap=cm.RdBu)
                plt.colorbar(cutout_plot, ax=ax[0])
                ax[0].set_title("cutout")

                # matched filter
                mf_plot = ax[1].imshow(bg_mf, cmap=cm.RdBu)
                # plt.colorbar(mf_plot,ax=ax[1])
                ax[1].set_title("matched filter")

                # filtered patch
                mf_plot = ax[2].imshow(template, cmap=cm.RdBu)
                plt.colorbar(mf_plot, ax=ax[2])
                ax[2].set_title("Template cutout")

                # filtered profile
                ax[3].plot(filtered_cutout[xpix // 2])
                # ax[3].plot(cutout[:,xpix//2])
                # ax[3].plot(100*template[:,xpix//2])
                plt.show()

            if measure == "v_th":
                df.loc[halo, "v_th_map"] = (np.sum((filtered_cutout))) / (np.sum((filtered_template)))

            elif measure == "v_ph":
                df.loc[halo, "v_ph_map"] = (np.sum((filtered_cutout))) / (np.sum((filtered_template)))


    df.to_csv(os.path.join(data_path, df_out_name))
    #canvas.show_map()
    #plt.show()
