import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pdb
from pprint import pprint

from lib import transform, pairwise, io

nmin = 0
nmax = 20000

r_max = 350
bin_size = 20
n_JK = 200

map_keys = ["AdvACTxDES"]
overwrite_geofactors = True

file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(file_path, "data"))
print(data_path)

if __name__ == "__main__":

    df_in = "websky_BG_CMBxGal.h5"
    df_in_fname = os.path.join(data_path, df_in)
    df = io.read_v_catalog_h5(df_in_fname, key="box", nmin=nmin, nmax=nmax)

    print("df loaded...\n")

    df_out = f"pairwise_{df_in}"
    df_out_fname = os.path.join(data_path, df_out)
    pairwise_h5 = h5py.File(df_out_fname, "a")

    print("output file loaded...\n")

    # calculate all the geofactors or load them from file
    if not all(key in pairwise_h5.keys() for key in ["q_ij", "ri_hat", "rj_hat", "dr_norm"]):

        # calculate qij
        print("calcualting geofactors...\n")

        q_ij, ri_hat, rj_hat = pairwise.calc_q_ij(df)
        geofactors = q_ij, ri_hat, rj_hat
        dr_norm = pairwise.get_dr_norm(df)

        pairwise_h5.create_dataset("q_ij", data=q_ij)
        pairwise_h5.create_dataset("ri_hat", data=ri_hat)
        pairwise_h5.create_dataset("rj_hat", data=rj_hat)
        pairwise_h5.create_dataset("dr_norm", data=dr_norm)

    else:
        print("loading geofactors...\n")
        q_ij = np.array(pairwise_h5["q_ij"])
        ri_hat = np.array(pairwise_h5["ri_hat"])
        rj_hat = np.array(pairwise_h5["rj_hat"])
        dr_norm = np.array(pairwise_h5["dr_norm"])

    # define geofactors
    geofactors = q_ij, ri_hat, rj_hat

    # calculate the simulation tij or load from file
    if "box" not in pairwise_h5.keys():

        # calculate tij
        print("calcualting tij from the box...\n")
        tij = pairwise.calc_t_ij(df, geofactors, t_col_ext="")
        tij_binned = pairwise.TijData(dr_norm, tij, q_ij, run_JK=False,
                                      r_max=r_max, bin_size=bin_size)

        pairwise_h5.create_dataset("box/tij/R", data=tij_binned.R)
        pairwise_h5.create_dataset("box/tij/Tij", data=tij_binned.Tij)



    for key in map_keys:
        print(f"calculating tij for {key}")
        print(df_in_fname)
        df = io.read_v_catalog_h5(df_in_fname, key=key, nmin=nmin, nmax=nmax)
        tij = pairwise.calc_t_ij(df, geofactors, t_col_ext="_map")
        tij_binned = pairwise.TijData(dr_norm, tij, q_ij, run_JK=True, n_JK=n_JK,
                                      r_max=r_max, bin_size=bin_size)

        try:
            pairwise_h5.create_dataset(f"map/{key}/R", data=tij_binned.R,)
            pairwise_h5.create_dataset(f"map/{key}/Tij", data=tij_binned.Tij)

            pairwise_h5.create_dataset(f"map/{key}/R_arr", data=tij_binned.R_arr)
            pairwise_h5.create_dataset(f"map/{key}/Tij_arr", data=tij_binned.Tij_arr)

            pairwise_h5.create_dataset(f"map/{key}/cov", data=tij_binned.cov)
            #pairwise_h5.create_dataset(f"map/{key}/cov_inv", data=tij_binned.cov_inv)

        except OSError:
            print("got OSError here")

            pairwise_h5[f"map/{key}/R"][:] = tij_binned.R
            pairwise_h5[f"map/{key}/Tij"][:] = tij_binned.Tij

            pairwise_h5[f"map/{key}/R_arr"][:] = tij_binned.R_arr
            pairwise_h5[f"map/{key}/Tij_arr"][:] = tij_binned.Tij_arr

            pairwise_h5[f"map/{key}/cov"][:] = tij_binned.cov
            #pairwise_h5[f"map/{key}/cov_inv"][:] = tij_binned.cov_inv
            print("success")
    pairwise_h5.close()
