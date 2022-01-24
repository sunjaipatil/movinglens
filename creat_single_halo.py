"""
Create a single halo dataset with closest redshift or Mass

"""

# Read the websky catalog

import pandas as pd
import numpy as np

data = pd.read_csv('/Users/sanjaykumarp/usc_postdoc/bg_project/AstroPaint/astroPaint/data/websky_lite_40x40_1E14.csv')



data_m = data[(data["M_200c"]>9e14) & (data["M_200c"]<10e14)]
data_mz = data_m[data_m["redshift"] == data_m["redshift"].min()]
data_mz.to_csv('/Users/sanjaykumarp/usc_postdoc/bg_project/AstroPaint/astroPaint/data/m_1e15_z.csv')

# redshift_list = np.arange(0.1, 1.5, 0.05)
# ind_list = np.zeros(len(redshift_list))
# for z in redshift_list:
#     ind = np.argmin(np.abs(data.redshift - z))
#
#     left_inds = np.arange(0,ind)
#     right_inds = np.arange(ind+1, len(data['x']))
#     drop_inds = np.concatenate([left_inds, right_inds])
#     temp = data.drop(drop_inds)
#     temp.to_csv('/Users/sanjaykumarp/usc_postdoc/bg_project/AstroPaint/astroPaint/data/one_halo_z_%0.4s.csv'%(z))
