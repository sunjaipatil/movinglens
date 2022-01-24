"""
Functions to analyse the results

"""
import pickle,gzip
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def get_eper(data_vel,model_vel, no_samples = 1000):
    """
    Gets Model and Data velocities
    and return error percentages
    """
    v_th, v_ph = data_vel
    v_thm, v_phm = model_vel
    eth = abs(v_th-v_thm)
    eph = abs(v_ph-v_phm)
    v_r = np.sqrt(v_th**2+v_ph**2)

    errper = ((eth + eph)/v_r)*100


    if len(v_th)<no_samples:
        errper = np.nan


    return errper
def get_epth(data_vel,model_vel):
    """
    Gets Model and Data velocities
    and return error percentages
    """
    v_th, v_ph = data_vel
    v_thm, v_phm = model_vel
    eth = (v_th-v_thm)
    #eph = abs(v_ph-v_phm)
    v_r = np.sqrt(v_th**2)

    errper = ((eth)/v_r)*100
    if len(v_th)<10:
        errper = 0
    return errper

def get_epph(data_vel,model_vel):
    """
    Gets Model and Data velocities
    and return error percentages
    """
    v_th, v_ph = data_vel
    v_thm, v_phm = model_vel
    #eth = abs(v_th-v_thm)
    eph = (v_ph-v_phm)
    v_r = np.sqrt(v_ph**2)

    errper = ((eph)/v_r)*100
    if len(v_th)<5:
        errper = 0
    return errper


def geterror_stat(file_name, mbins, zbins, no_samples = 1000, stat = 'max'):
    """
    a function to get the error statistics given a file name
    """
    data = pickle.load(gzip.open(file_name))
    result = np.zeros((len(zbins), len(mbins)))*np.nan
    clusternum = np.zeros((len(zbins), len(mbins)))
    for i,z in enumerate(zbins):
        for j,mm in enumerate(mbins):
            try:
                temp = data[z][round(mm/1e14, 2)]
            except:
                continue
            if len(temp['v_th']>no_samples):
                # 1000 is hard coded
                #rand_arr = np.random.choice(np.arange(1000), size = no_samples, replace = False )
                data_vel = (temp['v_th'][0:no_samples], temp['v_ph'][0:no_samples])
                model_vel = (temp['v_thm'][0:no_samples], temp['v_phm'][0:no_samples])
            else:
                data_vel = (temp['v_th'], temp['v_ph'])
                model_vel = (temp['v_thm'], temp['v_phm'])
            #from IPython import embed;embed()
            rs = get_eper(data_vel,model_vel, no_samples)
            #return rs
            #from IPython import embed;embed()
            if stat=='max':
                result[i,j] = np.max(get_eper(data_vel,model_vel, no_samples))
            elif stat == 'median':
                result[i,j] = np.median(get_eper(data_vel,model_vel, no_samples))

            clusternum[i,j] = len(temp['v_th'])


    return result,clusternum

def getbias(file_name, no_samples = 1000, mass = 2E14, redshift = 1):
    """
    calculates the velocity bias

    """
    zbins = np.arange(0.0, 3.0, 0.2)

    data = pickle.load(gzip.open(file_name))
    temp = data[zbins[redshift]][round(mass/1e14, 2)]
    print(zbins[redshift])
    data_th, data_ph = temp['v_th'][0:no_samples], temp['v_ph'][0:no_samples]
    model_th, model_ph = temp['v_thm'][0:no_samples], temp['v_phm'][0:no_samples]
    vth_bias = data_th - model_th
    vph_bias = data_ph - model_ph

    return vth_bias, vph_bias


zbins = np.arange(0.0, 3.0, 0.2)

mbins = np.arange(1.0, 8, 1)*1E14
    #result,cluster_num = geterror_stat('mf_results_websky_lite_vph.pkl.gz', mbins, zbins, no_samples = 1000, stat = 'max')
#result = geterror_stat('mf_results_full_extent_no_CMB_power.pkl.gz', mbins, zbins)
xvals = mbins/1e14
#plt.imshow(result, extent = [xvals.min(), xvals.max(),zbins.max(),zbins.min()], aspect = 'auto');plt.colorbar();plt.show()
#())
#plt.title("max error");plt.show()
