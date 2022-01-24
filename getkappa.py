import numpy as np, sys
from astropy import constants as const
from astropy import units as u
from astropy import coordinates as coord
import pdb
from scipy import integrate
from astropy.cosmology import FlatLambdaCDM
import scipy.interpolate as intrp

degrees2radians = np.pi / 180.
arcmins2radians = degrees2radians / 60.

def getkappa_imports(CMB_map, mapparams, mass, redshift):
    """

    """

    # define cosmology
    h = 0.68; Tcmb = 2.725;omega_m = 0.31
    cosmo = FlatLambdaCDM(H0 = h*100., Tcmb0 = Tcmb, Om0 = omega_m)

    
    clra, cldec = 0., 0.
    nx,ny,dx,dy = mapparams
    boxsize = nx*dx
    minval, maxval = clra-boxsize/2/60.,  clra+boxsize/2/60.
    ra = np.linspace(minval, maxval, nx)
    minval, maxval =cldec-boxsize/2/60.,  cldec+boxsize/2/60.
    dec = np.linspace(minval, maxval, ny)
    RA, DEC = np.meshgrid(ra,dec)

    ra_map, dec_map = RA, DEC

    kappa_map = np.zeros((nx,ny))
    map_coords = coord.SkyCoord(ra = ra_map*u.degree, dec = dec_map*u.degree)
    cluster_coords = coord.SkyCoord(ra = clra*u.degree, dec = cldec*u.degree)
    theta_map = map_coords.separation(cluster_coords).value*(np.pi/180.)

    KAPPA = get_NFW_kappa(cosmo,theta_map, mass, 3, redshift, 1100, 200, 'crit')

    dx *= arcmins2radians
    dy *= arcmins2radians
    lx, ly = get_lxly(mapparams)
    L = np.sqrt(lx**2. + ly**2.)
    small_number = 1e-20
    KAPPA[np.isnan(KAPPA)] = small_number
    PHI_FFT = -2. * dx * dy * np.fft.fft2(KAPPA)/(L**2)
    PHI_FFT[np.isnan(PHI_FFT)] = small_number
    DEF_X    = np.fft.ifft2(-1j * PHI_FFT * lx) / ( dx * dy )
    DEF_Y    = np.fft.ifft2(-1j * PHI_FFT * ly) / ( dx * dy )
    theta_x_list = np.array(range(1,nx+1)) * dx - dx * 0.5 * (nx - 1.)
    theta_x = np.tile(theta_x_list,(nx,1))
    theta_y = np.transpose(theta_x)


    to_evaluate_unlensed_theta_x = (theta_x + DEF_X).flatten().real
    to_evaluate_unlensed_theta_y = (theta_y + DEF_Y).flatten().real

    CMB_map = intrp.RectBivariateSpline( theta_y[:,0], theta_x[0,:], CMB_map, kx = 5, ky = 5).ev(to_evaluate_unlensed_theta_y, to_evaluate_unlensed_theta_x).reshape([ny,nx])


    return CMB_map

def get_lxly(mapparams):

    nx, ny, dx, dy = mapparams
    dx *= arcmins2radians
    dy *= arcmins2radians

    lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx ), np.fft.fftfreq( ny, dy ) )
    lx *= 2* np.pi
    ly *= 2* np.pi

    return lx, ly


def get_NFW_kappa(cosmo, theta, mass, concentration, z_L, z_S, mass_def, rho_def, theta_max = -1, ret_nfw_dic = 0, des_offsets_correction = 0, richval = None, param_dict = None, totalclus = None):
    """
    Explain all the parameters

    """
    import numpy as np, sys
    from astropy import constants as const
    from astropy import units as u
    from astropy import coordinates as coord
    import pdb
    from scipy import integrate
    from astropy.cosmology import FlatLambdaCDM
    import scipy.interpolate as intrp




    nfw_dic = {}
    mass = mass*u.Msun
    if (rho_def == 'crit'):
        rho_c_z = cosmo.critical_density(z_L)
    elif (rho_def == 'mean'):
        rho_c_z = cosmo.Om(z_L)*cosmo.critical_density(z_L)
    else:
        print("rho definition not specified correctly in cluster profile")
        assert(0)

    #NFW profile properties
    delta_c = (mass_def/3.)*(concentration**3.)/( np.log(1.+concentration) - concentration/(1.+concentration) )
    r_v = (((mass/(mass_def*4.*np.pi/3.))/rho_c_z)**(1./3.)).to('Mpc')
    #r_v = (((mass/(mass_def*4.*np.pi/3.))/ (.3*rho_c_z) )**(1./3.)).to('Mpc') #change of 20160531

    r_s = r_v/concentration
    #Angular diameter distances
    D_L = cosmo.comoving_distance(z_L)/(1.+z_L)
    D_S = cosmo.comoving_distance(z_S)/(1.+z_S)
    D_LS = (cosmo.comoving_distance(z_S)-cosmo.comoving_distance(z_L))/(1.+z_S)
    #Normalization of kappa
    Sigma_c = (((const.c.cgs**2.)/(4.*np.pi*const.G.cgs))*(D_S/(D_L*D_LS))).to('M_sun/Mpc2')
    #Useful variables



    R = D_L*theta
    x = (R/r_s).value
    g_theta = fn_get_g_theta(x)
    Sigma = ((2.*r_s*delta_c*rho_c_z)*g_theta).to('M_sun/Mpc2')
    kappa = Sigma/Sigma_c


    if (theta_max > 0):
        beyond_theta_max = np.where(theta > theta_max)
        kappa[beyond_theta_max] = 0.0
    nfw_dic['r_s'] = r_s
    nfw_dic['D_L'] = D_L

    if not ret_nfw_dic:
	    return kappa.value
    else:
	    return kappa.value, nfw_dic



def fn_get_g_theta(x):
    g_theta = np.zeros(x.shape)
    gt_one = np.where(x > 1.0)
    lt_one = np.where(x < 1.0)
    #eq_one = np.where(np.abs(x - 1.0) < 1.0e-5)
    eq_one = np.where(x == 1.)
    ##eq_zeros = np.where(x == 0.)
    g_theta[gt_one] = (1./(x[gt_one]**2. - 1))*(1. - (2./np.sqrt(x[gt_one]**2. - 1.))*np.arctan(np.sqrt((x[gt_one]-1.)/(x[gt_one]+1.))) )#.value)
    g_theta[lt_one] = (1./(x[lt_one]**2. - 1))*(1. - (2./np.sqrt(1. - x[lt_one]**2.))*np.arctanh(np.sqrt((1. - x[lt_one])/(x[lt_one]+1.))) )#.value)
    ##g_theta[eq_zeros] = (1./(x[eq_zeros]**2. - 1))*(1. - (2./np.sqrt(1. - x[eq_zeros]**2.))*np.arctanh(np.sqrt((1. - x[eq_zeros])/(x[eq_zeros]+1.))) )#.value)
    g_theta[eq_one] = 1./3.

    return g_theta
