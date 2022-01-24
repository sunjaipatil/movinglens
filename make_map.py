import os

import astropaint as ap
from astropaint import Catalog, Canvas
from astropaint.lib import transform

astropaint_path = ap.__path__[0]
print(astropaint_path)
map_path = os.path.join(astropaint_path, "examples")
mask_path = os.path.join(".", "data", "masks")
mask_fname = "SO_mask_16000_Gal_bool_Nside1024_apod_taperFWHM1deg.fits"
#map_fname = "websky_BG_NFW_NSIDE=8192_ray.fits"

nside = 128

map_fname = "websky_BG_NSIDE=128.fits"
map_out_fname = "websky_BG_CBN_NSIDE=128.fits"

add_cmb = True
add_noise = True
exp_name = "SO"
exp_freq = 145  # GHz
exp_beam = 1.4  # arcmin
fwhm_b = ap.lib.transform.arcmin2rad(exp_beam)

# load the catalog
catalog = Catalog("websky_lite_redshift")

# load the map
canvas = Canvas(catalog, nside=nside, R_times=10)
canvas.load_map_from_file(os.path.join(map_path, map_fname))

# load cmb and noise
if add_cmb:
    canvas.add_cmb()
    canvas.beam_smooth(fwhm_b)
if add_noise:
    canvas.add_noise(Nl=exp_name, frequency=exp_freq)

canvas.save_map_to_file(os.path.join(map_path, map_out_fname))
#canvas.show_map()
#plt.show()
