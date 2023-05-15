import sep
from astropy.io import fits
import numpy as np
from .infer_functions import write_regions

path = "/Users/edroberts/Desktop/im_gen/training_data/testing/ceers/"
file = 'x2178y3634'
# open a FITS file
with fits.open(path+'fits/'+file+'.fits') as hdul:
    test_image = np.array(hdul[0].data).astype(np.float32)


objects = sep.extract(test_image, 1, err=0.01)

# print the number of objects found
print(f'Number of objects found: {len(objects)}')

# print(objects)

# print the position of each object
for obj in objects:
    print(f'X position: {obj["x"]}, Y position: {obj["y"]}')

conf = (list(np.full(len(objects["x"]),0.95)))
xy = []

for i in range(len(objects['x'])):
    xy.append([objects['x'][i],objects['y'][i]])


write_regions(conf, xy, ('/Users/edroberts/Desktop/im_gen/training_data/testing/general_metrics/to_use/'+'/sep/'+file+'_2.reg'), 'null')
