from astropy.io import fits
import numpy as np

coords = [[1656,2712],
          [692,3619],
          [3614,1306],
          [3634,2178],
          [2755,2658],
          [1801,3312],
          [2890,7573],
          [2391,7743]]



with fits.open('/Users/edroberts/Desktop/im_gen/MISC/F444W_JWST/hlsp_ceers_jwst_nircam_nircam1_f444w_dr0.5_i2d.fits') as hdul:    
    # Get the data and header from the primary HDU
    data = hdul[1].data

mx = []

for [y,x] in coords:

    # Select the desired pixel region
    selected_data = data[y:(y+224), x:(x+224)]

    # print(selected_data)
    mx.append(np.max(selected_data))

    # Create a new FITS file with the selected data
    hdu = fits.PrimaryHDU(selected_data)
    hdu.writeto('/Users/edroberts/Desktop/im_gen/training_data/testing/CEERS/fits/x'
                +str(x)+'y'+str(y)+'.fits', overwrite=True)

print(f'Max pixel value is: {np.max(mx)}')