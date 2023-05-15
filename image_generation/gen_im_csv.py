#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make_test_image.py -
Use galsim to generate some images that can be used for training
and testing. Largely adopated from Ben Johnson's make_test_image.py.
These will be single band images. The images include
   * galaxy_grid - A small grid of galaxies with varied shape parameters
                   (size, sersic, axis ratio) and at different S/N
"""

import csv
import sys
from argparse import ArgumentParser, Namespace
from itertools import product
import yaml
import os

import numpy as np
import scipy.stats as stats
import powerlaw

from astropy.io import fits
from astropy.wcs import WCS
import galsim


def read_config(config_file):
    with open(config_file, "r") as c:
        conf = yaml.load(c, Loader=yaml.FullLoader)
    config = Namespace(**conf)
    return config


def make_truncnorm(min=0, max=1, mu=0, sigma=1, **extras):
    a = (min - mu) / sigma
    b = (max - mu) / sigma
    return stats.truncnorm(a, b, loc=mu, scale=sigma)


def draw_sources(config):
    '''
    Draw galaxy parameter
    
    Essentially setting up dictionary of parameters set by config file
    '''
    source_dict = {}
    # draw number of galaxies
    # Chosing number of galaxies to draw from a gaussian (truncated norm)
    number_obj_fct = make_truncnorm(**config.sources["number"])
    # number of galaxies:
    number_obj = int(number_obj_fct.rvs(1))


    # draw SNR from powerlaw (picking SNR for each source from a powerlaw and adding to src dictionary
    snr_list = powerlaw.Power_Law(xmin=config.sources['snr']['min'],
                                  parameters=[config.sources['snr']['powerlaw_index']]).generate_random(number_obj)
    source_dict['snr'] = np.clip(snr_list, config.sources['snr']['min'],
                                 config.sources['snr']['max'])

    # draw size in arcsec
    log_size_arcsec_fct = make_truncnorm(**config.sources["log_size"])
    size_arcsec_list = np.power(10, log_size_arcsec_fct.rvs(number_obj))
    source_dict['rhalf'] = size_arcsec_list

    # draw Sersic index
    config.sources["sersic"]["min"] = max(0.3, config.sources["sersic"]["min"])
    config.sources["sersic"]["max"] = min(6.2, config.sources["sersic"]["max"])
    sersic_fct = make_truncnorm(**config.sources["sersic"])
    sersic_list = sersic_fct.rvs(number_obj)
    source_dict['sersic'] = np.clip(sersic_list, config.sources['sersic']['min'],
                                    config.sources['sersic']['max'])

    # draw axis ratio
    q_fct = make_truncnorm(**config.sources["q"])
    q_list = q_fct.rvs(number_obj)
    source_dict['q'] = q_list

    return(number_obj, source_dict)


def make_catalog(grid_points, n_gal, band="fclear", pixel_scale=0.03, noise=1.0):
    '''
    Generate catalog.
    '''
    #Getting source dictionary keys
    grid_keys = list(grid_points.keys())
    
    #making collumns 
    cols = np.unique(grid_keys + ["pa", "ra", "dec", "id", "x", "y"] + [band])
    
    #defining catalog data type
    cat_dtype = np.dtype([(c, np.float64) for c in cols])
    
    #initialising catalog
    cat = np.zeros(n_gal, dtype=cat_dtype)
    
    for i, k in enumerate(grid_keys):
        cat[k] = grid_points[k]
    n_pix = np.pi * (cat["rhalf"] / pixel_scale)**2
    cat[band] = 2 * cat["snr"] * np.sqrt(n_pix) * noise
    cat["pa"] = np.random.uniform(low=-0.5*np.pi, high=0.5*np.pi, size=(n_gal))
    return cat


def make_image(cat, n_pix_per_side, n_pix_per_gal,
               band="fclear", pixel_scale=0.03, sigma_psf=1.):
    '''
    Generate image with galsim.
    '''
    psf = galsim.Gaussian(flux=1., sigma=sigma_psf)
    image = galsim.ImageF(n_pix_per_side, n_pix_per_side, scale=pixel_scale)
    gsp = galsim.GSParams(maximum_fft_size=10240)
    for i, row in enumerate(cat):
        gal = galsim.Sersic(half_light_radius=row["rhalf"],
                            n=row["sersic"], flux=row[band])
        egal = gal.shear(q=row["q"], beta=row["pa"] * galsim.radians)
        final_gal = galsim.Convolve([psf, egal], gsparams=gsp)
        # place the galaxy and draw it
        x, y = row["x"] + 1, row["y"] + 1
        # sets bounds of galaxy in terms of integar pixel values
        bounds = galsim.BoundsI(x - 0.5*n_pix_per_gal + 1, x + 0.5*n_pix_per_gal - 1,
                                y - 0.5*n_pix_per_gal + 1, y + 0.5*n_pix_per_gal - 1)
        final_gal.drawImage(image[bounds], add_to_image=True)
    return image


def make_header(config, idx_band):
    '''
    Generate header.
    '''
    pixel_scale = config.frames[idx_band]['scale']
    header = {}
    header["CRPIX1"] = 0.0
    header["CRPIX2"] = 0.0
    header["CRVAL1"] = config.ra
    header["CRVAL2"] = config.dec
    header["CD1_1"] = -config.frames[idx_band]['scale'] / 3600.
    header["CD2_2"] = config.frames[idx_band]['scale'] / 3600.
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["FILTER"] = config.frames[idx_band]['band']
    header["NOISE"] = config.frames[idx_band]['noise']
    header["PSFSIGMA"] = config.frames[idx_band]['fwhm']
    header["PIXSCALE"] = config.frames[idx_band]['scale']
    wcs = WCS(header)
    return header, wcs


def write_im(filename, header, noisy, noiseless, uncertainty, truth, cat, **kwargs):
    '''
    Write image to fits file.
    '''
    hdr = fits.Header()
    hdr.update(header)
    hdr.update(EXT1="noisy flux", EXT2="noiseless flux", EXT3="flux uncertainty",
               EXT4="truth", EXT5="source table")
    hdr.update(**kwargs)
    primary = fits.PrimaryHDU(header=hdr)
    with fits.HDUList([primary]) as hdul:
        hdul.append(fits.ImageHDU(noisy, header=hdr))
        # hdul.append(fits.ImageHDU(noiseless, header=hdr))
        # hdul.append(fits.ImageHDU(uncertainty, header=hdr))
        hdul.append(fits.ImageHDU(truth, header=hdr))
        hdul.append(fits.BinTableHDU(cat))
        hdul.writeto(filename, overwrite=True)


def write_centres(name,cat,path_to_file):
    '''
    function to write csv of y and x of centres
    '''
    os.chdir(path_to_file)

    with open(name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(cat["y"])
        writer.writerow(cat["x"])
	
	

if __name__ == "__main__":

    # setup argument parser
    parser = ArgumentParser()
    parser.add_argument("--counter", type=int,
                        default=0)
    parser.add_argument("--config_file", type=str,
                        default="config.yml")
    parser.add_argument("--output_dir", type=str,
                        default="/path_to_dir/")

    # read args
    args = parser.parse_args()

    # read config file
    conf = read_config(args.config_file)

    # set other parameters
    origin = 0
    nhalf = 5

    # draw sources
    n_gal, source_dict = draw_sources(conf)

    # make catalog
    cat = make_catalog(source_dict, n_gal, band=conf.frames[0]['band'], noise=conf.frames[0]['noise'],
                       pixel_scale=conf.frames[0]['scale'])

    n_pix_per_gal = int(np.ceil(nhalf * np.max(cat["rhalf"] / conf.frames[0]['scale']) * 2))
    if (n_pix_per_gal % 2) != 0:
        n_pix_per_gal += 1
    n_pix_per_side_withbound = int(conf.frames[0]['n_pix_per_side'] + 2*n_pix_per_gal)

    # get coordinates
    cat["x"] = np.floor(np.random.uniform(low=1.0 + n_pix_per_gal, high=n_pix_per_side_withbound - n_pix_per_gal - 1.0, size=n_gal)).astype(int)
    cat["y"] = np.floor(np.random.uniform(low=1.0 + n_pix_per_gal, high=n_pix_per_side_withbound - n_pix_per_gal - 1.0, size=n_gal)).astype(int)


    # make image
    im = make_image(cat, n_pix_per_side_withbound, n_pix_per_gal, band=conf.frames[0]['band'],
                    sigma_psf=conf.frames[0]['fwhm'], pixel_scale=conf.frames[0]['scale'])
    noiseless = im.copy().array[int(n_pix_per_gal):-int(n_pix_per_gal), int(n_pix_per_gal):-int(n_pix_per_gal)]
    im.addNoise(galsim.GaussianNoise(sigma=conf.frames[0]['noise']))
    noisy = im.copy().array[int(n_pix_per_gal):-int(n_pix_per_gal), int(n_pix_per_gal):-int(n_pix_per_gal)]

    # convert to image x, y
    cat["x"] -= n_pix_per_gal
    cat["y"] -= n_pix_per_gal


    # generate truth map (1 if center)
    truth = np.zeros_like(noiseless)
    for ii_obj in np.arange(n_gal):
        truth[int(cat["x"][ii_obj]), int(cat["y"][ii_obj])] = 1.0
    truth = truth.T

    # generate uncertainty frame
    uncertainty = conf.frames[0]['noise']*np.ones_like(noiseless)

    # generate header and get WCS
    header, wcs = make_header(conf, idx_band=0)
    header["NPIXGAL"] = n_pix_per_gal
    ra, dec = wcs.all_pix2world(cat["x"], cat["y"], origin)
    cat["ra"] = ra
    cat["dec"] = dec
    

    # write image original
    filename = args.output_dir +"/fits/"+ conf.name + "_{0:05d}.fits".format(args.counter)
    filename_csv = args.output_dir+"/csv/"+ conf.name + "_{0:05d}.csv".format(args.counter)
    
    write_im(filename, header, noisy, noiseless, uncertainty, truth, cat)
    print("successfully wrote image to", filename)
    
    #write centres csv file
    write_centres(filename_csv,cat,args.output_dir)
    print("Successfully wrote centres csv")
