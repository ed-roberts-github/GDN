# Image simulation code

This directory contains code used to generate the training data.

To run a generation you call ./run_gen.sh


config.yml is a yaml file which contains the dictionary defining the generation parameters.

gen_im_csv.py is the python code used to generate images. This code is adapted from code given to me by Sandro Tacchella.

run_gen.sh is the shell script called when you want to run a generation.

slurm_submit.peta4-cclake is slurm script used to submit a generation when run the generation on the peta4 cluster.