# Galaxy Detection Network (GDN)

This is where most of the core code used to make GDN is. 

To run an inference you need to run 'python -m inference.run_inference' when inside the /model/ directory. You'll need to change the config dictionary inside run_inference.py for this to work.


## Files and directories
/model contains code used to build, train and make predictions from the model.


    /model/train.py is the main python script which is called when training. In its is a config dictionary which define 
    
    /model/pl_model.py is the python code which builds the main class of the pytorch lightning model. This class defines the pytorch lightning code which explains how the pytorch lightning code is used in training, i.e. the hooks that link into the pytorch lightning code that run when training.

    /model/model.py is the python script which builds the main pytorch model class. It subclasses the pytorch lightning model and describes the loss functions and how data flows through the layers.

    /model/dataset.py defines the dataloader. This describes how data is laoded into the model when training.

    /model/__init__.py is just a python file required so I can use my code as modules, i.e. so I can import functions into other scrips


    /model/preprocessing/ this is the directory which contain code used to preprocess my simulated dataset

        /model/preprocessing/functions.py contains all my functions called when preprocessing.

        /model/preprocessing/main.py is the python code called when running the preporcessing


    /model/inference/ is the directory which contains files used for running model predictions and metrics.

        /model/inference/ceers_img.py is python code used to create sections of 224x224 parts from a real ceers fits file.

        /model/inference/infer_functions.py contains all main functions used for inferenceing

        /model/inference/run_inference.py runs an inference of the mode. It contains a config dictionary which can be used to define what file you wish to run an inference on. There are also two blocks of code: one for runnin gon simulated data and one for running on real images.

        /model/inference/run_metrics.py runs measurements of completeness and purity over a range of r_tp

        /model/inference/run_snr_metrics.py runs measurements over a range of SNR

        /model/inference/sep.py runs an inference of the SExtractor model

        /model/inference/plots.py makes plots from my data



/onnx contains some of the saved model versions which were experimented with more thoroughly than intial runs. The optimum model, and model GDN uses, is model-195.onnx.

/misc contains some messy code and notebooks used when building the model.

/data contains a few checkpoints and logs.

/wandb contains checkpoints and logs saved by the wanb logger.

slurm_submit.wilkes3 is the script used to submit a training run when on the hpc. (Please note this obviously won't work here as this was a remote file saved and run on the hpc). run.sh is a script called by slurm_submit.wilkes3 to run the python code.

slurm_infer.wilkes3 is a script used to run an inference on the hpc, with an accompanying reun_infer.sh used to call the appopriate python code.


