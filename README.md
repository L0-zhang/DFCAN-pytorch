# DFCAN-pytorch

DFCAN/DFGAN could get an amazing result in SIM  super resolution reconstruction  in optical microscopy. 

Code is basically a pytorch implementation of DFCAN/DFGAN(https://www.nature.com/articles/s41592-020-01048-5),and in reference to the tensorflow/keras implementation  from https://github.com/qc17-THU/DL-SR.  

All credit goes to the authors of DFCAN/DFGAN, listed in the paper“Evaluation and development of deep neural networks for image super-resolution in optical microscopy”.

This implementation is without the process of prctile_norm and recovery.And as a result,the tail in the coding has no sigmoid function. The expected high resolution picture will be acquired from the model directly.


