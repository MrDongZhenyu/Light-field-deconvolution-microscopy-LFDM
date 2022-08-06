# Light field deconvolution microscopy(LFDM)

### This is an executable matlab example of Light field deconvolution microscopy revised from the software in article:  Prevedel, R., Yoon, Y., Hoffmann, M. et al. Simultaneous whole-animal 3D imaging of neuronal activity using light-field microscopy. Nat Methods 11, 727–730 (2014). <https://doi.org/10.1038/nmeth.2964>

### Tested by Matlab 2019b and GPU Nvidia Geforce MX350

### This package contains:
1. Main_LFDM.m : Main script
2. ComputePSF.m : Computes Light field PSF according to [article](https://opg.optica.org/oe/fulltext.cfm?uri=oe-21-21-25418&id=269805)
3. Reconstruction3D.m : RL-deconvolution
4. utils for psf/main_LFDM/deconvolution
