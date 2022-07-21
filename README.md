This repository prowides Dynamic Time Warping (DTW) computation on CUDA. It is based on modified codes of tslearn library with the jit part rewritten and adapted to cuda.jit
Cuda-based computations reduced the calculation time of DTW matrix (30 seconds instead of initial 2 hours an Nvidia Titan for (4000, 12, 3) size dataset).
DTW can be computed for two sequences (see mytest_single.py as an example), as well as for a dataset (see mytest_matrix.py).
Both functions can be used with the Sacoe-Shiba and Itakura constraints (see original tslearn library for explications.)

At the moment, DTW-matrix can not be computed for two datasets : cdist_dtw_cuda(x1, x2) will give the wrong results.
Only cdist_dtw_cuda(x1) is possible at the moment, which gives us symmetric square matrix.
Constraints parameters are used as in the original tslearn library:
cdist_dtw_cuda(x1, sakoe_chiba_radius=...) or cdist_dtw_cuda(x1, itakura_max_slope=...)

Folder tslearn contains the original tslearn library codes that are partly used in my codes, as well as for the time comparing tests.
Folder tslearn_cuda contains my codes, where metrcis_cuda contains correct cuda-rewritten codes and metrcis_cuda_broken contains the temptations for DTW-matrix computation for 2 datasets that does not work yet.

Enjoy!
