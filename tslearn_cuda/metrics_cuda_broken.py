"""
The :mod:`tslearn.metrics` module gathers time series similarity metrics.
"""

import warnings
import numpy, cmath, numba
import math
from numba import njit, prange, cuda, jit


from tslearn_cuda.utils import to_time_series, to_time_series_dataset, ts_size, \
    check_equal_size

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

GLOBAL_CONSTRAINT_CODE = {None: 0, "": 0, "itakura": 1, "sakoe_chiba": 2}
VARIABLE_LENGTH_METRICS = ["dtw", "gak", "softdtw"]

SH_LIST = numpy.arange(10000)
SH1 = 0
SH2 = 0


def cdist_dtw_cuda(x1, x2= None, global_constraint=0, sakoe_chiba_radius=None, itakura_max_slope=None):
    x1 = to_time_series_dataset(x1)
    if x2 is not None:
        x2 = to_time_series_dataset(x2)
        mask = compute_mask(x1[0], x2[0], global_constraint=global_constraint, sakoe_chiba_radius=sakoe_chiba_radius,
                            itakura_max_slope=itakura_max_slope)
        matrix = numpy.zeros((len(x1), len(x2)), dtype=numpy.float64)
        x2 = cuda.to_device(x2)
    else:
        mask = compute_mask(x1[0], x1[0], global_constraint=global_constraint, sakoe_chiba_radius=sakoe_chiba_radius, itakura_max_slope=itakura_max_slope)
        matrix = numpy.zeros((len(x1), len(x1)), dtype=numpy.float64)

    x1 = cuda.to_device(x1)
    matrix = cuda.to_device(matrix)
    mask = cuda.to_device(mask)
    # it is a small trick to make SH a global constant
    # otherwise it does not work in cudajit cuda.local.array() function
    def funс_sh():
        global SH1, SH2
        if x2 is None:
            SH1 = int(x1.shape[1] + 1)
            return SH1
        else:
            SH1 = int(x1.shape[1] + 1)
            SH2 = int(x2.shape[1] + 1)
            return SH1, SH2
    funс_sh()
    print(SH1, SH2)
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(matrix.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(matrix.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    # increment_a_2D_array[blockspergrid, threadsperblock](an_array)
    cdist_dtw[blockspergrid, threadsperblock](x1, x2, matrix, mask); cuda.synchronize()
    # matrix_res = numpy.zeros((len(x1), len(x1)), dtype=numpy.float64)
    matrix_res = numpy.asarray(matrix)
    # matrix = matrix_new
    if x2 is None:
        matrix = matrix_res + matrix_res.T
    else:
        matrix = matrix_res
    return matrix




# @cuda.jit
# def dtw_cuda(dataset, matrix, i, j):
#     s1 = dataset[i]
#     s2 = dataset[j]
#     matrix[i][j]=i+j



@cuda.jit()
def cdist_dtw(dataset1, dataset2, matrix, mask):
    if dataset2 is None:
        dataset = dataset1
        cum_sum = cuda.local.array(shape=(SH1, SH1), dtype=numba.float64)
        i, j = cuda.grid(2)
        # print(matrix.shape[1])
        if i < matrix.shape[0] and j>=i+1 and j < matrix.shape[1]:
            for c1 in range(len(cum_sum)):
                for c2 in range(len(cum_sum[0])):
                    cum_sum[c1][c2] = numpy.inf
            s1 = dataset[i]
            s2 = dataset[j]

            l1 = s1.shape[0]
            l2 = s2.shape[0]
            cum_sum[0, 0] = 0


            for i1 in prange(l1):
                for j1 in prange(l2):
                    if not (mask[i1, j1] == numpy.inf):
                        # print(mask[i, j])
                        # if not (mask[i, j] == numpy.inf):
                        x, y = s1[i1], s2[j1]
                        dist = 0.
                        for di in prange(x.shape[0]):
                            diff = (x[di] - y[di])
                            dist += diff * diff
                        cum_sum[i1 + 1, j1 + 1] = dist
                        cum_sum[i1 + 1, j1 + 1] += min(cum_sum[i1, j1 + 1], cum_sum[i1 + 1, j1], cum_sum[i1, j1])
            cum_sum = cum_sum[1:, 1:]
            matrix[i][j] = cum_sum[-1, -1]**0.5

    else:
        cum_sum = cuda.local.array(shape=(SH1, SH2), dtype=numba.float64)
        i, j = cuda.grid(2)
        # print(matrix.shape[1])
        for c1 in range(len(cum_sum)):
            for c2 in range(len(cum_sum[0])):
                cum_sum[c1][c2] = numpy.inf
        s1 = dataset1[i]
        s2 = dataset2[j]

        l1 = s1.shape[0]
        l2 = s2.shape[0]
        cum_sum[0, 0] = 0

        for i1 in prange(l1):
            for j1 in prange(l2):
                if not (mask[i1, j1] == numpy.inf):
                    # print(mask[i, j])
                    # if not (mask[i, j] == numpy.inf):
                    x, y = s1[i1], s2[j1]
                    dist = 0.
                    for di in prange(x.shape[0]):
                        diff = (x[di] - y[di])
                        dist += diff * diff
                    cum_sum[i1 + 1, j1 + 1] = dist
                    cum_sum[i1 + 1, j1 + 1] += min(cum_sum[i1, j1 + 1], cum_sum[i1 + 1, j1], cum_sum[i1, j1])
        cum_sum = cum_sum[1:, 1:]
        matrix[i][j] = cum_sum[-1, -1] ** 0.5




def compute_mask(s1, s2, global_constraint=0,
                 sakoe_chiba_radius=None, itakura_max_slope=None):
    """Compute the mask (region constraint).

    Parameters
    ----------
    s1 : array
        A time series.

    s2: array
        Another time series.

    global_constraint : {0, 1, 2} (default: 0)
        Global constraint to restrict admissible paths for DTW:
        - "itakura" if 1
        - "sakoe_chiba" if 2
        - no constraint otherwise

    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        If None and `global_constraint` is set to 2 (sakoe-chiba), a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to 1 (itakura), a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    Returns
    -------
    mask : array
        Constraint region.
    """
    sz1 = s1.shape[0]
    sz2 = s2.shape[0]
    if (global_constraint == 0 and sakoe_chiba_radius is not None
            and itakura_max_slope is not None):
        raise RuntimeWarning("global_constraint is not set for DTW, but both "
                             "sakoe_chiba_radius and itakura_max_slope are "
                             "set, hence global_constraint cannot be inferred "
                             "and no global constraint will be used.")
    if global_constraint == 2 or (global_constraint == 0
                                  and sakoe_chiba_radius is not None):
        if sakoe_chiba_radius is None:
            sakoe_chiba_radius = 1
        mask = sakoe_chiba_mask(sz1, sz2, radius=sakoe_chiba_radius)
    elif global_constraint == 1 or (global_constraint == 0
                                    and itakura_max_slope is not None):
        if itakura_max_slope is None:
            itakura_max_slope = 2.
        mask = itakura_mask(sz1, sz2, max_slope=itakura_max_slope)
    else:
        mask = numpy.zeros((sz1, sz2))
    return mask


@njit()
def sakoe_chiba_mask(sz1, sz2, radius=1):
    """Compute the Sakoe-Chiba mask.

    Parameters
    ----------
    sz1 : int
        The size of the first time series

    sz2 : int
        The size of the second time series.

    radius : int
        The radius of the band.

    Returns
    -------
    mask : array, shape = (sz1, sz2)
        Sakoe-Chiba mask.

    Examples
    --------
    >>> sakoe_chiba_mask(4, 4, 1)
    array([[ 0.,  0., inf, inf],
           [ 0.,  0.,  0., inf],
           [inf,  0.,  0.,  0.],
           [inf, inf,  0.,  0.]])
    >>> sakoe_chiba_mask(7, 3, 1)
    array([[ 0.,  0., inf],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [inf,  0.,  0.]])
    """
    mask = numpy.full((sz1, sz2), numpy.inf)
    if sz1 > sz2:
        width = sz1 - sz2 + radius
        for i in prange(sz2):
            lower = max(0, i - radius)
            upper = min(sz1, i + width) + 1
            mask[lower:upper, i] = 0.
    else:
        width = sz2 - sz1 + radius
        for i in prange(sz1):
            lower = max(0, i - radius)
            upper = min(sz2, i + width) + 1
            mask[i, lower:upper] = 0.
    return mask


@njit()
def _njit_itakura_mask(sz1, sz2, max_slope=2.):
    """Compute the Itakura mask without checking that the constraints
    are feasible. In most cases, you should use itakura_mask instead.

    Parameters
    ----------
    sz1 : int
        The size of the first time series

    sz2 : int
        The size of the second time series.

    max_slope : float (default = 2)
        The maximum slope of the parallelogram.

    Returns
    -------
    mask : array, shape = (sz1, sz2)
        Itakura mask.
    """
    min_slope = 1 / float(max_slope)
    max_slope *= (float(sz1) / float(sz2))
    min_slope *= (float(sz1) / float(sz2))

    lower_bound = numpy.empty((2, sz2))
    lower_bound[0] = min_slope * numpy.arange(sz2)
    lower_bound[1] = ((sz1 - 1) - max_slope * (sz2 - 1)
                      + max_slope * numpy.arange(sz2))
    lower_bound_ = numpy.empty(sz2)
    for i in prange(sz2):
        lower_bound_[i] = max(round(lower_bound[0, i], 2),
                              round(lower_bound[1, i], 2))
    lower_bound_ = numpy.ceil(lower_bound_)

    upper_bound = numpy.empty((2, sz2))
    upper_bound[0] = max_slope * numpy.arange(sz2)
    upper_bound[1] = ((sz1 - 1) - min_slope * (sz2 - 1)
                      + min_slope * numpy.arange(sz2))
    upper_bound_ = numpy.empty(sz2)
    for i in prange(sz2):
        upper_bound_[i] = min(round(upper_bound[0, i], 2),
                              round(upper_bound[1, i], 2))
    upper_bound_ = numpy.floor(upper_bound_ + 1)

    mask = numpy.full((sz1, sz2), numpy.inf)
    for i in prange(sz2):
        mask[int(lower_bound_[i]):int(upper_bound_[i]), i] = 0.
    return mask


def itakura_mask(sz1, sz2, max_slope=2.):
    """Compute the Itakura mask.

    Parameters
    ----------
    sz1 : int
        The size of the first time series

    sz2 : int
        The size of the second time series.

    max_slope : float (default = 2)
        The maximum slope of the parallelogram.

    Returns
    -------
    mask : array, shape = (sz1, sz2)
        Itakura mask.

    Examples
    --------
    >>> itakura_mask(6, 6)
    array([[ 0., inf, inf, inf, inf, inf],
           [inf,  0.,  0., inf, inf, inf],
           [inf,  0.,  0.,  0., inf, inf],
           [inf, inf,  0.,  0.,  0., inf],
           [inf, inf, inf,  0.,  0., inf],
           [inf, inf, inf, inf, inf,  0.]])
    """
    mask = _njit_itakura_mask(sz1, sz2, max_slope=max_slope)

    # Post-check
    raise_warning = False
    for i in prange(sz1):
        if not numpy.any(numpy.isfinite(mask[i])):
            raise_warning = True
            break
    if not raise_warning:
        for j in prange(sz2):
            if not numpy.any(numpy.isfinite(mask[:, j])):
                raise_warning = True
                break
    if raise_warning:
        warnings.warn("'itakura_max_slope' constraint is unfeasible "
                      "(ie. leads to no admissible path) for the "
                      "provided time series sizes",
                      RuntimeWarning)

    return mask
