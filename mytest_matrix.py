import numpy as np
import time, datetime
import sys
from tslearn_cuda.metrics_cuda import cdist_dtw_cuda
from tslearn.metrics import cdist_dtw as cdist_dtw_cpu
np.set_printoptions(threshold=sys.maxsize)

from numba import cuda
print(cuda.gpus)


# Print stats about current calculations to console and to file
def print_stats(stats_file, text, print_to_console=True):
    with open(stats_file, 'a') as f:
        if isinstance(text, list):
            for t in text:
                f.write(t + "\n")
                if print_to_console:
                    print(t)
        else:
            f.write(text + "\n")
            if print_to_console:
                print(text)
    f.close()


#nbr - number of sequences
#t - number of timestamps
#f - number of features

stats_file = "statistics.txt"
params = [[7, 12, 5], [3751, 24, 4], [3751, 24, 5], [3550, 12, 4], [3550, 12, 5], [3739, 24, 4], [3739, 24, 5], [10725, 12, 4], [10725, 12, 5], [12236, 24, 4], [12236, 24, 5]]
nbr, t, f = params[0]
print_stats(stats_file, "objects=" + str(nbr) + ", timestamps=" + str(t) + ", features=" + str(f),
        print_to_console=True)

# Creating a dataset
x1 = np.arange(nbr*t*f).reshape(nbr,t,f)
x1 = np.asarray(x1, dtype=np.float64)*1.55

x2 = np.arange(nbr*(t-1)*f).reshape(nbr,(t-1),f)
x2 = np.asarray(x2, dtype=np.float64)*2

# Starting CPU
start = time.time()
matrix_cpu = cdist_dtw_cpu(x1, n_jobs=6, sakoe_chiba_radius=2)
#print(matrix_cpu[6][5])
print(matrix_cpu)

end = time.time()
computation_time = str(datetime.timedelta(seconds=end-start))
print_stats(stats_file, "Computation time CPU " + computation_time, print_to_console=True)



# Starting GPU
start = time.time()
#matrix = cdist_dtw_cuda(x1, x2, sakoe_chiba_radius=2)
matrix_gpu = cdist_dtw_cuda(x1, sakoe_chiba_radius=2)
#print(matrix[6][5])
print(matrix_gpu)

end = time.time()
computation_time = str(datetime.timedelta(seconds=end-start))

print_stats(stats_file, "Computation time GPU " + computation_time, print_to_console=True)

print(np.equal(matrix_cpu, matrix_gpu))


