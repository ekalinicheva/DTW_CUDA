import numpy as np
import time, datetime
import sys

from tslearn_cuda.metrics_cuda_broken import dtw_cuda
from tslearn.metrics import dtw as dtw_cpu
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


stats_file = "statistics.txt"


# We create 2 sequences
t = 300   # number of timestamps
f = 3     # number of features
x = np.arange(t * f).reshape(t, f)
y = np.asarray(x, dtype=np.float64) * 1.55


print_stats(stats_file, "timestamps=" + str(t) + ", features=" + str(f),
        print_to_console=True)


# Starting CPU
start = time.time()
dist = dtw_cpu(x, y)
print(dist)
end = time.time()
computation_time = str(datetime.timedelta(seconds=end-start))
print_stats(stats_file, "Computation time CPU " + computation_time, print_to_console=True)

# Starting GPU
start = time.time()
dist = dtw_cuda(x, y)
print(dist)
end = time.time()
computation_time = str(datetime.timedelta(seconds=end-start))

print_stats(stats_file, "Computation time GPU " + computation_time, print_to_console=True)


