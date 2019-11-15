import time
import math
import torch


def logging(s, f_log=None, print_=True, log_=True):
    if print_:
        print(s)
    if log_ and f_log is not None:
        f_log.write(s)
        f_log.write('\n')


def timeSince(start):
    now = time.time()
    s = now - start
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    if h == 0:
        if m == 0:
            return '%ds' % s
        else:
            return '%dm %ds' % (m, s)
    else:
        return '%dh %dm %ds' % (h, m, s)


def print_cuda_mem():
    print(f"Cuda Mem (MiB): {torch.cuda.memory_allocated() / (1024 ** 2)}")
    print(f"Cuda Mem Cached (MiB): {torch.cuda.memory_cached() / (1024 ** 2)}")
    print(f"Cuda Mem Max (MiB): {torch.cuda.max_memory_allocated() / (1024 ** 2)}")
    return
