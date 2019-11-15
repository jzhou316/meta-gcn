import torch
# from torch_geometric.utils import scatter_
import sys
sys.path.append('../models')
from common import scatter_
from misc import print_cuda_mem, timeSince
from tqdm import tqdm
import time

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

a = torch.rand(100000, 64).cuda()
a.requires_grad = True
index = torch.randint(100000, (2, 3000000)).cuda()
print_cuda_mem()

t0 = time.time()
'''
##### method 1: use torch.index_select
# max: 1584 MiB
b = torch.index_select(a, 0, index[0])
print_cuda_mem()
c = scatter_('add', b, index[1], dim_size=a.size(0))
l = c.sum()
print_cuda_mem()
l.backward()
print_cuda_mem()
print('time: ', time.time() - t0)
print(l)
print(a.grad)

'''
##### method 2: use torch.index_select but with for loops
# when n=4, max: 510 MiB
# when n=2, max: ~800 MiB
# when n=8, max: 328 MiB, and even faster?? (0.007 s compared with 0.013 s in method 1)
n = 2
p = 3000000 // n
l = 0
a = a.clone()
norm = torch.rand(p, 1).cuda()
for i in tqdm(range(n)):
    b = torch.index_select(a, 0, index[0][p * i:p * (i + 1)])
    print_cuda_mem()
    b = b * norm
    print_cuda_mem()
    a = scatter_('add', b, index[1][p * i:p * (i + 1)], out=a, dim_size=a.size(0))
#    l += a.sum()
l = a.sum()
print_cuda_mem()
l.backward()
print_cuda_mem()
print('time: ', time.time() - t0)
print(l)
print(a.grad)


'''
##### method 2: use torch.nn.functional.embedding
# max: 1678 MiB
b = torch.nn.functional.embedding(index[0], a)
print_cuda_mem()
c = scatter_('add', b, index[1], dim_size=a.size(0))
l = c.sum()
print_cuda_mem()
l.backward()
print_cuda_mem()


##### method 3: use for loop
# This uses much less memory, but takes too much time! Not only the forward pass,
# the backward also takes way too long.
# Max: 121 MiB

l = 0
for x in tqdm(index):
    l = l + a[x].sum()
print_cuda_mem()
l.backward()
print_cuda_mem()


##### method 4: direct index
# not good, takes too much memory: max 6000 MiB
l = a[index].sum()
print_cuda_mem()
l.backward()
print_cuda_mem()
'''
