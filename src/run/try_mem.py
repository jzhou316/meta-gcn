import torch
from misc import print_cuda_mem
from tqdm import tqdm

a = torch.rand(100000, 64).cuda()
a.requires_grad = True
index = torch.randint(100000, (3000000,)).cuda()
print_cuda_mem()

'''
##### method 1: use torch.index_select
# max: 800 MiB
b = torch.index_select(a, 0, index)
print_cuda_mem()
l = b.sum()
print_cuda_mem()
l.backward()
print_cuda_mem()

'''
##### method 2: use torch.nn.functional.embedding
# max: 1600 MiB
b = torch.nn.functional.embedding(index, a)
print_cuda_mem()
l = b.sum()
print_cuda_mem()
l.backward()
print_cuda_mem()

'''
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
