import torch
import numpy as np

a = torch.FloatTensor(3, 2)
print(a)

b = torch.FloatTensor(3, 2, 2)
print(b)

# In Pytorch xxx_ is inplace function
a.zero_()
print(a)

# Create Tensor from Numpy
n = np.zeros(shape = (2,3) , dtype = np.float32)
torch.tensor(n)
print(n)

# Gradients properties
v1 = torch.tensor([1.0, 1.0], requires_grad = True)
print(v1)