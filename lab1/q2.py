import torch
import numpy as np
# Create a 3D tensor with shape (2, 3, 4)
tensor = torch.rand(2).reshape(1, 2, 1)
print("Original Tensor (2x3x4):\n", tensor)
permuted_tensor = tensor.permute(2, 1, 0)
print("\nPermuted Tensor (4x3x2):\n", permuted_tensor)

print("INDEXING")
print(tensor[0][1])

print("NUMPY")
np_arr=np.array([1,2,3])
print(np_arr)
tensor=torch.from_numpy(np_arr)
print(tensor)
print(tensor.cpu().numpy())

print("RANDOM")
tensor1=torch.rand(49).reshape(7,7)
print(tensor1)

print("multiplication")
tensor2=torch.rand(7).reshape(1,7).T
print(torch.matmul(tensor1,tensor2))

print("GPU")
device = 'cuda'
tensor1=torch.rand(6).reshape(2,3).to(device)
tensor2=torch.rand(6).reshape(2,3).to(device)
result=torch.matmul(tensor1,tensor2.T)
print(result)

print("MAX & INDICES")
print(tensor1.max())
print(tensor1.min())

print(tensor1.argmax())
print(tensor1.argmin())
