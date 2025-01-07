import torch
import numpy as np
tensor = torch.arange(4)
print("Original Tensor (4x4):\n", tensor)

reshaped_tensor = tensor.reshape(2,2)
print("\nReshaped Tensor (2x8):\n", reshaped_tensor)

viewed_tensor = reshaped_tensor.view(-1)
print("\nViewed Tensor (1D):\n", viewed_tensor)
stacked_tensor = torch.stack([tensor, tensor], dim=0)  # Stack along the first dimension
print("\nStacked Tensor (2x4x4):\n", stacked_tensor)

unsqueezed_tensor = tensor.unsqueeze(0)
print("\nUnsqueezed Tensor:\n", unsqueezed_tensor)  # Now it has shape (1, 4)

squeezed_tensor = unsqueezed_tensor.squeeze()
print("\nSqueezed Tensor:\n", squeezed_tensor)  # Now it has shape (4,)

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