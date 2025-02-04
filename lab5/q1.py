import torch
import torch.nn.functional as F

img = torch.rand(1,1,6,6)
kernel=torch.rand(1,1,3,3)
# kernel = kernel.unsqueeze(dim=0)
# kernel = kernel.unsqueeze(dim=0)
outimage = F.conv2d(img, kernel, stride=1, padding=3)
print("outimage=", outimage.shape)
print("in image=", img.shape)
print("kernel=", kernel.shape)


'''

Stride formula : n=[(n-k)/s]+1
stride : 1 
outimage= torch.Size([1, 1, 4, 4])
in image= torch.Size([1, 1, 6, 6])
kernel= torch.Size([1, 1, 3, 3])

stride : 2
outimage= torch.Size([1, 1, 2, 2])
in image= torch.Size([1, 1, 6, 6])
kernel= torch.Size([1, 1, 3, 3])

stride : 3
same as 2

stride : 4
outimage= torch.Size([1, 1, 1, 1])
in image= torch.Size([1, 1, 6, 6])
kernel= torch.Size([1, 1, 3, 3])

Padding formula : n = n - k + 2p +1

stride : 1 
padding : 1
outimage= torch.Size([1, 1, 6, 6])
in image= torch.Size([1, 1, 6, 6])
kernel= torch.Size([1, 1, 3, 3])

padding : 2
outimage= torch.Size([1, 1, 8, 8])
in image= torch.Size([1, 1, 6, 6])
kernel= torch.Size([1, 1, 3, 3])

padding : 3
outimage= torch.Size([1, 1, 10, 10])
in image= torch.Size([1, 1, 6, 6])
kernel= torch.Size([1, 1, 3, 3])




Combined formula : n=[(n-k+2p)/s] + 1
'''