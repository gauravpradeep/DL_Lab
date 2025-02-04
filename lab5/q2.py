import torch
import torch.nn as nn

img = torch.rand(1,1,6,6)
# kernel = kernel.unsqueeze(dim=0)
# kernel = kernel.unsqueeze(dim=0)
conv = nn.Conv2d(1,3,3,stride=1,padding=0)
outimage = conv(img)
print("outimage=", outimage.shape)
print("in image=", img.shape)

'''
outimage= torch.Size([1, 3, 4, 4])
in image= torch.Size([1, 1, 6, 6])

'''

# img = torch.rand(1,1,6,6)
# kernel=torch.rand(3,1,3,3)
# # kernel = kernel.unsqueeze(dim=0)
# # kernel = kernel.unsqueeze(dim=0)
# outimage = F.conv2d(img,kernel,stride=1,padding=0)
# print("outimage=", outimage.shape)
# print("in image=", img.shape)

'''
outimage= torch.Size([1, 3, 4, 4])
in image= torch.Size([1, 1, 6, 6])

'''