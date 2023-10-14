import torch

Male, Female = 0 , 1
inche = 2.54
# Normalized data
min = torch.tensor([157.4800, 147.3200,   1.0000,   1.0000,   0.0000])
max = torch.tensor([199.3900, 179.0700,  15.0000,  15.0000,   1.0000])

def Normalize_minMax(X):
  return (X - min) / (max - min)

def UnNormalize_minMax(X):
  return X * (max-min) + min


model = torch.load('Data/Model1.pth')

test = torch.tensor([73*inche, 65*inche,9,4,Female], dtype=torch.float32)
test = Normalize_minMax(test)
with torch.no_grad():
    print (model(test), model(test) / inche )