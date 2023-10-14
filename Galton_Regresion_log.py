import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd

#prepair data
File_Name = "Data/Galton_DataFrame.pth"
df = torch.load(File_Name)
print(df)

tensor = torch.tensor(df.values, dtype=torch.float32)
print(tensor[0:6])
X = tensor[:,1:6] # slice: father, mother, children, childNum, gender
Y = tensor[:,6:] # childHeight
print(X[0:5])
print(Y[0:5])


# convert from inches to centimeters
inches = torch.tensor([2.54, 2.54, 1, 1, 1]).reshape([1,-1])
X = X*inches
Y = Y*2.54
print(X[0:5])
print(Y[0:5])
print ("shape x, y",X.shape, Y.shape)
# exit()

# Normalized data
min, minIndex = X.min(dim=0)
max, maxIndex = X.max(dim=0)
print (min, minIndex)
print (max, maxIndex)
def Normalize_minMax(X):
  return (X - min) / (max - min)

def UnNormalize_minMax(X):
  return X * (max-min) + min

X = Normalize_minMax(X)
print(X[0:5])

learning_rate = 0.01
epochs = 50000
losses = []

# design model
in_features = X.shape[1]
out_features=1
Model = nn.Sequential(
  nn.Linear(in_features, 10, bias=True),
  nn.Sigmoid(),
  nn.Linear(10, 32, bias=True),
  nn.Sigmoid(),
  nn.Linear(32, 1, bias=True),
)


#construct loss and optimizer
Loss = nn.MSELoss()

# init optimizer
# optim = torch.optim.SGD(Model.parameters(), lr=learning_rate)
optim = torch.optim.Adam(Model.parameters(), lr=learning_rate)

for epoch in range(epochs):
  # forward
  Y_predict = Model(X)

  # backward
  loss = Loss(Y_predict, Y)
  loss.backward()

  # update wights
  optim.step()  

  if epoch % 10 == 0:
    print(f"epoch= {epoch} loss={loss.item():.9f} ")
    # print (Y_predict)
    losses.append(loss.item())
  

  # zero grads
  optim.zero_grad()

# Plot results
torch.save(Model, 'Data/Model1.pth')
test = torch.tensor([174, 149,2,1,1], dtype=torch.float32)
test = Normalize_minMax(test)
with torch.no_grad():
    print (Model(test))
losses = list(filter(lambda k:  0< k <= 30, losses ))
plt.plot(losses)
plt.show()