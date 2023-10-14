import pandas as pd
import torch
import numpy as np

FILE_NAME = 'Data/galton.csv'
df = pd.read_csv(FILE_NAME)
# print(df.to_string())
print(df.dtypes)
print(df.to_string())
print(df['family'].loc[609:616])
df['family'].loc[609:616]=205
print(df['family'].loc[609:616])
print(df.to_string())


df["family"] = df['family'].astype(int)
df["gender"] = np.where(df["gender"] == "female", 1, 0)

print(df.dtypes)
print(df.to_string())

array_np = df.to_numpy(dtype=np.float32)
print(array_np)
print(array_np.dtype)
torch.save(df,"Data/Galton_DataFrame.pth")

# data = np.loadtxt(FILE_NAME, delimiter=',')
# print (data)

# t = torch.tensor(df)
# print(t)

