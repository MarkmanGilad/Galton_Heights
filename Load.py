import torch
import pandas as pd
# gender 0 = male; 1 = female
File_Name = "Data/Galton_DataFrame.pth"
df = torch.load(File_Name)
print(df)

tensor = torch.tensor(df.values, dtype=torch.float32)
print(tensor[0:5])
print(tensor.shape)