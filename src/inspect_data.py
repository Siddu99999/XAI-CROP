import pandas as pd

df = pd.read_csv("../data/cropds.csv")

print("Shape of dataset:", df.shape)
print("\nColumn names:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())
