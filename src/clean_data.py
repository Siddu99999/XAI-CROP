"""
XAI-CROP: Data Extraction and Preprocessing Module

Loads the agricultural dataset, performs initial inspection,
and cleans the data by handling missing values, removing duplicates,
and filtering invalid records. This module prepares a reliable
and consistent dataset for downstream transformation and modeling.
"""


import pandas as pd
df=pd.read_csv("../data/cropds.csv")
print("Initial shape:", df.shape)

#checking for missing values in the columns
print("\nMissing values per column:")
print(df.isnull().sum())

df = df.dropna()     #removing the rows with missing valued columns
print("After dropping missing values:", df.shape)

#removing duplicate rows
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]

print(f"Removed {before - after} duplicate rows")
print("After removing duplicates:", df.shape)

#considering valid rows only
df = df[(df['Area'] > 0) & (df['Production'] > 0)]
# Clean categorical text columns
df['Season'] = df['Season'].str.strip().str.lower()
df['District_Name'] = df['District_Name'].str.strip().str.upper()
df['Crop'] = df['Crop'].str.strip().str.title()

print("After removing invalid area/production:", df.shape)
df = df.reset_index(drop=True)

print("\nFinal dataset info:")
print(df.info())

print("\nBasic statistics:")
print(df[['Area', 'Production']].describe())

df.to_csv("../data/cropds_cleaned.csv", index=False)
