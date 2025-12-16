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
print("After removing invalid area/production:", df.shape)
df = df.reset_index(drop=True)

print("\nFinal dataset info:")
print(df.info())

print("\nBasic statistics:")
print(df[['Area', 'Production']].describe())

df.to_csv("../data/cropds_cleaned.csv", index=False)
