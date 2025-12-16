import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def main():
    df = pd.read_csv("../data/cropds_cleaned.csv")
    df = df.reset_index(drop=True)

    district_encoder = LabelEncoder()
    season_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()

    df['District_Name'] = district_encoder.fit_transform(df['District_Name'])
    df['Season'] = season_encoder.fit_transform(df['Season'])
    df['Crop'] = crop_encoder.fit_transform(df['Crop'])

    print(df.head())
    print("\nData types after encoding:")
    print(df.dtypes)

    X = df.drop('Crop', axis=1)
    y = df['Crop']

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Number of crop classes:", y.nunique())

    X.to_csv("../data/X_transformed.csv", index=False)
    y.to_csv("../data/y_labels.csv", index=False)


    print("\nCrop label mapping:")
    for i, crop in enumerate(crop_encoder.classes_):
        print(f"{i} -> {crop}")

    joblib.dump(district_encoder, "../models/district_encoder.pkl")
    joblib.dump(season_encoder, "../models/season_encoder.pkl")
    joblib.dump(crop_encoder, "../models/crop_encoder.pkl")

if __name__ == "__main__":
    main()







""" Crop label mapping:
0 -> Bajra
1 -> Coconut
2 -> Cotton(lint)
3 -> Groundnut
4 -> Jowar
5 -> Maize
6 -> Moong(Green Gram)
7 -> Niger seed
8 -> Paddy
9 -> Ragi
10 -> Rice
11 -> Sesamum
12 -> Soyabean
13 -> Sugarcane
14 -> Sunflower
15 -> Tur
16 -> Urad
17 -> Wheat """