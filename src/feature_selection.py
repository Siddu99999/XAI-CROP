import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#load cleaned & transformed data
df = pd.read_csv("../data/cropds_cleaned.csv")
from sklearn.preprocessing import LabelEncoder

#encoding
df['District_Name'] = LabelEncoder().fit_transform(df['District_Name'])
df['Season'] = LabelEncoder().fit_transform(df['Season'])
df['Crop'] = LabelEncoder().fit_transform(df['Crop'])

X = df.drop('Crop', axis=1)
y = df['Crop']

#random forest is used for feature ranking(more reliable than single DT)
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

importances = rf.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

plt.figure(figsize=(8,5))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.title("Feature Importance - XAI-CROP")
plt.ylabel("Importance Score")
plt.xlabel("Features")
plt.show()

feature_importance_df.to_csv(
    "../data/feature_importance.csv",
    index=False
)
