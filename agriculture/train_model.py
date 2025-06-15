import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv(r"E:\Vaishu coding\Python\AI&ML\agriculture\crop_yield.csv")  # ✅ Update path if needed
df.dropna(inplace=True)

# Rename columns for consistency
df.rename(columns={
    'Crop_Year': 'Year',
    'State': 'State_Name'
}, inplace=True)

# Filter non-zero area and production
df = df[(df['Area'] > 0) & (df['Production'] > 0)]

# Calculate Yield
df['Yield'] = df['Production'] / df['Area']

# Encode categorical variables
df['Crop'] = df['Crop'].astype('category')
df['State_Name'] = df['State_Name'].astype('category')

df['Crop_Code'] = df['Crop'].cat.codes
df['State_Code'] = df['State_Name'].cat.codes

# Mappings
crop_mapping = dict(enumerate(df['Crop'].cat.categories))
state_mapping = dict(enumerate(df['State_Name'].cat.categories))
reverse_crop_mapping = {v: k for k, v in crop_mapping.items()}
reverse_state_mapping = {v: k for k, v in state_mapping.items()}

# Features and target
X = df[['Crop_Code', 'State_Code', 'Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
y = df['Yield']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"✅ Model R² score: {r2:.4f}")

# Save model and mappings
with open("yield_model.pkl", "wb") as f:
    pickle.dump((model, crop_mapping, state_mapping, reverse_crop_mapping, reverse_state_mapping), f)

print("✅ Model and mappings saved to yield_model.pkl")

# Feature Importance Graph
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("📊 Feature importance graph saved as feature_importance.png")
