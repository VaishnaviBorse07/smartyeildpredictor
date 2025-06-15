import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load model and mappings
with open("yield_model.pkl", "rb") as f:
    model, crop_mapping, state_mapping, reverse_crop_mapping, reverse_state_mapping = pickle.load(f)

# Load original data
df = pd.read_csv(r"E:\Vaishu coding\Python\AI&ML\agriculture\crop_yield.csv")
df.dropna(inplace=True)
df = df[(df['Area'] > 0) & (df['Production'] > 0)]
df['Yield'] = df['Production'] / df['Area']

# Convert categorical for consistency
df['Crop'] = df['Crop'].astype('category')
df['State'] = df['State'].astype('category')

# Streamlit config
st.set_page_config(page_title="Smart Agri Yield Predictor", page_icon="🌿", layout="wide")

st.markdown("<h1 style='text-align:center; color:green;'>🌾 Smart Agricultural Yield Predictor</h1>", unsafe_allow_html=True)
st.markdown("Predict crop yield using environmental and agricultural inputs, and explore trends across states.")

# Sidebar input form
st.sidebar.header("📌 Prediction Inputs")
crop_name = st.sidebar.selectbox("Select Crop", list(reverse_crop_mapping.keys()))
state_name = st.sidebar.selectbox("Select State", list(reverse_state_mapping.keys()))
year = st.sidebar.slider("Select Year", 1997, 2020, 2010)
area = st.sidebar.number_input("Enter Area (hectares)", min_value=1.0)
rainfall = st.sidebar.number_input("Enter Annual Rainfall (mm)", min_value=0.0)
fertilizer = st.sidebar.number_input("Enter Fertilizer Used (kg/hectare)", min_value=0.0)
pesticide = st.sidebar.number_input("Enter Pesticide Used (kg/hectare)", min_value=0.0)

if st.sidebar.button("Predict Yield"):
    # Correct the mapping for crop and state codes
    crop_code = reverse_crop_mapping[crop_name]
    state_code = reverse_state_mapping[state_name]

    # Prepare input data with the correct feature names expected by the model
    input_data = pd.DataFrame([[crop_code, state_code, year, area, rainfall, fertilizer, pesticide]],
        columns=["Crop_Code", "State_Code", "Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"])
    
    # Predict the yield
    prediction = model.predict(input_data)[0]
    st.success(f"🌱 **Predicted Yield:** {prediction:.2f} units per hectare")

# ===============================
# 📊 Section: Yield Trend Graph
# ===============================
st.markdown("### 📊 Yield Trend Over the Years")
selected_state = st.selectbox("Select a State to View Trends", df['State'].unique())
selected_crop = st.selectbox("Select a Crop", df['Crop'].unique())

trend_df = df[(df['State'] == selected_state) & (df['Crop'] == selected_crop)]

fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=trend_df, x="Crop_Year", y="Yield", marker="o", ax=ax1)
ax1.set_title(f"Yield of {selected_crop} in {selected_state} (1997–2020)")
ax1.set_ylabel("Yield (Production / Area)")
ax1.set_xlabel("Year")
st.pyplot(fig1)

# ===============================
# 🏆 Section: Top 5 Crops by Total Production
# ===============================
st.markdown("### 🏆 Top 5 Crops by Total Production in Selected State")

# Filter data for the selected state
state_data = df[df['State'] == selected_state]

# Group by crop and sum the production
top5 = state_data.groupby('Crop')['Production'].sum().sort_values(ascending=False).head(5)

# Plot
fig2, ax2 = plt.subplots(figsize=(10, 6))
top5.plot(kind='barh', color='seagreen', ax=ax2)
ax2.set_xlabel('Production')
ax2.set_title(f"Top 5 Crops by Total Production in {selected_state}")
ax2.invert_yaxis()  # To display the highest at the top
plt.tight_layout()
st.pyplot(fig2)

# ===============================
# 📌 Section: Feature Importance
# ===============================
st.markdown("### 📌 Feature Importance (Model)")
try:
    image = Image.open("feature_importance.png")
    st.image(image, caption="Feature Importance Graph", use_column_width=True)
except FileNotFoundError:
    st.warning("Feature importance graph not found. Run `train_model.py` again to generate it.")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.markdown("<p style='text-align:center;'>👨‍💻 Developed by <b>TEAM AVINYA</b> </p>", unsafe_allow_html=True)
