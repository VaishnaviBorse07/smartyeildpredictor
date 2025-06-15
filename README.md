# 🌾 Agriculture Crop Yield Prediction

This project predicts crop yield using machine learning based on historical data. It includes data preprocessing, model training, and a simple interface to make predictions using a trained model.

## 📁 Project Structure

```
agriculture/
│
├── app.py                     # Main application file (e.g., Flask app)
├── crop_yield.csv            # Dataset containing crop yield data
├── feature_importance.png    # Feature importance plot of trained model
├── train_model.py            # Script to train the ML model
└── yield_model.pkl           # Trained model saved using pickle
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/VaishnaviBorse07/agriculture-yield-prediction.git
cd agriculture
```

### 2. Install Dependencies
Make sure you have Python 3.x installed. Then, install required packages:
```bash
pip install -r requirements.txt
```

If `requirements.txt` is not provided, these are likely dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn flask
```

### 3. Train the Model
```bash
python train_model.py
```

### 4. Run the Web App
```bash
python app.py
```

The app will be hosted at `http://127.0.0.1:5000/`

## 📊 Dataset

The dataset `crop_yield.csv` includes historical crop yield data with various features (e.g., rainfall, temperature, soil quality). This data is used to train a regression model.

## 🧠 Model

- The machine learning model is trained using `scikit-learn`.
- The model is saved as `yield_model.pkl`.
- Feature importance is visualized in `feature_importance.png`.

## 📸 Screenshot

![Feature Importance](feature_importance.png)

## 📌 Future Improvements

- Add UI for file upload and prediction.
- Integrate more datasets (e.g., weather APIs).
- Improve model accuracy using advanced techniques (e.g., XGBoost, Hyperparameter tuning).

## 📜 License

This project is licensed under the MIT License.

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
