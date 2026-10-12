# California House Price Predictor 🏡

A Machine Learning web application that predicts the median house value of a California neighbourhood using a trained Random Forest Regressor.

## 🚀 Features

- Predicts estimated median house value in US Dollars (USD).
- Displays approximate Indian Rupee (INR) conversion.
- Uses a preprocessing pipeline for input transformation.
- Validates user inputs before prediction.
- Responsive web interface built with HTML and CSS.
- Built with Flask and Scikit-learn.

## 🛠️ Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- HTML and CSS

## 📁 Project Structure

```text
california-house-price-predictor/
├── app.py
├── model.pkl
├── pipeline.pkl
├── requirements.txt
├── .python-version
├── README.md
└── templates/
    └── index.html
```

## ⚙️ Run Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/ashish-coding-hub/california-house-price-predictor.git
   ```

2. Open the project directory:

   ```bash
   cd california-house-price-predictor
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the Flask application:

   ```bash
   python app.py
   ```

5. Open `http://127.0.0.1:5000` in your browser.

## 📊 Model

The project uses a trained Random Forest Regressor with a preprocessing pipeline to generate predictions from California Housing dataset features.

The prediction represents the estimated median house value of a neighbourhood, not the price of an individual property. The dataset is based on 1990 California census data, so predictions should not be treated as current market valuations.

## 🌐 Deployment

The application can be deployed using a Python-compatible hosting platform such as Render.

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app`

## 👨‍💻 Author

**Ashish Singh**

- GitHub: [ashish-coding-hub](https://github.com/ashish-coding-hub)
- LinkedIn: [Ashish Singh](https://linkedin.com/in/ashish-singh-4012ab342)
