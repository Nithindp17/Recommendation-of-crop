# A Machine Learning–Based Crop Recommendation System Using Random Forest Classifier

This project is a machine learning application that recommends the best crop to grow based on soil and weather conditions. It uses a Random Forest Classifier trained on the Crop Recommendation Dataset from Kaggle.

## Folder Structure

```
crop_recommendation/
│
├── Crop_recommendation.csv
├── crop_model.py
├── crop_app.py
├── model.pkl
├── requirements.txt
└── README.md
```

## How to Run the Project

### 1. Setup the Environment

First, make sure you have Python installed. Then, create a virtual environment and install the required dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

### 2. Train the Model

Run the `crop_model.py` script to train the Random Forest model. This will create a `model.pkl` file.

```bash
python crop_model.py
```

The script will also generate a `feature_importance.png` plot to visualize the importance of each feature.

### 3. Run the Streamlit App

Once the model is trained and saved, you can run the Streamlit application.

```bash
streamlit run crop_app.py
```

This will open a new tab in your browser with the Crop Recommendation System dashboard. You can input the soil and weather conditions to get a crop recommendation.

## Deploying to Streamlit Cloud (Optional)

To deploy this app to Streamlit Cloud, follow these steps:

1.  **Push your project to a GitHub repository.** Make sure your repository includes the `crop_app.py`, `model.pkl`, and `requirements.txt` files.
2.  **Sign up for Streamlit Cloud.** If you don't have an account, you can sign up for free at [streamlit.io/cloud](https://streamlit.io/cloud).
3.  **Deploy the app.**
    *   Click on the "New app" button on your Streamlit Cloud dashboard.
    *   Connect your GitHub account and select the repository you created.
    *   Make sure the main file path is set to `crop_app.py`.
    *   Click "Deploy!".

Your app will be deployed and accessible via a public URL.
