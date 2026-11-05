import pandas as pd
import joblib
from feature_extraction import extract_features
from urllib.parse import urlsplit

# Load our trained model
saved = joblib.load("phishing_model.pkl")

model = saved["model"]
feature_columns = saved["features"]

# Ask the user for a URL
url = input("Enter a URL: ").strip()

if not url.startswith(("http://", "https://")):
    print("Invalid URL. Please include http:// or https://")
    exit()

from urllib.parse import urlsplit, urlunsplit

parsed = urlsplit(url)

if not parsed.hostname:
    print("Invalid URL. Please enter a valid website address.")
    exit()

# Extract the same 18 features used during training
features = extract_features(url)
X_new = pd.DataFrame([features])

# Convert TLD into numerical columns
X_new = pd.get_dummies(X_new, columns=["TLD"], dtype=int)

# Match the columns expected by our model
X_new = X_new.reindex(columns=feature_columns, fill_value=0)

# Make prediction
prediction = model.predict(X_new)[0]

probabilities = model.predict_proba(X_new)[0]
confidence = probabilities[int(prediction)] * 100

# Display result
if prediction == 0:
    print("\nPrediction: Legitimate")
else:
    print("\nPrediction: Phishing")

print(f"Model Confidence: {confidence:.2f}%")
