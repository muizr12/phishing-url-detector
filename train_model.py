
import pandas as pd
from sklearn.model_selection import train_test_split

# Load extracted features
df = pd.read_csv("data/phishdataset_with_features.csv")

# Remove duplicate URLs
df = df.drop_duplicates(subset=["URLs"])

# Separate input features and target labels
X = df.drop(columns=["URLs", "Labels"])
y = df["Labels"]

# Convert categorical TLD values into numerical features
X = pd.get_dummies(X, columns=["TLD"], dtype=int)

# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Total URLs:", len(df))
print("Training URLs:", len(X_train))
print("Testing URLs:", len(X_test))
print("Features after TLD encoding:", X.shape[1])

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train using the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

from sklearn.metrics import confusion_matrix, classification_report

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Phishing"]
))

import pandas as pd

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

print("\nTop 20 Most Important Features:")
print(importance.sort_values(ascending=False).head(20))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Create a Logistic Regression model
logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

# Train it on the same training data
logistic_model.fit(X_train, y_train)

# Test it on the same testing data
logistic_predictions = logistic_model.predict(X_test)

print("\nLogistic Regression Accuracy:")
print(
    f"{accuracy_score(y_test, logistic_predictions) * 100:.2f}%"
)

import joblib

joblib.dump(
    {"model": model, "features": X.columns.tolist()},
    "phishing_model.pkl"
)

print("\nModel saved successfully!")