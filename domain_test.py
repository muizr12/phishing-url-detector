
import pandas as pd
from urllib.parse import urlsplit
from sklearn.model_selection import GroupShuffleSplit

# Load our extracted dataset
df = pd.read_csv("data/phishdataset_with_features.csv")

# Remove duplicate URLs
df = df.drop_duplicates(subset=["URLs"]).copy()

# Extract the hostname from each URL
df["Hostname"] = df["URLs"].apply(
    lambda url: urlsplit(url).hostname or ""
)

# Separate features and labels
X = df.drop(columns=["URLs", "Labels", "Hostname"])
y = df["Labels"]

# Split by hostname instead of individual URL
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=df["Hostname"])
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Verify that no hostnames overlap
train_hosts = set(df.iloc[train_idx]["Hostname"])
test_hosts = set(df.iloc[test_idx]["Hostname"])

print("Overlapping hostnames:", len(train_hosts & test_hosts))

print("\nTraining URLs:", len(X_train))
print("Testing URLs:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Convert TLDs into numerical columns
X_train = pd.get_dummies(X_train, columns=["TLD"], dtype=int)
X_test = pd.get_dummies(X_test, columns=["TLD"], dtype=int)

# Ensure both datasets have the same feature columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Create and train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predict on URLs from unseen hostnames
y_pred = model.predict(X_test)

print("\nDomain-Based Test Accuracy:")
print(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Phishing"]
))