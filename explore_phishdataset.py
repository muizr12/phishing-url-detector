
import pandas as pd
import numpy as np
import math
import re
import ipaddress
from collections import Counter
from urllib.parse import urlsplit, unquote
from feature_extraction import extract_features

# Load dataset
df = pd.read_excel("data/data_bal - 20000.xlsx")

# Label meanings:
# 0 = Legitimate
# 1 = Phishing

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------

def extract_features(url):
    url = str(url)
    parsed = urlsplit(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    features = {}

    # 1. URL Length
    features["URLLength"] = len(url)

    # 2. Is Domain an IP Address?
    try:
        ipaddress.ip_address(hostname)
        features["IsDomainIP"] = 1
    except ValueError:
        features["IsDomainIP"] = 0

    # 3. Top-Level Domain (simplified extraction)
    domain_parts = hostname.split(".")
    features["TLD"] = domain_parts[-1] if len(domain_parts) >= 2 else ""

    # 4. Number of Subdomains (simplified)
    features["NoOfSubDomain"] = max(len(domain_parts) - 2, 0)

    # 5. Has Obfuscation
    # Detects percent-encoded characters in the URL
    features["HasObfuscation"] = int(
        bool(re.search(r"%[0-9a-fA-F]{2}", url))
    )

    # 6. Letter Ratio
    features["LetterRatioInURL"] = (
        sum(c.isalpha() for c in url) / len(url)
        if url else 0
    )

    # 7. Digit Ratio
    features["DegitRatioInURL"] = (
        sum(c.isdigit() for c in url) / len(url)
        if url else 0
    )

    # 8-10. Query-related characters
    features["NoOfEqualsInURL"] = url.count("=")
    features["NoOfQMarkInURL"] = url.count("?")
    features["NoOfAmpersandInURL"] = url.count("&")

    # 11. Other Special Characters
    features["NoOfOtherSpecialCharsInURL"] = sum(
        not c.isalnum() and c not in "=?&"
        for c in url
    )

    # 12. HTTPS
    features["IsHTTPS"] = int(parsed.scheme.lower() == "https")

    # -----------------------------
    # 6 ADDITIONAL FEATURES
    # -----------------------------

    # 13. Path Length
    features["PathLength"] = len(path)

    # 14. URL Entropy (Shannon entropy)
    if url:
        counts = Counter(url)
        features["URLEntropy"] = -sum(
            (count / len(url)) * math.log2(count / len(url))
            for count in counts.values()
        )
    else:
        features["URLEntropy"] = 0

    # 15. Number of Dots
    features["NoOfDots"] = url.count(".")

    # 16. Number of Hyphens in Domain
    features["NoOfHyphensInDomain"] = hostname.count("-")

    # 17. Suspicious Keywords
    keywords = [
        "login", "verify", "account", "secure",
        "update", "password", "bank", "signin"
    ]
    features["HasSuspiciousKeyword"] = int(
        any(word in url.lower() for word in keywords)
    )

    # 18. Suspicious File Extension
    suspicious_extensions = (
        ".exe", ".scr", ".bat", ".cmd", ".html", ".php"
    )
    features["HasSuspiciousExtension"] = int(
        unquote(path).lower().endswith(suspicious_extensions)
    )

    return pd.Series(features)


# -----------------------------
# APPLY FEATURE EXTRACTION
# -----------------------------

df = df.dropna(subset=["URLs", "Labels"]).copy()
df["URLs"] = df["URLs"].astype(str)

feature_df = df["URLs"].apply(extract_features)

df = pd.concat([df, feature_df], axis=1)

print("Total URLs:", len(df))
print("\nClass Distribution:")
print(df["Labels"].value_counts())

# -----------------------------
# COMPARE NUMERICAL FEATURES
# -----------------------------

numerical_features = [
    col for col in feature_df.columns if col != "TLD"
]

for feature in numerical_features:
    print(f"\n{'=' * 55}")
    print(f"FEATURE: {feature}")
    print("=" * 55)

    print(
        df.groupby("Labels")[feature]
        .describe()
        .round(3)
        .to_string()
    )

# -----------------------------
# COMPARE TLD DISTRIBUTIONS
# -----------------------------

print("\n" + "=" * 55)
print("FEATURE: TLD")
print("=" * 55)

for label, name in [(0, "Legitimate"), (1, "Phishing")]:
    print(f"\n{name} - Top 15 TLDs:")
    print(
        df[df["Labels"] == label]["TLD"]
        .value_counts()
        .head(15)
        .to_string()
    )

print("\nDuplicate URLs:", df.duplicated(subset=["URLs"]).sum())

conflicts = df.groupby("URLs")["Labels"].nunique()

print("URLs with conflicting labels:", (conflicts > 1).sum())

from urllib.parse import urlsplit

df["HasQuery"] = df["URLs"].apply(
    lambda url: bool(urlsplit(url).query)
)

print("\nURLs containing query parameters:")
for label in [0, 1]:
    subset = df[df["Labels"] == label]
    count = subset["HasQuery"].sum()
    percentage = subset["HasQuery"].mean() * 100

    print(f"Label {label}: {count} URLs ({percentage:.2f}%)")

print("\nURL Length Distribution:")

for label in [0, 1]:
    subset = df[df["Labels"] == label]

    print(f"\nLabel {label}:")

    for minimum, maximum in [
        (0, 50),
        (51, 100),
        (101, 200),
        (201, 500),
        (501, float("inf"))
    ]:
        count = subset[
            (subset["URLLength"] >= minimum) &
            (subset["URLLength"] <= maximum)
        ].shape[0]

        print(f"{minimum}-{maximum}: {count} URLs ({count / len(subset) * 100:.2f}%)")    