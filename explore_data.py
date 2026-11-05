import pandas as pd

df = pd.read_csv("data/PhiUSIIL_Phishing_URL_Dataset.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nLabel counts:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nDuplicate URLs:")
print(df["URL"].duplicated().sum())

print("\nAverage URL Length:")
print(df.groupby("label")["URLLength"].mean())

print("\nAverage Domain Length:")
print(df.groupby("label")["DomainLength"].mean())

print("\nURL Length Statistics:")
print(df.groupby("label")["URLLength"].describe())

print("\nDomain Length Statistics:")
print(df.groupby("label")["DomainLength"].describe())

print("\nIsDomainIP:")
print(df.groupby("label")["IsDomainIP"].value_counts(normalize=True) * 100)

print("\nTop TLDs - Phishing:")
print(df[df["label"] == 0]["TLD"].value_counts().head(15))

print("\nTop TLDs - Legitimate:")
print(df[df["label"] == 1]["TLD"].value_counts().head(15))

print("\nNumber of Subdomains Statistics:")
print(df.groupby("label")["NoOfSubDomain"].describe())

print("\nSubdomain Counts - Phishing:")
print(df[df["label"] == 0]["NoOfSubDomain"].value_counts(normalize=True).sort_index() * 100)

print("\nSubdomain Counts - Legitimate:")
print(df[df["label"] == 1]["NoOfSubDomain"].value_counts(normalize=True).sort_index() * 100)

print("\nHas Obfuscation:")
print(df.groupby("label")["HasObfuscation"].value_counts(normalize=True) * 100)

print("\nNumber of Obfuscated Characters:")
print(df.groupby("label")["NoOfObfuscatedChar"].describe())

print("\nObfuscation Ratio:")
print(df.groupby("label")["ObfuscationRatio"].describe())

print("\nLetter Ratio Statistics:")
print(df.groupby("label")["LetterRatioInURL"].describe())

print("\nDigit Ratio Statistics:")
print(df.groupby("label")["DegitRatioInURL"].describe())

print("\nEquals Sign Statistics:")
print(df.groupby("label")["NoOfEqualsInURL"].describe())

print("\nQuestion Mark Statistics:")
print(df.groupby("label")["NoOfQMarkInURL"].describe())

print("\nAmpersand Statistics:")
print(df.groupby("label")["NoOfAmpersandInURL"].describe())

print("\nOther Special Characters Statistics:")
print(df.groupby("label")["NoOfOtherSpecialCharsInURL"].describe())

print("\nSpecial Character Ratio Statistics:")
print(df.groupby("label")["SpacialCharRatioInURL"].describe())

print("\nHTTPS Usage:")
print(df.groupby("label")["IsHTTPS"].value_counts(normalize=True) * 100)

duplicates = df[df.duplicated(subset=["URL"], keep=False)]

print("Total rows involving duplicate URLs:", len(duplicates))
print("Number of extra duplicate rows:", df.duplicated(subset=["URL"]).sum())

conflicting_urls = df.groupby("URL")["label"].nunique()

print("URLs with conflicting labels:", (conflicting_urls > 1).sum())

df = df.drop_duplicates(subset=["URL"]).copy()

print("Rows after cleaning:", len(df))

print(df.groupby("label")["URLLength"].agg(["min", "max", "mean"]))

print("\nSample legitimate URLs:")
print(df[df["label"] == 1]["URL"].sample(10).to_string(index=False))

from urllib.parse import urlsplit

legitimate_urls = df[df["label"] == 1]["URL"]

has_path = legitimate_urls.apply(
    lambda url: urlsplit(url).path not in ("", "/")
)

print("Legitimate URLs with paths:", has_path.sum())
print("Percentage:", round(has_path.mean() * 100, 2), "%")