import pandas as pd
import math
import re
import ipaddress
from collections import Counter
from urllib.parse import urlsplit, urlunsplit, unquote



def extract_features(url):
    parsed = urlsplit(url)

    if not parsed.path:
        url = urlunsplit((
            parsed.scheme,
            parsed.netloc,
            "/",
            parsed.query,
            parsed.fragment
        ))
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