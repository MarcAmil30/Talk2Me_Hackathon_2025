import pandas as pd
import requests
import os

# Load the GitHub CSV (raw JSON-like structure inside a CSV)
url = "https://raw.githubusercontent.com/MarcAmil30/Talk2Me_Hackathon_2025/main/icml24_AI4MATH.csv"
df = pd.read_csv(url)

# If the CSV was loaded as strings (e.g. JSON-like dictionaries), convert using eval safely
import ast
rows = [ast.literal_eval(r) for r in df.columns]  # your case: CSV column contains JSON objects

# Extract full PDF URLs
pdf_links = []
for row in rows:
    try:
        pdf_path = row["content"]["pdf"]["value"]  # like "/pdf/abc123.pdf"
        full_url = f"https://openreview.net{pdf_path}"
        pdf_links.append(full_url)
    except KeyError:
        continue

print(f"✅ Found {len(pdf_links)} PDF links.")
print("Example:", pdf_links[:3])
