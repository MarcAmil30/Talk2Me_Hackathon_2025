pip install sentence-transformers faiss-cpu PyPDF2 requests
!pip install selenium 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
!apt-get update
!apt install chromium-chromedriver
!cp /usr/lib/chromium-browser/chromedriver /usr/bin
!pip install selenium
!apt-get update
!apt install -y chromium-chromedriver
!cp /usr/lib/chromium-browser/chromedriver /usr/bin
!pip install selenium

import pandas as pd

# Read the CSV file directly from GitHub
csv_url = 'https://raw.githubusercontent.com/MarcAmil30/Talk2Me_Hackathon_2025/main/icml24_AI4MATH.csv'
df = pd.read_csv(csv_url)

# Display the first few rows to inspect the data
print(df.head())
df = pd.read_csv(csv_url)

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


import pandas as pd
import ast
import json
import re

# Load CSV and parse JSON-like column
csv_url = 'https://raw.githubusercontent.com/MarcAmil30/Talk2Me_Hackathon_2025/main/icml24_AI4MATH.csv'
df = pd.read_csv(csv_url)
rows = [ast.literal_eval(r) for r in df.columns]

# Build RAG documents with paper_id
rag_docs = []

for row in rows:
    try:
        content = row.get("content", {})
        title = content.get("title", {}).get("value", "")
        authors = content.get("authors", {}).get("value", [])
        abstract = content.get("abstract", {}).get("value", "")
        tldr = content.get("TLDR", {}).get("value", "")
        pdf_path = content.get("pdf", {}).get("value", "")
        url = f"https://openreview.net{pdf_path}" if pdf_path else ""

        # Extract paper_id (hash) from PDF path
        match = re.search(r'/pdf/([a-f0-9]{40})\.pdf', pdf_path)
        paper_id = match.group(1) if match else ""

        full_text = f"{title}\n\n{tldr}\n\n{abstract}"

        rag_docs.append({
            "text": full_text,
            "metadata": {
                "paper_id": paper_id,
                "title": title,
                "authors": authors,
                "url": url,
                "source": "ICML 2024 AI4MATH"
            }
        })
    except Exception as e:
        print(f"⚠️ Skipping one row due to error: {e}")

# Save to JSON
with open("rag_documents.json", "w") as f:
    json.dump(rag_docs, f, indent=2)

print(f"✅ Created {len(rag_docs)} RAG-ready documents.")

for url in pdf_links:
    filename = url.split("/")[-1]  # Extract the filename from the URL
    file_path = os.path.join(pdf_dir, filename)

    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Downloaded: {file_path}")
        else:
            print(f"❌ Failed to download ({response.status_code}): {url}")
    except Exception as e:
        print(f"⚠️ Error downloading {url}: {e}")

  from PyPDF2 import PdfReader

for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        path = os.path.join(pdf_dir, filename)
        try:
            reader = PdfReader(path)
            print(f"✅ Valid PDF: {filename}, Pages: {len(reader.pages)}")
        except:
            print(f"❌ Corrupt or unreadable: {filename}")



from PyPDF2 import PdfReader
import os
import pandas as pd

# Folder containing your validated PDFs
pdf_dir = "downloaded_pdfs"

# Keywords to search for
method_keywords = ["methodology", "approach", "methods", "Approaches", "methods"]
future_keywords = ["future work", "conclusion", "limitations", "discussion"]

def extract_section(text, keywords):
    text_lower = text.lower()
    for keyword in keywords:
        idx = text_lower.find(keyword)
        if idx != -1:
            return text[idx:idx + 2000]  # Return a snippet
    return ""

# Extract text and sections
data = []
for file in os.listdir(pdf_dir):
    if file.endswith(".pdf"):
        path = os.path.join(pdf_dir, file)
        try:
            reader = PdfReader(path)
            full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
            methodology = extract_section(full_text, method_keywords)
            future_work = extract_section(full_text, future_keywords)
            data.append({
                "filename": file,
                "methodology": methodology.strip(),
                "future_work": future_work.strip()
            })
        except Exception as e:
            data.append({"filename": file, "methodology": "", "future_work": f"Error: {e}"})

# Create DataFrame
df_sections = pd.DataFrame(data)
df_sections.head()
df_sections["paper_id"] = df_sections["filename"].str.extract(r'([a-f0-9]{40})')

df_sections.head()
df_rag = pd.DataFrame([{
    "paper_id": doc["metadata"]["paper_id"],
    "title": doc["metadata"]["title"],
    "authors": doc["metadata"]["authors"],
    "url": doc["metadata"]["url"],
    "text": doc["text"]
} for doc in rag_docs])

df_merged = pd.merge(df_rag, df_sections, on="paper_id", how="left")

