import requests
import os

# Create a directory to save PDFs
pdf_dir = "downloaded_pdfs"
os.makedirs(pdf_dir, exist_ok=True)

# Example list of PDF URLs from OpenReview
pdf_links = [
    "https://openreview.net/attachment?id=4XzGkm1jK0&name=pdf",
    "https://openreview.net/attachment?id=AkJvzpYMvK&name=pdf"
    # Add real links here
]

# Download each PDF
for url in pdf_links:
    paper_id = url.split('id=')[-1].replace("/", "-")
    file_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Downloaded: {file_path}")
    else:
        print(f"❌ Failed to download: {url}")
