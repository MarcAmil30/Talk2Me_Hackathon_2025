from flask import Flask, render_template, request
import pandas as pd, re, faiss, torch
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# === Data & Model Setup ===

# Load metadata
meta = pd.read_parquet("meta.parquet")

# Load FAISS index
idx = faiss.read_index("name.index")

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM
llm = Llama(
    model_path="/Users/rostislavfedorov/.cache/huggingface/hub/models--cleatherbury--Phi-3-mini-128k-instruct-Q4_K_M-GGUF/snapshots/a8bfa8927a872053499341c196d8730069aab659/phi-3-mini-128k-instruct.Q4_K_M.gguf"
)

# === Ask function ===
def ask(query, k=3):
    q_vec = embedder.encode([query])
    faiss.normalize_L2(q_vec)
    scores, ids = idx.search(q_vec, k)
    context = "\n\n".join(
        f"[{meta.iloc[i]['name']}] — {meta.iloc[i]['clean_abstract']}"
        for i in ids[0]
    )

    prompt = f"""
### System:
You are an academic connector. Given context bios, suggest the best person to contact.

### Context
{context}

### Question
“{query}”

### Answer (one name and one-line why). Add a biography of this person, given the context.
"""
    out = llm(prompt, temperature=0.2, max_tokens=200, stop=["###"])
    text = out.get("choices", [{}])[0].get("text", "").strip()

    # If empty, retry with a simple variation
    if text is None or text == "":
        retry_prompt = prompt 
        out = llm(retry_prompt, temperature=0.5, max_tokens=200, stop=["###"])
        text = out.get("choices", [{}])[0].get("text", "").strip()
        if not text:
            return "No answer found."

    # Remove special symbols (keeps letters, numbers, whitespace and common punctuation)
    cleaned_text = re.sub(r'[^\w\s\.\,\;\:\!\?\'\"-]', '', text)
    return cleaned_text

# === Flask UI ===

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        about = request.form.get("about", "")
        goals = request.form.get("goals", "")
        event = request.form.get("event", "")
        combined_query = f"About me: {about}. I want to find person who can help me with: {goals}."
        answer = ask(combined_query)
    return render_template("index_v4.html", answer=answer)


if __name__ == "__main__":
    app.run(debug=True, port=5002)