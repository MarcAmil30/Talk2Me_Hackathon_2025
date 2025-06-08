import pandas as pd, re


df = pd.read_csv("/Users/rostislavfedorov/Documents/Documents for  PhD/hack_sci/Talk2Me_Hackathon_2025/cleaned_matches.csv")

# Normalise
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
df['clean_abstract'] = (df['text_analyze']
                        .fillna('')
                        .str.replace(r'\s+', ' ', regex=True)
                        .str.strip())

import faiss, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# combine fields for embedding
df["doc"] = df["clean_abstract"]
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#embedder = SentenceTransformer("thenlper/gte-small")  # or the GGUF via llama.cpp embed API
vectors = embedder.encode(df["doc"].tolist(), batch_size=64, show_progress_bar=True)



index = faiss.IndexFlatIP(vectors.shape[1])           # cosine sim == inner product on L2-normalised vecs
faiss.normalize_L2(vectors)
index.add(vectors)

faiss.write_index(index, "name.index")
df[["name","ai_summ","clean_abstract"]].to_parquet("meta.parquet")

import json, torch, pathlib, faiss, pandas as pd
from llama_cpp import Llama  # pip install llama-cpp-python

idx  = faiss.read_index("name.index")
meta = pd.read_parquet("meta.parquet")

llm  = Llama(model_path="/Users/rostislavfedorov/.cache/huggingface/hub/models--cleatherbury--Phi-3-mini-128k-instruct-Q4_K_M-GGUF/snapshots/a8bfa8927a872053499341c196d8730069aab659/phi-3-mini-128k-instruct.Q4_K_M.gguf")



def ask(query, k=3):
    q_vec = embedder.encode([query])
    faiss.normalize_L2(q_vec)
    scores, ids = idx.search(q_vec, k)
    context = "\n\n".join(
        f"[{meta.iloc[i]['name']}] —  {meta.iloc[i]['clean_abstract']}"
        for i in ids[0]
    )
    prompt = f"""
### System:
You are an academic connector.  Given context bios, suggest the best person to contact.

### Context
{context}

### Question
“{query}”

### Answer (one name and one-line why). 
"""
    out = llm(prompt, temperature=0.2, max_tokens=200, stop=["###"])
    if "choices" not in out or len(out["choices"]) == 0:
        return "No answer found."
    

   

    return out["choices"][0]["text"].strip()

print(ask("I need to find a person. He is doing geometric deep learning on molecules."))