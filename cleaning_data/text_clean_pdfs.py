import re

def clean_methodology(text):
    if not isinstance(text, str):
        return ""

    # Remove LaTeX formulas or math environments
    text = re.sub(r"\$.*?\$", "", text)  # inline math
    text = re.sub(r"\\\[(.*?)\\\]", "", text)  # \[ math \]
    
    # Remove proof environments and code-like sections
    text = re.sub(r"proof.*?qed", "", text, flags=re.DOTALL|re.IGNORECASE)
    
    # Remove figure references and mentions like “Figure 3”
    #text = re.sub(r"Figure\s?\d+", "", text, flags=re.IGNORECASE)
    
    # Remove Unicode characters commonly used in math proofs
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Replace multiple spaces/newlines with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove residual theorem problem statements (like “Putnam 2001 A1”)
    #text = re.sub(r"Putnam\s?\d{4}\s?[AB]?\d?", "", text)

    # Optionally truncate to 1000 characters to stay under embedding model limits
    return text#.strip()[:1000]

# Apply to your column
df_merged['methodology_clean'] = df_merged['methodology'].apply(clean_methodology)
df_merged["future_work_clean"] = df_merged["future_work"].apply(clean_methodology)
df_merged["text"] = df_merged["text"].apply(clean_methodology)


def remove_trailing_sections(text):
  """Removes text after common trailing sections like References, Bibliography, Acknowledgments."""
  if not isinstance(text, str):
      return ""

  # Pattern to find trailing sections (case-insensitive, with optional plurals)
  # Using word boundaries \b to avoid partial matches
  pattern = re.compile(r'\b(?:References|Bibliography|Acknowledgements?)\b', re.IGNORECASE)

  match = pattern.search(text)
  if match:
      return text[:match.start()].strip() # Return text up to the start of the matched section
  else:
      return text # Return original text if no matching section is found

df_merged['future_work_clean'] = df_merged['future_work_clean'].apply(remove_trailing_sections)
df_merged['methodology_clean'] = df_merged['methodology_clean'].apply(remove_trailing_sections)


df_merged['text_clean'] = df_merged['text'].apply(clean_methodology)
df_merged['text_clean'] = df_merged['text_clean'].apply(remove_trailing_sections)
