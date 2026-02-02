## Sinhala Text Simplifier 🪄

Modern Flask web app for **automatic Sinhala text simplification** using a fine‑tuned **mT5 (multilingual T5)** model, optimized for **SARI** and output length. It provides a clean Sinhala UI and REST API to turn complex Sinhala sentences into simpler, more readable text.

---

## ✨ Key Features

- **Sinhala text simplification** powered by a fine‑tuned mT5 model  
- **Flask web interface** with Sinhala‑friendly UI  
- **REST API** for programmatic access (`/api/simplify`, `/api/stats`, `/api/translate`)  
- **SARI‑optimized decoding pipeline**:
  - Multi‑candidate generation with adaptive configs
  - SARI‑based candidate selection
  - Aggressive artifact removal and length control  
- **Evaluation support** via SARI, BLEU and detailed metrics

For a deep dive into the research and algorithms, see `PROJECT_EXPLANATION.md`.

---

## 🧱 Tech Stack

- **Backend**: `Flask`, `Flask-CORS`
- **Model**: `mT5` via `transformers` + `torch`
- **Tokenization**: `sentencepiece`
- **Data & Evaluation**: `pandas`, `numpy`

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python 3.9+**
- Sufficient disk space and RAM for the mT5 model
- (Optional) GPU with CUDA for faster inference

### 2. Install Dependencies

From the project root:

```bash
python -m venv venv
venv\Scripts\activate  # On Windows PowerShell
# source venv/bin/activate  # On Linux/macOS

pip install -r requirement.txt
```

### 3. Required Model Files

Make sure the following are present in the project root (same folder as `app.py` and `run.py`):

- `sinhala_text_simplification_results.pkl`
- `final_sinhala_simplifier/` directory containing at least:
  - `model.safetensors`
  - `tokenizer_config.json`
  - `spiece.model`
  - `config.json`

If these are missing, `run.py` will show a clear message and exit.

### 4. Run the Application

```bash
python run.py
```

Then open your browser at `http://localhost:5000` to use the Sinhala web UI.

---

## 🌐 API Overview

- **GET `/`**
  - Renders the main Sinhala web interface.

- **POST `/api/simplify`**
  - **Body (JSON)**:
    ```json
    { "text": "ඔබ සරල කිරීමට අවශ්‍ය සංකීර්ණ සිංහල වාක්‍යය" }
    ```
  - **Response**:
    ```json
    {
      "input": "...",
      "simplified": "...",
      "success": true,
      "timestamp": "..."
    }
    ```

- **GET `/api/stats`**
  - Returns key metrics such as SARI score, BLEU score, average lengths and improvement percentages.

- **POST `/api/translate`**
  - Translates Sinhala text to English using `googletrans`.

- **GET `/api/health`**
  - Simple health check: model/data load status and timestamp.

---

## 🧠 How Simplification Works (High‑Level)

1. **Pre‑processing**
   - Clean input: remove HTML, extra whitespace and noise.
   - Format prompt as: `simplify sinhala: {input_text}`.

2. **Multi‑Candidate Generation**
   - Generate several candidate simplifications with different decoding configs:
     - Adaptive `max_new_tokens`, `min_length`, and `length_penalty` based on input length.
     - Beam search with configurable `num_beams`, `temperature`, and `repetition_penalty`.

3. **SARI‑Aware Selection & Optimization**
   - Use `sari_optimizer.py` to:
     - Compute SARI (when references available) or heuristic SARI‑inspired scores.
     - Penalize artifacts (`<extra_id_0>`, `<pad>`, “sinhala”, “english”, etc.).
     - Prefer outputs with good **Addition**, **Deletion**, and **Keep** balance.

4. **Post‑Processing**
   - Aggressively clean artifacts and normalize whitespace and punctuation.
   - If output is too short, extend it using:
     - Important words from the original sentence.
     - Common Sinhala words for context.

For formulas, strategies, and measured improvements, see `PROJECT_EXPLANATION.md` and `BERTSCORE_GUIDE.md`.

---

## 📁 Project Structure (Simplified)

```text
app.py                      # Flask app, routes, model loading & inference
run.py                      # Safe development runner with file checks
sari_optimizer.py           # SARI-based generation & post-processing logic
calculate_sari_score.py     # SARI / BLEU evaluation utilities
final_sinhala_simplifier/   # Fine-tuned mT5 model & tokenizer files
sinhala_text_simplification_results.pkl  # Evaluation data / references
templates/
  index.html                # Sinhala web UI (front-end)
static/                     # Static assets (created at runtime if missing)
PROJECT_EXPLANATION.md      # Detailed technical write-up
BERTSCORE_GUIDE.md          # Notes on BERTScore usage (if populated)
requirement.txt             # Python dependencies
```

---

## 🔍 Troubleshooting

- **Server starts but model is not available**
  - Check that `final_sinhala_simplifier/` exists and contains all model and tokenizer files.
  - Check that `sinhala_text_simplification_results.pkl` is in the project root.

- **Out-of-memory or very slow inference**
  - Run on a machine with more RAM, or with GPU + CUDA.
  - Reduce concurrent requests; avoid running heavy evaluations at the same time.

- **Encoding issues with Sinhala text**
  - The app is configured for UTF‑8 (`JSON_AS_ASCII = False`), but ensure your terminal and browser are set to UTF‑8 as well.

---

## 📜 License & Credits

- This project builds on **mT5** and the **Hugging Face Transformers** ecosystem.  
- Add your preferred **license** here (e.g., MIT, Apache‑2.0) and any dataset or research paper acknowledgements you want to credit.

