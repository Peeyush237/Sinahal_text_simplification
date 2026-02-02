## Sinhala Text Simplifier

Flask web app for **automatic Sinhala text simplification** using a fine‑tuned **mT5** model. It exposes a simple Sinhala UI and REST API so you can turn complex Sinhala sentences into shorter, easier versions.

---

## ✨ What This Project Does

- **Simplifies Sinhala text** using a transformer model
- **Web interface** at `http://localhost:5000` for interactive use
- **API endpoints** for integration into other tools
- **Evaluation and analysis** (SARI, BLEU, etc.) supported by helper scripts

For detailed research, architecture, and metric explanations, see `PROJECT_EXPLANATION.md`.

---

## 🎥 Project Demo

Most Markdown viewers (including many IDEs) do **not** play local `.mp4` files inline, so the embedded player may appear as an empty bar.  
Use the link below to open the video in your default media player or browser:

- **Demo video**: [▶ Watch “project demo final.mp4”](project%20demo%20final.mp4)

If that still doesn’t open in your viewer, open the file directly from your file explorer: `project demo final.mp4`.

---

## 🚀 Quick Start

- **Requirements**
  - Python 3.9+
  - `sinhala_text_simplification_results.pkl`
  - `final_sinhala_simplifier/` model folder

- **Install dependencies**

```bash
pip install -r requirement.txt
```

- **Run the app**

```bash
python run.py
```

Then open `http://localhost:5000` in your browser.

---

## 🌐 Main Endpoints

- **`GET /`** – Web UI  
- **`POST /api/simplify`** – Simplify Sinhala text (JSON body `{ "text": "..." }`)  
- **`GET /api/stats`** – View key metrics and improvements  
- **`POST /api/translate`** – Sinhala → English translation  
- **`GET /api/health`** – Basic health check  

---

## 📁 Files to Know

- `app.py` – Flask app and API routes  
- `run.py` – Safe entry point that checks required files and starts the server  
- `sari_optimizer.py`, `calculate_sari_score.py` – Evaluation and optimization logic  
- `final_sinhala_simplifier/` – Fine‑tuned model and tokenizer  
- `PROJECT_EXPLANATION.md` – Full technical write‑up and results  
