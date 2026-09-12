# 🏥 SehatAI — Rural Health Risk & Triage Copilot

Diabetes + Cardiovascular risk screening copilot, built with 100% free tools:
Google Colab (training), Streamlit (app + deployment), GitHub (code), Supabase (database).

---

## 📁 Project Structure

```
sehatai/
├── notebooks/
│   └── train_models.py        # Run in Google Colab to train + save models
├── app/
│   ├── streamlit_app.py       # Main Streamlit app
│   └── utils/
│       ├── ocr_utils.py       # Lab report OCR extraction
│       ├── nlp_utils.py       # Symptom text parser
│       ├── fusion_utils.py    # Combines symptoms + OCR + form data
│       └── db_utils.py        # Supabase read/write
├── models/                    # Trained .pkl models go here (empty until you train)
├── data/                      # Downloaded datasets go here
├── supabase_schema.sql        # Run this in Supabase SQL editor
├── requirements.txt           # Python deps for Streamlit Cloud
├── packages.txt               # System deps (tesseract) for Streamlit Cloud
└── README.md                  # This file
```

---

## ✅ Step-by-Step: What To Do

### Step 1 — Get a Kaggle account + API token
1. Sign up free at kaggle.com
2. Go to Account → Create New API Token → downloads `kaggle.json`
3. Keep this file, you'll upload it in Colab.

### Step 2 — Train the models (Google Colab)
1. Open [Google Colab](https://colab.research.google.com), upload `notebooks/train_models.py`
   (or copy-paste its content into a new notebook).
2. Run all cells. It will:
   - Ask you to upload `kaggle.json`
   - Download both datasets automatically
   - Clean the data
   - Train diabetes + cardio models
   - Save `.pkl` files
3. Download the generated files from `models/` folder in Colab (left sidebar → Files)
   and place them into your local `sehatai/models/` folder.

### Step 3 — Create a free Supabase project
1. Sign up at [supabase.com](https://supabase.com) → New Project (free tier).
2. Go to SQL Editor → paste contents of `supabase_schema.sql` → Run.
3. Go to Project Settings → API → copy your **Project URL** and **anon public key**.

### Step 4 — Push everything to GitHub
```bash
cd sehatai
git init
git add .
git commit -m "Initial SehatAI project"
git branch -M main
git remote add origin https://github.com/<your-username>/sehatai.git
git push -u origin main
```
> Note: `.pkl` model files should be pushed too (they're small, a few hundred KB–MB).

### Step 5 — Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub.
2. Click "New app" → select your repo → set main file path to `app/streamlit_app.py`.
3. Before deploying, click **Advanced settings → Secrets** and paste:
   ```toml
   SUPABASE_URL = "your-project-url"
   SUPABASE_KEY = "your-anon-key"
   ```
4. Click Deploy. Wait 2–3 minutes. You'll get a public URL like:
   `https://sehatai.streamlit.app`

### Step 6 — Test it
- Fill the patient form, type symptoms, optionally upload a lab report image.
- Click "Analyze Risk" → see diabetes/cardio risk % + risk level.
- Check the "Doctor Dashboard" tab → should show saved patient history.

---

## 🆓 Free Resources Used

| Purpose | Tool | Link |
|---|---|---|
| Model training compute | Google Colab | https://colab.research.google.com |
| Diabetes dataset | Kaggle (uciml) | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |
| Cardio dataset | Kaggle (cherngs) | https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci |
| Database | Supabase | https://supabase.com |
| Deployment | Streamlit Cloud | https://share.streamlit.io |
| Version control | GitHub | https://github.com |

---

## 🛠️ Troubleshooting

- **"Model file not found" in Streamlit** → make sure `models/*.pkl` files are actually
  pushed to GitHub and paths match (`models/diabetes_model.pkl` relative to repo root).
- **OCR not working on Streamlit Cloud** → make sure `packages.txt` (with `tesseract-ocr`)
  is in the repo root — Streamlit Cloud reads it automatically to install system packages.
- **Supabase insert fails** → double check `SUPABASE_URL` / `SUPABASE_KEY` in Secrets,
  and that you ran `supabase_schema.sql` to create the table first.
- **Cardio dataset target column name mismatch** → open the CSV and check if the label
  column is `condition` or `target`; update `train_models.py` accordingly if needed.
