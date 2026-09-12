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
