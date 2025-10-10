Fake News Detection System

Overview

The Fake News Detection System is an AI-powered web application designed to assess the credibility of news content. It provides a modular, explainable pipeline that accepts text, URLs, images, and videos, extracts textual claims, runs AI-based credibility checks, cross-references claims with reputable sources, and generates community-style notes explaining findings.

Architecture & Modules

- Input Module
  - Accepts raw text, URLs, image files, and video/audio files.
  - Handles user uploads and URL submission.

- Text Extraction Module
  - Web scraping (requests, BeautifulSoup) to pull page content and handle different site structures.
  - OCR for images using Pillow + pytesseract.
  - Audio transcription from videos using openai-whisper or Whisper's Python bindings.
  - Basic cleaning and normalization (remove boilerplate, dedupe, handle encodings).

- AI Prediction Module
  - Text preprocessing: tokenization, lowercasing, stopword removal, punctuation handling.
  - Feature extraction: TF-IDF for classical models, or transformer-based embeddings for deep models.
  - Classification: Naive Bayes, Logistic Regression, Random Forest, Gradient Boosting, or transformer fine-tuning.
  - Output: binary (Real / Fake) plus confidence score and feature importance / attention highlights.

- Fact Verification Module
  - Claim detection using NER and simple heuristics.
  - Search reputable sources (e.g., public APIs, curated list of known fact-checkers) to find supporting/contradictory evidence.
  - Use NLI-style models to judge whether found evidence supports or refutes claims.

- Community Notes Generator
  - Summarizes findings into short notes explaining verdict, confidence, contradictory evidence, and links to sources.
  - Emphasizes transparency (what was checked and why).

- Output Module
  - Returns a human-friendly report: verdict, confidence, supporting text snippets, source links, and community notes.

Key Features & Techniques

- Multi-input handling: URLs, images, audio, and text.
- OCR and speech-to-text for non-text inputs.
- Hybrid ML stack: classical features for speed and transformers for higher accuracy.
- Fact-checking by cross-referencing external sources and using NLI for claim-level verification.
- Explainability via highlights, feature importances, and generated notes.

Tech Stack (suggested)

- Python 3.10+
- Web/app: Streamlit for the UI; Django if a backend is needed for user management or APIs.
- ML/NLP: scikit-learn, transformers (Hugging Face), sentence-transformers, torch
- OCR: Pillow, pytesseract
- Speech transcription: openai-whisper or whisper
- Scraping: requests, BeautifulSoup
- DB: SQLite for prototyping; Postgres for production

Files & Where to Look

- The Django project is in `fakenews_project/`.
- The key app is `detector/` which contains the prediction code and views.

How it works (high level)

1. User submits a URL/image/video/text.
2. The extraction module pulls or transcribes text.
3. The prediction module returns a Real/Fake verdict and confidence.
4. The verification module searches external sources for claims found and returns supporting/contradictory evidence.
5. The community notes generator summarizes findings and links to sources.
6. The UI displays the report and explanation to the user.

Running locally (example)

Windows / PowerShell

1. Create and activate virtualenv

```powershell
python -m venv venv
; .\venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Run Django development server (if using Django backend)

```powershell
cd fakenews_project
; .\venv\Scripts\Activate.ps1
; python manage.py migrate
; python manage.py runserver
```

4. Run Streamlit UI (if separate)

```powershell
; .\venv\Scripts\Activate.ps1
streamlit run path\to\app.py
```

Next steps & Improvements

- Add a `requirements.txt` to pin dependencies.
- Add unit tests for extraction, prediction, and verification modules.
- Add a Dockerfile and Docker Compose for easier local deployment.
- Move heavy ML models to a separate model server or use model sharding.
- Secure any API keys and add instructions to set them as environment variables.

Environment variables and .env

- For local development it's convenient to store secrets such as `OPENAI_API_KEY` in a `.env` file at the project root. Keep `.env` out of version control by adding it to `.gitignore`.
- Example `.env` content:

```
OPENAI_API_KEY="sk-REPLACE_WITH_YOUR_KEY"
```

- To load `.env` automatically in this project, install `python-dotenv` and the Django settings file will load `.env` at startup if present. Installation:

```powershell
pip install python-dotenv
```

- Alternatively, set the environment variable in PowerShell for the session:

```powershell
$env:OPENAI_API_KEY = "sk-REPLACE_WITH_YOUR_KEY"
python manage.py runserver
```

License & Credits

Add licensing information and acknowledgment for any third-party models or datasets used.
