# AI Resume Analyzer & Job Match Assistant

A modern, high-performance web application designed to analyze resumes, evaluate candidate-job fit, inspect keyword context with the **Knuth-Morris-Pratt (KMP)** algorithm, perform mathematical **Set Theory** skill gap analysis, and provide AI-driven feedback using **Google Gemini LLM**.

---

## 🌟 Key Features

1. **Deterministic KMP Pattern Matcher**:
   - Fast $O(N + M)$ exact phrase and keyword location in resume text.
   - Extracts matched keyword occurrences with contextual text windows.

2. **Mathematical Set Operations Engine**:
   - Computes:
     - **Matched Skills** ($R \cap J$)
     - **Missing Job Requirements** ($J \setminus R$)
     - **Candidate Value-Add / Bonus Skills** ($R \setminus J$)
     - **Jaccard Similarity Index** ($\frac{|R \cap J|}{|R \cup J|}$)
     - **Dice Coefficient & Overlap Recall**
     - **Composite ATS Fit Score**

3. **NLP Text Parser (NLTK)**:
   - Automated text extraction from **PDF**, **DOCX**, and **TXT** files.
   - Contact detail extraction (Emails, Phone Numbers, LinkedIn, GitHub).
   - Word count, sentence metrics, and non-stopword token frequencies.

4. **AI Career Coach & Generator (Google Gemini)**:
   - **ATS Compatibility Critique & Red Flags Analyzer**
   - **Google X-Y-Z Resume Bullet Point Rewriter**
   - **Tailored Role-Specific Cover Letter Generator**
   - **Custom Technical & Behavioral Mock Interview Preparation**

5. **Batch Multi-Job Ranking (Pandas)**:
   - Rank a resume simultaneously against an entire catalog/CSV of job postings.
   - Interactive Plotly visualizations and CSV export.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+
- Recommended: Google Gemini API key (Free at [Google AI Studio](https://aistudio.google.com/app/apikey))

### 2. Installation
```bash
# Clone or navigate to the project directory
cd codex

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key (Optional)
Create a `.env` file or enter your key directly in the Streamlit UI sidebar:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
codex/
├── algorithms/
│   ├── kmp.py             # Knuth-Morris-Pratt substring search algorithm
│   └── set_matcher.py     # Set operations, Jaccard similarity, and ATS scoring
├── utils/
│   ├── parser.py          # PDF/DOCX/TXT extractor, regex contact parser, NLTK analyzer
│   ├── skills_db.py       # Curated 500+ skill taxonomy database
│   └── llm_client.py      # Google Gemini 2.5 Flash API client
├── data/
│   └── sample_jobs.csv    # Sample job catalog for batch matching
├── app.py                 # Streamlit UI dashboard with tabs & interactive charts
├── test_core.py           # Unit tests for core algorithms
├── requirements.txt       # Python package dependencies
├── .env.example           # Example configuration
└── README.md              # Project documentation
```
