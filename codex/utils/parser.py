"""
Resume & Job Description Text Parser, Cleaner, and Entity Extractor.
Uses NLTK, regex, pdfplumber, and python-docx.
"""
import re
import io
from typing import Dict, Any, Set, List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from utils.skills_db import SKILLS_TAXONOMY, get_all_skills_flat
from algorithms.kmp import kmp_search

# Ensure required NLTK resources are downloaded safely on demand
_NLTK_INITIALIZED = False

def ensure_nltk_resources():
    global _NLTK_INITIALIZED
    if _NLTK_INITIALIZED:
        return
    _NLTK_INITIALIZED = True
    packages = ['punkt', 'stopwords', 'wordnet']
    for p in packages:
        try:
            nltk.data.find(f'tokenizers/{p}' if 'punkt' in p else f'corpora/{p}')
        except (LookupError, Exception):
            pass



def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extracts raw text from PDF, DOCX, or plain text file uploads."""
    text = ""
    filename_lower = filename.lower()

    if filename_lower.endswith('.pdf'):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except Exception as e:
            text = f"Error reading PDF: {str(e)}"

    elif filename_lower.endswith('.docx'):
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            text = f"Error reading DOCX: {str(e)}"

    else:
        # Fallback to UTF-8 / plain text
        try:
            text = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            text = file_bytes.decode('latin-1', errors='ignore')

    return text.strip()


def clean_text_for_nlp(text: str) -> str:
    """Normalizes whitespace, removes unwanted special characters, preserves alphanumeric tokens."""
    if not text:
        return ""
    # Normalize unicode whitespace
    text = re.sub(r'[\r\n\t]+', ' ', text)
    # Replace multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_contact_info(text: str) -> Dict[str, Any]:
    """Extracts email addresses, phone numbers, LinkedIn URLs, and GitHub handles using regex."""
    emails = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    phones = re.findall(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    linkedin = re.findall(r'(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+', text, re.IGNORECASE)
    github = re.findall(r'(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9_-]+', text, re.IGNORECASE)

    return {
        "emails": list(set(emails)),
        "phones": list(set(phones)) if phones else [],
        "linkedin": list(set(linkedin)),
        "github": list(set(github))
    }


def extract_skills_with_kmp(text: str) -> Set[str]:
    """
    Extracts skills from text using exact KMP pattern searching against the skills taxonomy.
    Handles single-word and multi-word phrases (e.g. 'machine learning', 'ci/cd', 'react.js').
    """
    all_skills = get_all_skills_flat()
    found_skills = set()
    text_lower = f" {text.lower()} "

    for skill in all_skills:
        skill_lower = skill.lower()
        # For very short symbols like 'c' or 'r', ensure word boundaries
        if len(skill_lower) <= 2 and skill_lower in ["c", "r", "go"]:
            # Check with word boundary regex or space padding
            pattern = rf'\b{re.escape(skill_lower)}\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill_lower)
        else:
            # Use KMP to search for skill
            indices = kmp_search(text_lower, skill_lower, case_sensitive=False)
            if indices:
                # Validate boundary for letters to prevent substring false positives
                for idx in indices:
                    char_before = text_lower[idx - 1] if idx > 0 else " "
                    idx_after = idx + len(skill_lower)
                    char_after = text_lower[idx_after] if idx_after < len(text_lower) else " "

                    # If not surrounded by alphabetical characters, it's a valid match
                    if not (char_before.isalnum() or char_after.isalnum()):
                        found_skills.add(skill_lower)
                        break

    return found_skills


def analyze_text_statistics(text: str) -> Dict[str, Any]:
    """Extracts sentence count, word count, reading ease, and NLTK top token analysis."""
    if not text:
        return {"word_count": 0, "sentence_count": 0, "avg_sentence_len": 0, "top_tokens": []}

    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = text.split('.')

    try:
        tokens = word_tokenize(text.lower())
    except Exception:
        tokens = text.lower().split()

    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = {"the", "a", "an", "in", "and", "or", "to", "for", "of", "with", "at", "by", "on"}

    # Filter alphanumeric words not in stopwords
    filtered_words = [w for w in tokens if w.isalnum() and w not in stop_words and len(w) > 2]
    
    # Calculate frequency
    freq = {}
    for w in filtered_words:
        freq[w] = freq.get(w, 0) + 1

    top_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:15]

    word_count = len(tokens)
    sent_count = max(1, len(sentences))
    avg_sent_len = round(word_count / sent_count, 1)

    return {
        "word_count": word_count,
        "sentence_count": sent_count,
        "avg_sentence_len": avg_sent_len,
        "top_tokens": top_tokens
    }
