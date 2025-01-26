# src/preprocessing.py
import re
import logging

logger = logging.getLogger(__name__)

def clean_resume_text(raw_text: str) -> str:
    """
    Cleans and normalizes resume text data.
    """
    # 1. Normalize whitespace
    text = raw_text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    text = text.strip()

    # 2. Standardize key-value delimiters
    text = re.sub(r'\s*-\s*', ': ', text)
    text = re.sub(r'\s*:\s*', ': ', text)

    # 3. Remove special characters but keep emails, plus, minus, etc.
    text = re.sub(r'[^\w\s@.+-]', '', text)

    # 4. Normalize phone numbers (example pattern)
    text = re.sub(r'(\+?\d{1,3})[\s-]?(\d{4})[\s-]?(\d{5})', r'\1-\2-\3', text)

    # 5. Standardize dates (very naive example)
    text = re.sub(r'(\b[A-Z]{3,9})[-\s]?(\d{1,2})[-\s]?(\d{4})', r'\1 \2, \3', text)

    # 6. Remove excessive blank lines or spaces
    text = re.sub(r'\s{2,}', ' ', text)

    # 7. Title-case for large uppercase headings
    text = re.sub(r'(\b[A-Z ]{5,}\b)', lambda match: match.group(0).title(), text)

    return text.lower()
