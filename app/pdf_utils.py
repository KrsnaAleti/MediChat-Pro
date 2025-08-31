from pypdf import PdfReader, PdfWriter
from typing import List, Optional
from io import BytesIO


def extract_text_from_pdf(file):
    """Extracts text from a PDF file.
    Args:
        file: A file path (str) or a file-like object (BytesIO) for the PDF.
    Returns:
        A string containing the extracted text from all pages.
    """
    reader = PdfReader(file)
    text = " "
    for page in reader.pages:
        text = text + page.extract_text() or ''
    return text
