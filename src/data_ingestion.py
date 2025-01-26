# src/data_ingestion.py
import os
import PyPDF2
import fitz  # PyMuPDF
import logging

logger = logging.getLogger(__name__)

def split_pdf(input_pdf_path: str, output_folder: str) -> None:
    """
    Splits a single compiled PDF into individual PDFs based on unique markers.
    """
    os.makedirs(output_folder, exist_ok=True)

    with open(input_pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        writer = None
        current_student_name = None

        for i, page in enumerate(reader.pages):
            text = page.extract_text()

            # Identify the start of a new resume based on unique markers
            if "NATIONAL INSTITUTE OF TECHNOLOGY KARNATAKA, SURATHKAL" in text and "Branch" in text:
                # Save the previous student's PDF if it exists
                if writer and current_student_name:
                    output_file = os.path.join(output_folder, f"{current_student_name}.pdf")
                    with open(output_file, 'wb') as output_pdf:
                        writer.write(output_pdf)

                # Initialize a new writer and update the current student's name
                writer = PyPDF2.PdfWriter()
                lines = text.splitlines()
                for line in lines:
                    if "Branch" in line:
                        current_student_name = line.split(":")[-1].strip()
                        break
            if writer:
                writer.add_page(page)

        # Save the last student's PDF
        if writer and current_student_name:
            output_file = os.path.join(output_folder, f"{current_student_name}.pdf")
            with open(output_file, 'wb') as output_pdf:
                writer.write(output_pdf)

    logger.info("Resumes have been split and saved to: %s" ,output_folder)


def pdf_to_text_pymupdf(file_path: str) -> str:
    """
    Reads a PDF using PyMuPDF and returns the extracted text.
    """
    text = ""
    with fitz.open(file_path) as pdf_file:
        for page in pdf_file:
            text += page.get_text()
    return text
