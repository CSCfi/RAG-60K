import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz  # PyMuPDF
from tqdm import tqdm


def extract_text_without_headers_footers(pdf_path):
    filename = os.path.basename(pdf_path)
    # Open the PDF file
    doc = fitz.open(pdf_path)

    # Initialize an empty string to store the extracted text
    extracted_text = ""

    # Iterate through each page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Load each page

        # Get page dimensions
        width, height = page.rect.width, page.rect.height

        # Define header and footer heights
        if page_num == 0:  # First page
            header_height = 100
        else:  # Subsequent pages
            header_height = 50

        footer_height = 50

        # Define the rectangle to exclude header and footer (we want the main body)
        body_rect = fitz.Rect(0, header_height, width, height - footer_height)

        # Extract text within the defined rectangle (excluding header and footer)
        text = page.get_text("text", clip=body_rect)

        # Check if 'References' is in the extracted text
        # If 'References' is found, we stop extracting and keep only the text before it
        if "References" in text:
            # Split the text at 'References' and keep only the part before it
            extracted_text += text.split("References")[0]
            break  # Stop processing after finding 'References'

        # If 'References' is not found, continue adding text
        extracted_text += text + "\n"

    # Return the dictionary with filename and text
    return {"filename": filename, "text": extracted_text}


def extract_text_from_pdfs_in_folder(folder_path):
    # Initialize a list to store the results
    results = []

    # Get all the PDF files from the folder
    pdf_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.lower().endswith(".pdf")]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # Submit all the tasks for concurrent processing
        future_to_pdf = {executor.submit(extract_text_without_headers_footers, pdf_path): pdf_path for pdf_path in pdf_files}

        # Process results as they are completed
        for future in tqdm(as_completed(future_to_pdf), total=len(pdf_files)):
            result = future.result()
            results.append(result)

    return results


def save_texts_as_json(pdf_texts, output_file):
    # Save the extracted texts as a JSON file
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(pdf_texts, json_file, ensure_ascii=False, indent=4)


# Usage example
folder_path = "./copernicus"  # Replace with your folder path containing PDFs
pdf_texts = extract_text_from_pdfs_in_folder(folder_path)

# Save the extracted texts as a JSON file
output_file = "./extracted_texts.json"  # Specify your output JSON file path
save_texts_as_json(pdf_texts, output_file)

print(f"Extracted texts saved to {output_file}")
