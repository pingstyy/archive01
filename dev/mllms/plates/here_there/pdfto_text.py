import PyPDF2
import re

def pdf_to_text(pdf_file, text_file):
    # Open the PDF file in binary mode
    with open(pdf_file, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Initialize an empty string to store the text
        text = ""

        # Iterate over all pages in the PDF
        for page_num in range(len(pdf_reader.pages)):
            # Get the current page
            page = pdf_reader.pages[page_num]

            # Extract the text from the page
            page_text = page.extract_text()

            # Remove non-alphanumeric characters and convert to UTF-8
            page_text = re.sub(r'[^a-zA-Z0-9\s]+', ' ', page_text)
            page_text = page_text.encode('utf-8', errors='ignore').decode('utf-8')

            # Append the page text to the overall text
            text += page_text

    # Write the text to the output file with UTF-8 encoding
    with open(text_file, 'w', encoding='utf-8') as file:
        file.write(text)

if __name__ == '__main__':
    # Prompt the user for the PDF file path
    pdf_file = input("Enter the path to the PDF file: ")

    # Prompt the user for the output text file path
    text_file = "output.txt"

    # Call the pdf_to_text function
    pdf_to_text(pdf_file, text_file)
    print("Text extraction completed.")