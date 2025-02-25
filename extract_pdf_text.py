import os
import PyPDF2
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import re
from typing import List, Optional
import logging
from urllib.parse import urljoin
import html2text
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction.log'),
        logging.StreamHandler()
    ]
)

class TextCleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Replace multiple newlines with single newline
        text = re.sub(r'\n\s*\n', '\n', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters except basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\']', ' ', text)
        # Remove any URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Remove email addresses
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()

class WebScraper:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True
        
    def scrape_website(self, url: str) -> Optional[str]:
        """Scrape text content from a website with improved HTML handling."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer']):
                element.decompose()
            
            # Extract text from main content areas
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            if main_content:
                text = self.html_converter.handle(str(main_content))
            else:
                text = self.html_converter.handle(str(soup))
            
            return TextCleaner.clean_text(text)
            
        except Exception as e:
            logging.error(f"Error scraping website {url}: {str(e)}")
            return None

class PDFExtractor:
    @staticmethod
    def extract_text_from_pdf(pdf_file: str) -> Optional[str]:
        """Extract text from a PDF file with improved handling."""
        if not os.path.exists(pdf_file):
            logging.error(f"File not found: {pdf_file}")
            return None
        
        try:
            with open(pdf_file, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text_parts = []
                
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(TextCleaner.clean_text(page_text))
                    except Exception as e:
                        logging.warning(f"Error extracting text from page {page_num} in {pdf_file}: {str(e)}")
                
                return '\n'.join(text_parts) if text_parts else None
                
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_file}: {str(e)}")
            return None

class ContentManager:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def save_content(self, filename: str, content: str) -> bool:
        """Save content to a file with error handling."""
        try:
            output_file = self.output_dir / filename
            output_file.write_text(content, encoding="utf-8")
            logging.info(f"Successfully saved content to {filename}")
            return True
        except Exception as e:
            logging.error(f"Error saving content to {filename}: {str(e)}")
            return False

def main():
    # Define the list of PDF files
    pdf_files = [
        r"C:\Users\Ajithkumar.p\Downloads\droidal web\The company - Droidal.pdf",
        r"C:\Users\Ajithkumar.p\Downloads\droidal web\EDI Automation Services in Healthcare - Copy.pdf",
        r"C:\Users\Ajithkumar.p\Downloads\droidal web\Healthcare Denial Management Automation Services.pdf",
        r"C:\Users\Ajithkumar.p\Downloads\droidal web\Compliance and Audit Automation Services in Healthcare _ Droidal.pdf",
        r"C:\Users\Ajithkumar.p\Downloads\droidal web\home droidal.pdf",
        r"C:\Users\Ajithkumar.p\Downloads\droidal web\Jobs - Droidal.pdf",
        r"C:\Users\Ajithkumar.p\Downloads\droidal web\News & Blog - Droidal.pdf",
        r"C:\Users\Ajithkumar.p\Downloads\droidal web\Case Studies - Droidal.pdf",
        r"C:\Users\Ajithkumar.p\Downloads\droidal web\AI for healthcare - Droidal.pdf",
        r"C:\Users\Ajithkumar.p\Downloads\droidal web\Why Droidal - Droidal.pdf"
    ]

    # Initialize components
    content_manager = ContentManager("output")
    web_scraper = WebScraper('https://droidal.com/')
    pdf_extractor = PDFExtractor()

    # Scrape website
    logging.info("Starting website scraping...")
    website_text = web_scraper.scrape_website('https://droidal.com/') or ""
    content_manager.save_content("droidal_website_text.txt", website_text)

    # Process PDFs
    logging.info("Starting PDF processing...")
    pdf_texts = []
    for pdf_file in pdf_files:
        logging.info(f"Processing {Path(pdf_file).name}")
        pdf_text = pdf_extractor.extract_text_from_pdf(pdf_file)
        if pdf_text:
            pdf_texts.append(pdf_text)
        time.sleep(0.1)  # Small delay to prevent resource overload

    # Save cleaned PDF content
    cleaned_pdf_text = '\n'.join(pdf_texts)
    content_manager.save_content("cleaned_droidal_text.txt", cleaned_pdf_text)

    # Combine all content
    logging.info("Combining content...")
    combined_text = f"{website_text}\n\n{cleaned_pdf_text}"
    content_manager.save_content("droidal_combined_text.txt", combined_text)

    logging.info("Content extraction and processing completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Critical error in main execution: {str(e)}")