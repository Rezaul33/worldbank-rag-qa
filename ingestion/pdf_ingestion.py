"""
PDF ingestion module for World Bank reports using pdfplumber.
Extracts text from PDF files and stores with metadata.
"""

import os
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFIngestion:
    """Handles PDF text extraction and metadata collection."""
    
    def __init__(self, pdf_dir: str = "data/world_bank_pdfs"):
        """
        Initialize PDF ingestion.
        
        Args:
            pdf_dir: Directory containing PDF files
        """
        self.pdf_dir = Path(pdf_dir)
        self.extracted_documents = []
    
    def get_pdf_files(self) -> List[Path]:
        """Get all PDF files from the directory."""
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        return pdf_files
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        document = {
            "filename": pdf_path.name,
            "filepath": str(pdf_path),
            "pages": [],
            "total_pages": 0,
            "full_text": ""
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                document["total_pages"] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        
                        page_data = {
                            "page_number": page_num,
                            "text": page_text if page_text else "",
                            "char_count": len(page_text) if page_text else 0
                        }
                        
                        document["pages"].append(page_data)
                        
                        if page_text:
                            document["full_text"] += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num} in {pdf_path.name}: {str(e)}")
                        page_data = {
                            "page_number": page_num,
                            "text": "",
                            "char_count": 0,
                            "error": str(e)
                        }
                        document["pages"].append(page_data)
                
                logger.info(f"Extracted {len(document['pages'])} pages from {pdf_path.name}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path.name}: {str(e)}")
            document["error"] = str(e)
        
        return document
    
    def extract_all_pdfs(self) -> List[Dict[str, Any]]:
        """
        Extract text from all PDF files in the directory.
        
        Returns:
            List of document dictionaries with text and metadata
        """
        pdf_files = self.get_pdf_files()
        
        for pdf_path in pdf_files:
            logger.info(f"Processing: {pdf_path.name}")
            document = self.extract_text_from_pdf(pdf_path)
            self.extracted_documents.append(document)
        
        logger.info(f"Successfully extracted text from {len(self.extracted_documents)} documents")
        return self.extracted_documents
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about extracted documents."""
        if not self.extracted_documents:
            return {"error": "No documents extracted yet"}
        
        total_pages = sum(doc["total_pages"] for doc in self.extracted_documents)
        total_chars = sum(len(doc["full_text"]) for doc in self.extracted_documents)
        
        stats = {
            "total_documents": len(self.extracted_documents),
            "total_pages": total_pages,
            "total_characters": total_chars,
            "average_pages_per_doc": total_pages / len(self.extracted_documents),
            "documents": [
                {
                    "filename": doc["filename"],
                    "pages": doc["total_pages"],
                    "characters": len(doc["full_text"])
                }
                for doc in self.extracted_documents
            ]
        }
        
        return stats


if __name__ == "__main__":
    # Example usage
    ingestion = PDFIngestion()
    documents = ingestion.extract_all_pdfs()
    
    # Print statistics
    stats = ingestion.get_document_stats()
    print(f"Extracted {stats['total_documents']} documents")
    print(f"Total pages: {stats['total_pages']}")
    print(f"Total characters: {stats['total_characters']}")
    
    # Show first document info
    if documents:
        first_doc = documents[0]
        print(f"\nFirst document: {first_doc['filename']}")
        print(f"Pages: {first_doc['total_pages']}")
        print(f"Sample text: {first_doc['full_text'][:200]}...")
