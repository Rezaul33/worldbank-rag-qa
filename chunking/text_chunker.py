"""
Text chunking module for splitting documents into optimal-sized chunks.
Uses tiktoken for accurate token counting and implements overlapping chunks.
"""

import tiktoken
from typing import List, Dict, Any, Optional
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """Handles text chunking with token-based sizing and overlap."""
    
    def __init__(self, 
                 chunk_size: int = 750, 
                 overlap_size: int = 100,
                 model_name: str = "gpt-3.5-turbo"):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Target chunk size in tokens (default: 750)
            overlap_size: Overlap size in tokens (default: 100)
            model_name: Model name for tiktoken encoding
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.chunks = []
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'\d+/\d+/\d+', '', text)
        
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\"\'\/\&\@\#\$\%\*\+\=\[\]\{\}]', ' ', text)
        
        return text.strip()
    
    def split_text_into_chunks(self, 
                             text: str, 
                             metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks based on token count.
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Split into sentences for better chunk boundaries
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence exceeds chunk size, create chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk with metadata
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": chunk_index,
                    "token_count": current_tokens,
                    "char_count": len(current_chunk)
                })
                
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": chunk_metadata
                })
                
                chunk_index += 1
                
                # Start new chunk with overlap
                if self.overlap_size > 0:
                    # Find sentences to keep for overlap
                    overlap_text = ""
                    overlap_tokens = 0
                    
                    # Go backwards from current position to find overlap
                    for j in range(i-1, max(-1, i-20), -1):
                        if j < 0:
                            break
                        prev_sentence = sentences[j]
                        prev_tokens = self.count_tokens(prev_sentence)
                        
                        if overlap_tokens + prev_tokens <= self.overlap_size:
                            overlap_text = prev_sentence + " " + overlap_text
                            overlap_tokens += prev_tokens
                        else:
                            break
                    
                    current_chunk = overlap_text.strip()
                    current_tokens = overlap_tokens
                else:
                    current_chunk = ""
                    current_tokens = 0
            
            # Add current sentence to chunk
            current_chunk += (" " if current_chunk else "") + sentence
            current_tokens += sentence_tokens
            i += 1
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": chunk_index,
                "token_count": current_tokens,
                "char_count": len(current_chunk)
            })
            
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": chunk_metadata
            })
        
        return chunks
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single document.
        
        Args:
            document: Document dictionary with pages and metadata
            
        Returns:
            List of all chunks from the document
        """
        all_chunks = []
        
        base_metadata = {
            "filename": document["filename"],
            "filepath": document["filepath"],
            "total_pages": document["total_pages"]
        }
        
        # Option 1: Chunk by page (keeps page boundaries)
        for page in document["pages"]:
            if page["text"] and page["text"].strip():
                page_metadata = base_metadata.copy()
                page_metadata.update({
                    "page_number": page["page_number"],
                    "page_char_count": page["char_count"]
                })
                
                page_chunks = self.split_text_into_chunks(page["text"], page_metadata)
                all_chunks.extend(page_chunks)
        
        # Option 2: Chunk entire document as one text
        # Uncomment below if you prefer document-level chunking
        # full_text = document["full_text"]
        # doc_chunks = self.split_text_into_chunks(full_text, base_metadata)
        # all_chunks.extend(doc_chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {document['filename']}")
        return all_chunks
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            doc_chunks = self.chunk_document(doc)
            all_chunks.extend(doc_chunks)
        
        self.chunks = all_chunks
        logger.info(f"Created total of {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def get_chunking_stats(self) -> Dict[str, Any]:
        """Get statistics about chunking results."""
        if not self.chunks:
            return {"error": "No chunks available"}
        
        token_counts = [chunk["metadata"]["token_count"] for chunk in self.chunks]
        char_counts = [chunk["metadata"]["char_count"] for chunk in self.chunks]
        
        stats = {
            "total_chunks": len(self.chunks),
            "total_tokens": sum(token_counts),
            "total_characters": sum(char_counts),
            "average_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "average_chars_per_chunk": sum(char_counts) / len(char_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
            "documents_processed": len(set(chunk["metadata"]["filename"] for chunk in self.chunks))
        }
        
        return stats


if __name__ == "__main__":
    # Example usage
    from ingestion.pdf_ingestion import PDFIngestion
    
    # First ingest documents
    ingestion = PDFIngestion()
    documents = ingestion.extract_all_pdfs()
    
    # Then chunk them
    chunker = TextChunker(chunk_size=750, overlap_size=100)
    chunks = chunker.chunk_documents(documents)
    
    # Print statistics
    stats = chunker.get_chunking_stats()
    print(f"Chunking Statistics:")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Average tokens per chunk: {stats['average_tokens_per_chunk']:.1f}")
    print(f"Min/Max tokens: {stats['min_tokens']} / {stats['max_tokens']}")
    
    # Show sample chunk
    if chunks:
        sample_chunk = chunks[0]
        print(f"\nSample chunk:")
        print(f"Document: {sample_chunk['metadata']['filename']}")
        print(f"Page: {sample_chunk['metadata']['page_number']}")
        print(f"Tokens: {sample_chunk['metadata']['token_count']}")
        print(f"Text: {sample_chunk['text'][:200]}...")
