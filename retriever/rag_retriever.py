"""
RAG Retriever module that combines Chroma vector search with Ollama LLM.
Handles query processing, document retrieval, and answer generation.
"""

import requests
import json
from typing import List, Dict, Any, Optional
import logging
import time
from vector_store.chroma_db import ChromaVectorStore
from embeddings.embedding_generator import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieval-Augmented Generation system using Chroma and Ollama."""
    
    def __init__(self, 
                 collection_name: str = "worldbank_documents",
                 ollama_model: str = "llama2:latest",
                 ollama_base_url: str = "http://localhost:11434",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG retriever.
        
        Args:
            collection_name: Chroma collection name
            ollama_model: Ollama model name
            ollama_base_url: Ollama API base URL
            embedding_model: SentenceTransformer model name
        """
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        
        # Initialize vector store
        self.vector_store = ChromaVectorStore(collection_name)
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        
        logger.info(f"RAG Retriever initialized with model: {ollama_model}")
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                if self.ollama_model in model_names:
                    logger.info(f"Ollama connected. Model '{self.ollama_model}' is available.")
                    return True
                else:
                    logger.warning(f"Model '{self.ollama_model}' not found. Available models: {model_names}")
                    return False
            else:
                logger.error(f"Ollama API returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {str(e)}")
            return False
    
    def retrieve_documents(self, 
                         query: str, 
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        try:
            results = self.vector_store.query(
                query_text=query,
                embedding_model=self.embedding_generator.model,
                n_results=top_k
            )
            
            if "error" in results:
                logger.error(f"Retrieval error: {results['error']}")
                return []
            
            return results["results"]
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def generate_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generate context from retrieved documents.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc["metadata"]
            context_part = f"""
Document {i}:
Source: {metadata.get('filename', 'Unknown')} - Page {metadata.get('page_number', 'Unknown')}
Content: {doc['document']}
Similarity Score: {doc['similarity']:.4f}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate_prompt(self, query: str, context: str) -> str:
        """
        Generate prompt for Ollama LLM.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful assistant that answers questions based on World Bank development reports. Use only the provided context to answer the question. If the context doesn't contain enough information to answer the question, say so clearly.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Answer the question based only on the provided context
2. Cite the source documents and page numbers
3. If multiple documents provide relevant information, synthesize them
4. If the context is insufficient, state that clearly
5. Provide a concise, well-structured answer

ANSWER:"""
        
        return prompt
    
    def call_ollama(self, prompt: str) -> Optional[str]:
        """
        Call Ollama API to generate response.
        
        Args:
            prompt: Prompt for the LLM
            
        Returns:
            Generated response or None if failed
        """
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            }
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            return None
    
    def answer_query(self, 
                    query: str, 
                    top_k: int = 5,
                    include_sources: bool = True) -> Dict[str, Any]:
        """
        Answer a user query using RAG.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            include_sources: Whether to include source information
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant documents
            logger.info(f"Retrieving documents for query: {query}")
            retrieved_docs = self.retrieve_documents(query, top_k)
            
            if not retrieved_docs:
                return {
                    "query": query,
                    "answer": "I couldn't find relevant information in the World Bank reports to answer your question.",
                    "sources": [],
                    "retrieval_time": time.time() - start_time,
                    "error": "No relevant documents found"
                }
            
            # Step 2: Generate context
            context = self.generate_context(retrieved_docs)
            
            # Step 3: Generate prompt
            prompt = self.generate_prompt(query, context)
            
            # Step 4: Generate answer using Ollama
            logger.info("Generating answer with Ollama...")
            answer = self.call_ollama(prompt)
            
            if answer is None:
                return {
                    "query": query,
                    "answer": "I encountered an error while generating the answer. Please try again.",
                    "sources": [],
                    "retrieval_time": time.time() - start_time,
                    "error": "LLM generation failed"
                }
            
            # Step 5: Prepare response
            response = {
                "query": query,
                "answer": answer.strip(),
                "retrieval_time": time.time() - start_time,
                "documents_retrieved": len(retrieved_docs)
            }
            
            if include_sources:
                sources = []
                for doc in retrieved_docs:
                    sources.append({
                        "filename": doc["metadata"].get("filename", "Unknown"),
                        "page": doc["metadata"].get("page_number", "Unknown"),
                        "similarity": doc["similarity"],
                        "chunk_index": doc["metadata"].get("chunk_index", "Unknown")
                    })
                response["sources"] = sources
            
            logger.info(f"Answer generated in {response['retrieval_time']:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"Error answering query: {str(e)}")
            return {
                "query": query,
                "answer": "An error occurred while processing your question. Please try again.",
                "sources": [],
                "retrieval_time": time.time() - start_time,
                "error": str(e)
            }


if __name__ == "__main__":
    # Example usage
    retriever = RAGRetriever(ollama_model="llama2:latest")
    
    # Check Ollama connection
    if not retriever.check_ollama_connection():
        print("Error: Cannot connect to Ollama. Please make sure Ollama is running.")
        print("Install Ollama from https://ollama.ai/ and run: ollama serve")
        exit(1)
    
    # Test queries
    test_queries = [
        "What are the main challenges in global development?",
        "How does climate change affect developing countries?",
        "What are the recommendations for economic growth?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        result = retriever.answer_query(query, top_k=3)
        
        print(f"\nAnswer:")
        print(result["answer"])
        
        if "sources" in result and result["sources"]:
            print(f"\nSources:")
            for i, source in enumerate(result["sources"], 1):
                print(f"{i}. {source['filename']} - Page {source['page']} (Similarity: {source['similarity']:.4f})")
        
        print(f"\nRetrieval time: {result['retrieval_time']:.2f} seconds")
        if "documents_retrieved" in result:
            print(f"Documents retrieved: {result['documents_retrieved']}")
        else:
            print("Documents retrieved: Unknown")
