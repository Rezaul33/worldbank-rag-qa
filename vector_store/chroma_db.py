"""
Chroma vector database module for storing and retrieving document embeddings.
Provides persistent storage with metadata support and similarity search.
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Handles Chroma vector database operations."""
    
    def __init__(self, 
                 collection_name: str = "worldbank_documents",
                 persist_directory: str = "vector_store/chroma_db"):
        """
        Initialize Chroma vector store.
        
        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client with persistent storage
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "World Bank Documents RAG Collection"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_chunks(self, chunks_with_embeddings: List[Dict[str, Any]]) -> bool:
        """
        Add chunks with embeddings to the Chroma collection.
        
        Args:
            chunks_with_embeddings: List of chunks with embeddings and metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not chunks_with_embeddings:
            logger.warning("No chunks provided to add")
            return False
        
        try:
            start_time = time.time()
            
            # Prepare data for Chroma
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks_with_embeddings):
                # Create unique ID
                chunk_id = f"{chunk['metadata']['filename'].replace('.pdf', '')}_page{chunk['metadata']['page_number']}_chunk{chunk['metadata']['chunk_index']}"
                ids.append(chunk_id)
                
                # Add embedding
                embeddings.append(chunk["embedding"])
                
                # Add document text
                documents.append(chunk["text"])
                
                # Prepare metadata (convert all values to strings for Chroma)
                metadata = chunk["metadata"].copy()
                # Ensure all metadata values are JSON serializable
                serialized_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        serialized_metadata[key] = value
                    else:
                        serialized_metadata[key] = str(value)
                metadatas.append(serialized_metadata)
            
            # Add to Chroma in batches to avoid memory issues
            batch_size = 100
            total_batches = (len(ids) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(ids))
                
                batch_ids = ids[start_idx:end_idx]
                batch_embeddings = embeddings[start_idx:end_idx]
                batch_documents = documents[start_idx:end_idx]
                batch_metadatas = metadatas[start_idx:end_idx]
                
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                
                logger.info(f"Added batch {batch_idx + 1}/{total_batches} ({len(batch_ids)} chunks)")
            
            add_time = time.time() - start_time
            logger.info(f"Successfully added {len(chunks_with_embeddings)} chunks to Chroma in {add_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks to Chroma: {str(e)}")
            return False
    
    def query(self, 
              query_text: str, 
              embedding_model, 
              n_results: int = 10) -> Dict[str, Any]:
        """
        Query the Chroma collection for similar documents.
        
        Args:
            query_text: Query text
            embedding_model: Embedding model to encode query
            n_results: Number of results to return
            
        Returns:
            Dictionary with query results
        """
        try:
            # Generate query embedding
            query_embedding = embedding_model.encode([query_text])[0].tolist()
            
            # Query Chroma
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                result = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": 1 - (results["distances"][0][i] / 2)  # Convert cosine distance to similarity
                }
                formatted_results.append(result)
            
            return {
                "query": query_text,
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            logger.error(f"Error querying Chroma: {str(e)}")
            return {"error": str(e), "results": []}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Chroma collection."""
        try:
            count = self.collection.count()
            
            # Get sample of metadata to understand structure
            sample_results = self.collection.get(limit=5, include=["metadatas"])
            
            stats = {
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "total_documents": count,
                "sample_metadata": sample_results["metadatas"] if sample_results["metadatas"] else []
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False
    
    def load_from_embeddings_file(self, 
                                 embeddings_file: str = "embeddings/chunks_with_embeddings.pkl",
                                 embedding_model=None) -> bool:
        """
        Load chunks from embeddings file and add to Chroma.
        
        Args:
            embeddings_file: Path to the embeddings pickle file
            embedding_model: Embedding model (for validation)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import pickle
            
            with open(embeddings_file, 'rb') as f:
                chunks_with_embeddings = pickle.load(f)
            
            logger.info(f"Loaded {len(chunks_with_embeddings)} chunks from {embeddings_file}")
            
            # Validate embeddings if model provided
            if embedding_model is not None:
                sample_embedding = chunks_with_embeddings[0]["embedding"]
                expected_dim = embedding_model.get_sentence_embedding_dimension()
                if len(sample_embedding) != expected_dim:
                    logger.warning(f"Embedding dimension mismatch: {len(sample_embedding)} vs {expected_dim}")
            
            # Add to Chroma
            return self.add_chunks(chunks_with_embeddings)
            
        except Exception as e:
            logger.error(f"Error loading from embeddings file: {str(e)}")
            return False


if __name__ == "__main__":
    # Example usage
    from embeddings.embedding_generator import EmbeddingGenerator
    
    # Initialize vector store
    vector_store = ChromaVectorStore(collection_name="worldbank_documents")
    
    # Load embeddings and add to Chroma
    logger.info("Loading embeddings and adding to Chroma...")
    success = vector_store.load_from_embeddings_file()
    
    if success:
        # Get collection stats
        stats = vector_store.get_collection_stats()
        print(f"\nCollection Statistics:")
        print(f"Collection name: {stats['collection_name']}")
        print(f"Total documents: {stats['total_documents']}")
        print(f"Persist directory: {stats['persist_directory']}")
        
        # Test query (need embedding model)
        generator = EmbeddingGenerator()
        test_query = "What are the main challenges in global development?"
        
        logger.info(f"Testing query: {test_query}")
        results = vector_store.query(test_query, generator.model, n_results=3)
        
        if "error" not in results:
            print(f"\nQuery Results for: '{test_query}'")
            for i, result in enumerate(results["results"]):
                print(f"\nResult {i+1}:")
                print(f"Similarity: {result['similarity']:.4f}")
                print(f"Document: {result['metadata']['filename']}")
                print(f"Page: {result['metadata']['page_number']}")
                print(f"Text preview: {result['document'][:150]}...")
        else:
            print(f"Query error: {results['error']}")
    else:
        print("Failed to load embeddings into Chroma")
