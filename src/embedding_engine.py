"""
Embedding Engine Module
Handles text embeddings and FAISS vector database operations
"""

import asyncio
import numpy as np
import faiss
import logging
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """Manages text embeddings and vector similarity search using FAISS"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.texts = []
        self.metadata = []
        self.embedding_dim = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            # Initialize FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            
            self.is_initialized = True
            logger.info(f"Embedding engine initialized with dimension {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Error initializing embedding engine: {str(e)}")
            raise
    
    async def index_content(self, document_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create embeddings for document content and index in FAISS
        
        Args:
            document_content: Processed document content
            
        Returns:
            Dictionary with indexed content and metadata
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Clear previous data
            self.texts = []
            self.metadata = []
            
            # Extract text chunks for embedding
            text_chunks = self._create_text_chunks(document_content)
            
            if not text_chunks:
                raise ValueError("No text content found to index")
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(text_chunks)} text chunks")
            embeddings = await self._generate_embeddings(text_chunks)
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Clear and rebuild FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(embeddings.astype('float32'))
            
            # Store texts and metadata
            self.texts = [chunk['text'] for chunk in text_chunks]
            self.metadata = [chunk['metadata'] for chunk in text_chunks]
            
            indexed_content = {
                'total_chunks': len(text_chunks),
                'embedding_dimension': self.embedding_dim,
                'index_size': self.index.ntotal,
                'chunks': text_chunks
            }
            
            logger.info(f"Successfully indexed {len(text_chunks)} text chunks")
            return indexed_content
            
        except Exception as e:
            logger.error(f"Error indexing content: {str(e)}")
            raise
    
    async def search_similar(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for similar text chunks using vector similarity
        
        Args:
            query: Search query
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar text chunks with scores
        """
        if not self.is_initialized or self.index.ntotal == 0:
            return []
        
        try:
            # Generate query embedding
            query_embedding = await self._generate_embeddings([query])
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= threshold and idx < len(self.texts):
                    results.append({
                        'text': self.texts[idx],
                        'similarity_score': float(score),
                        'metadata': self.metadata[idx],
                        'index': int(idx)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar content: {str(e)}")
            return []
    
    def _create_text_chunks(self, document_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create text chunks from document content for embedding
        Implements sliding window approach for better context preservation
        """
        chunks = []
        chunk_size = 512  # Characters per chunk
        overlap = 100     # Character overlap between chunks
        
        # Main document text
        main_text = document_content.get('text', '')
        if main_text:
            text_chunks = self._split_text_with_overlap(main_text, chunk_size, overlap)
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    'text': chunk,
                    'metadata': {
                        'source': 'main_text',
                        'chunk_id': i,
                        'type': 'text_chunk'
                    }
                })
        
        # Section-based chunks
        sections = document_content.get('sections', {})
        for section_name, section_text in sections.items():
            if section_text and len(section_text.strip()) > 50:
                section_chunks = self._split_text_with_overlap(section_text, chunk_size, overlap)
                for i, chunk in enumerate(section_chunks):
                    chunks.append({
                        'text': chunk,
                        'metadata': {
                            'source': 'section',
                            'section_name': section_name,
                            'chunk_id': i,
                            'type': 'section_chunk'
                        }
                    })
        
        # Table content
        tables = document_content.get('tables', [])
        for table_idx, table in enumerate(tables):
            table_text = self._table_to_text(table.get('data', []))
            if table_text:
                chunks.append({
                    'text': table_text,
                    'metadata': {
                        'source': 'table',
                        'table_id': table_idx,
                        'type': 'table_content'
                    }
                })
        
        # Page-based chunks (for PDFs)
        pages = document_content.get('pages', [])
        for page in pages:
            page_text = page.get('text', '')
            if page_text and len(page_text.strip()) > 50:
                page_chunks = self._split_text_with_overlap(page_text, chunk_size, overlap)
                for i, chunk in enumerate(page_chunks):
                    chunks.append({
                        'text': chunk,
                        'metadata': {
                            'source': 'page',
                            'page_number': page.get('page_number'),
                            'chunk_id': i,
                            'type': 'page_chunk'
                        }
                    })
        
        # Remove duplicates and empty chunks
        unique_chunks = []
        seen_texts = set()
        
        for chunk in chunks:
            text = chunk['text'].strip()
            if text and len(text) > 20 and text not in seen_texts:
                seen_texts.add(text)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _split_text_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for punct in ['. ', '! ', '? ', '\n']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct > start + chunk_size // 2:
                        end = last_punct + 1
                        break
            
            chunks.append(text[start:end].strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _table_to_text(self, table_data: List[List[str]]) -> str:
        """Convert table data to searchable text"""
        if not table_data:
            return ""
        
        text_parts = []
        
        # Assume first row is headers
        if table_data:
            headers = table_data[0]
            for row in table_data[1:]:
                row_texts = []
                for i, cell in enumerate(row):
                    if i < len(headers) and cell.strip():
                        row_texts.append(f"{headers[i]}: {cell}")
                if row_texts:
                    text_parts.append("; ".join(row_texts))
        
        return ". ".join(text_parts)
    
    async def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        try:
            # Use asyncio to run in thread pool for better performance
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, 
                self.model.encode, 
                texts
            )
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def get_embedding_dimensions(self) -> int:
        """Get the embedding dimension"""
        return self.embedding_dim if self.embedding_dim else 0
    
    def save_index(self, filepath: str):
        """Save FAISS index and metadata to disk"""
        try:
            index_path = f"{filepath}.index"
            metadata_path = f"{filepath}.metadata"
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save texts and metadata
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'texts': self.texts,
                    'metadata': self.metadata,
                    'embedding_dim': self.embedding_dim
                }, f)
            
            logger.info(f"Index saved to {index_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def load_index(self, filepath: str):
        """Load FAISS index and metadata from disk"""
        try:
            index_path = f"{filepath}.index"
            metadata_path = f"{filepath}.metadata"
            
            if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
                raise FileNotFoundError(f"Index files not found: {filepath}")
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load texts and metadata
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.texts = data['texts']
                self.metadata = data['metadata']
                self.embedding_dim = data['embedding_dim']
            
            logger.info(f"Index loaded from {index_path}")
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise
