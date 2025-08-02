"""
Enhanced Embedding Engine with Improved Accuracy
Advanced semantic search, better chunking, and multi-model support
"""

import asyncio
import numpy as np
import faiss
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
import pickle
import os
from pathlib import Path
import json
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Enhanced metadata for text chunks"""
    chunk_id: str
    source: str
    section_name: Optional[str]
    content_type: str
    importance_score: float
    entity_density: float
    keyword_density: float
    position_in_document: float
    word_count: int
    char_count: int

class EnhancedEmbeddingEngine:
    """Enhanced embedding engine with improved accuracy and performance"""
    
    def __init__(self, 
                 primary_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 secondary_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.primary_model_name = primary_model
        self.secondary_model_name = secondary_model
        self.primary_model = None
        self.secondary_model = None
        
        # Multiple indices for different content types
        self.indices = {}
        self.chunk_metadata = {}
        self.texts = {}
        self.embeddings_cache = {}
        
        self.embedding_dim = None
        self.is_initialized = False
        
        # Enhanced chunking parameters
        self.chunk_configs = {
            'default': {'size': 512, 'overlap': 100, 'min_size': 50},
            'definition': {'size': 256, 'overlap': 50, 'min_size': 30},
            'procedure': {'size': 768, 'overlap': 150, 'min_size': 100},
            'financial': {'size': 384, 'overlap': 75, 'min_size': 50},
            'legal': {'size': 640, 'overlap': 120, 'min_size': 80}
        }
        
    async def initialize(self):
        """Initialize embedding models with enhanced configuration"""
        try:
            logger.info(f"Loading primary embedding model: {self.primary_model_name}")
            self.primary_model = SentenceTransformer(self.primary_model_name)
            self.embedding_dim = self.primary_model.get_sentence_embedding_dimension()
            
            # Load secondary model for cross-validation
            logger.info(f"Loading secondary embedding model: {self.secondary_model_name}")
            self.secondary_model = SentenceTransformer(self.secondary_model_name)
            
            # Initialize multiple FAISS indices
            self.indices = {
                'general': faiss.IndexFlatIP(self.embedding_dim),
                'definitions': faiss.IndexFlatIP(self.embedding_dim),
                'procedures': faiss.IndexFlatIP(self.embedding_dim),
                'financial': faiss.IndexFlatIP(self.embedding_dim),
                'temporal': faiss.IndexFlatIP(self.embedding_dim)
            }
            
            # Initialize storage for each index
            for index_name in self.indices.keys():
                self.texts[index_name] = []
                self.chunk_metadata[index_name] = []
            
            self.is_initialized = True
            logger.info(f"Enhanced embedding engine initialized with dimension {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Error initializing enhanced embedding engine: {str(e)}")
            raise
    
    async def index_content(self, document_content: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced content indexing with improved chunking and categorization"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Clear previous data
            for index_name in self.indices.keys():
                self.texts[index_name] = []
                self.chunk_metadata[index_name] = []
                self.indices[index_name] = faiss.IndexFlatIP(self.embedding_dim)
            
            # Enhanced chunk creation with categorization
            categorized_chunks = await self._create_enhanced_chunks(document_content)
            
            total_chunks = 0
            for category, chunks in categorized_chunks.items():
                if chunks:
                    logger.info(f"Processing {len(chunks)} chunks for category: {category}")
                    
                    # Generate embeddings for this category
                    chunk_texts = [chunk['text'] for chunk in chunks]
                    embeddings = await self._generate_enhanced_embeddings(chunk_texts, category)
                    
                    # Normalize embeddings
                    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                    
                    # Add to appropriate index
                    if category in self.indices:
                        self.indices[category].add(embeddings.astype('float32'))
                        self.texts[category] = chunk_texts
                        self.chunk_metadata[category] = [chunk['metadata'] for chunk in chunks]
                    else:
                        # Add to general index if category not found
                        self.indices['general'].add(embeddings.astype('float32'))
                        self.texts['general'].extend(chunk_texts)
                        self.chunk_metadata['general'].extend([chunk['metadata'] for chunk in chunks])
                    
                    total_chunks += len(chunks)
            
            # Create cross-reference index for better search
            await self._create_cross_reference_index(categorized_chunks)
            
            indexed_content = {
                'total_chunks': total_chunks,
                'categories': list(categorized_chunks.keys()),
                'category_counts': {k: len(v) for k, v in categorized_chunks.items()},
                'embedding_dimension': self.embedding_dim,
                'index_sizes': {k: v.ntotal for k, v in self.indices.items()},
                'enhanced_features': True
            }
            
            logger.info(f"Successfully indexed {total_chunks} chunks across {len(categorized_chunks)} categories")
            return indexed_content
            
        except Exception as e:
            logger.error(f"Error indexing enhanced content: {str(e)}")
            raise
    
    async def _create_enhanced_chunks(self, document_content: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Create enhanced chunks with intelligent categorization"""
        categorized_chunks = {
            'general': [],
            'definitions': [],
            'procedures': [],
            'financial': [],
            'temporal': []
        }
        
        # Process structured content if available
        if 'structured_content' in document_content:
            for block in document_content['structured_content']:
                category = self._determine_chunk_category(block)
                chunks = await self._create_chunks_for_block(block, category)
                categorized_chunks[category].extend(chunks)
        
        # Process sections
        if 'sections' in document_content:
            for section_name, section_data in document_content['sections'].items():
                if isinstance(section_data, dict):
                    section_text = section_data.get('content', '')
                    section_type = section_data.get('section_type', 'general')
                else:
                    section_text = str(section_data)
                    section_type = 'general'
                
                if section_text:
                    category = self._map_section_type_to_category(section_type)
                    chunks = await self._create_adaptive_chunks(
                        section_text, 
                        category, 
                        {'source': 'section', 'section_name': section_name}
                    )
                    categorized_chunks[category].extend(chunks)
        
        # Process main text if no structured content
        if not categorized_chunks['general'] and 'text' in document_content:
            main_text = document_content['text']
            chunks = await self._create_adaptive_chunks(
                main_text, 
                'general', 
                {'source': 'main_text'}
            )
            categorized_chunks['general'].extend(chunks)
        
        # Remove empty categories
        categorized_chunks = {k: v for k, v in categorized_chunks.items() if v}
        
        return categorized_chunks
    
    def _determine_chunk_category(self, block: Dict[str, Any]) -> str:
        """Determine the category of a content block"""
        block_type = block.get('type', 'general')
        
        category_mapping = {
            'definition': 'definitions',
            'procedure': 'procedures',
            'financial': 'financial',
            'temporal': 'temporal',
            'coverage': 'general',
            'exclusion': 'general',
            'general': 'general'
        }
        
        return category_mapping.get(block_type, 'general')
    
    def _map_section_type_to_category(self, section_type: str) -> str:
        """Map section types to chunk categories"""
        mapping = {
            'definition': 'definitions',
            'procedure': 'procedures',
            'financial': 'financial',
            'temporal': 'temporal',
            'preamble': 'general',
            'heading': 'general'
        }
        
        return mapping.get(section_type, 'general')
    
    async def _create_chunks_for_block(self, block: Dict[str, Any], category: str) -> List[Dict[str, Any]]:
        """Create chunks for a specific content block"""
        text = block.get('text', '')
        if not text or len(text.strip()) < 20:
            return []
        
        # Get chunk configuration for this category
        config = self.chunk_configs.get(category, self.chunk_configs['default'])
        
        # Create chunks with metadata
        chunks = []
        chunk_texts = self._split_text_adaptive(text, config['size'], config['overlap'], config['min_size'])
        
        for i, chunk_text in enumerate(chunk_texts):
            chunk_id = hashlib.md5(f"{block.get('id', 0)}_{i}_{chunk_text[:50]}".encode()).hexdigest()
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                source=f"block_{block.get('id', 0)}",
                section_name=None,
                content_type=block.get('type', 'general'),
                importance_score=self._calculate_importance_score(chunk_text, block),
                entity_density=self._calculate_entity_density(chunk_text, block.get('entities', {})),
                keyword_density=self._calculate_keyword_density(chunk_text),
                position_in_document=block.get('id', 0) / 100,  # Normalized position
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text)
            )
            
            chunks.append({
                'text': chunk_text,
                'metadata': metadata.__dict__
            })
        
        return chunks
    
    async def _create_adaptive_chunks(self, text: str, category: str, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create adaptive chunks based on content type"""
        if not text or len(text.strip()) < 20:
            return []
        
        config = self.chunk_configs.get(category, self.chunk_configs['default'])
        chunk_texts = self._split_text_adaptive(text, config['size'], config['overlap'], config['min_size'])
        
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk_id = hashlib.md5(f"{category}_{i}_{chunk_text[:50]}".encode()).hexdigest()
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                source=base_metadata.get('source', 'unknown'),
                section_name=base_metadata.get('section_name'),
                content_type=category,
                importance_score=self._calculate_importance_score(chunk_text),
                entity_density=self._calculate_entity_density(chunk_text),
                keyword_density=self._calculate_keyword_density(chunk_text),
                position_in_document=i / len(chunk_texts),
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text)
            )
            
            chunks.append({
                'text': chunk_text,
                'metadata': metadata.__dict__
            })
        
        return chunks
    
    def _split_text_adaptive(self, text: str, chunk_size: int, overlap: int, min_size: int) -> List[str]:
        """Adaptive text splitting with intelligent boundary detection"""
        if len(text) <= chunk_size:
            return [text] if len(text) >= min_size else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                # Last chunk
                remaining_text = text[start:].strip()
                if len(remaining_text) >= min_size:
                    chunks.append(remaining_text)
                elif chunks:  # Merge with previous chunk if too small
                    chunks[-1] += ' ' + remaining_text
                break
            
            # Find optimal break point
            break_point = self._find_optimal_break_point(text, start, end)
            
            chunk = text[start:break_point].strip()
            if len(chunk) >= min_size:
                chunks.append(chunk)
            
            # Calculate next start position with overlap
            start = max(break_point - overlap, start + min_size)
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _find_optimal_break_point(self, text: str, start: int, end: int) -> int:
        """Find optimal break point for text chunking"""
        # Priority order for break points
        break_patterns = [
            r'\.\s+',      # Sentence end
            r'[!?]\s+',    # Exclamation/question end
            r';\s+',       # Semicolon
            r':\s+',       # Colon
            r',\s+',       # Comma
            r'\s+',        # Any whitespace
        ]
        
        search_window = min(100, (end - start) // 4)  # Search within last 25% of chunk
        search_start = max(start, end - search_window)
        
        for pattern in break_patterns:
            import re
            matches = list(re.finditer(pattern, text[search_start:end]))
            if matches:
                # Take the last match (closest to end)
                last_match = matches[-1]
                return search_start + last_match.end()
        
        # Fallback to original end
        return end
    
    def _calculate_importance_score(self, text: str, block: Dict[str, Any] = None) -> float:
        """Calculate importance score for a text chunk"""
        score = 0.5  # Base score
        
        text_lower = text.lower()
        
        # High importance keywords
        high_importance_words = [
            'coverage', 'covered', 'benefit', 'entitled', 'eligible',
            'exclusion', 'excluded', 'not covered', 'limitation',
            'premium', 'deductible', 'amount', 'cost', 'fee',
            'procedure', 'process', 'steps', 'how to', 'must', 'shall',
            'definition', 'means', 'defined as', 'refers to'
        ]
        
        for word in high_importance_words:
            if word in text_lower:
                score += 0.1
        
        # Boost for financial information
        import re
        if re.search(r'\$[\d,]+|INR\s*[\d,]+|\d+(?:\.\d+)?%', text):
            score += 0.2
        
        # Boost for time periods
        if re.search(r'\d+\s*(?:days?|months?|years?)', text_lower):
            score += 0.15
        
        # Boost for structured content
        if block and block.get('type') in ['definition', 'procedure', 'financial']:
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_entity_density(self, text: str, entities: Dict[str, List[str]] = None) -> float:
        """Calculate entity density in text"""
        if not entities:
            # Basic entity detection
            import re
            entity_patterns = [
                r'\$[\d,]+',           # Money
                r'\d+(?:\.\d+)?%',     # Percentages
                r'\d+\s*(?:days?|months?|years?)',  # Time periods
                r'[A-Z][a-z]+\s+Insurance',  # Insurance companies
            ]
            
            entity_count = 0
            for pattern in entity_patterns:
                entity_count += len(re.findall(pattern, text))
            
            word_count = len(text.split())
            return min(1.0, entity_count / max(word_count, 1))
        
        # Use provided entities
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        word_count = len(text.split())
        return min(1.0, total_entities / max(word_count, 1))
    
    def _calculate_keyword_density(self, text: str) -> float:
        """Calculate keyword density for important terms"""
        important_keywords = [
            'policy', 'insurance', 'coverage', 'benefit', 'claim',
            'premium', 'deductible', 'exclusion', 'condition', 'term'
        ]
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in important_keywords if keyword in text_lower)
        
        return min(1.0, keyword_count / len(important_keywords))
    
    async def _generate_enhanced_embeddings(self, texts: List[str], category: str) -> np.ndarray:
        """Generate enhanced embeddings with category-specific optimization"""
        try:
            # Use primary model for all categories
            embeddings = await self._generate_embeddings_with_model(texts, self.primary_model)
            
            # For critical categories, use ensemble approach
            if category in ['definitions', 'financial', 'procedures']:
                secondary_embeddings = await self._generate_embeddings_with_model(texts, self.secondary_model)
                
                # Weighted combination (primary model gets more weight)
                embeddings = 0.7 * embeddings + 0.3 * secondary_embeddings
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating enhanced embeddings: {str(e)}")
            raise
    
    async def _generate_embeddings_with_model(self, texts: List[str], model: SentenceTransformer) -> np.ndarray:
        """Generate embeddings with a specific model"""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, model.encode, texts)
        return np.array(embeddings)
    
    async def _create_cross_reference_index(self, categorized_chunks: Dict[str, List[Dict[str, Any]]]):
        """Create cross-reference index for better search across categories"""
        # This could be implemented to create relationships between chunks
        # For now, we'll store category relationships
        self.category_relationships = {}
        
        for category in categorized_chunks.keys():
            related_categories = []
            
            # Define category relationships
            if category == 'definitions':
                related_categories = ['general', 'procedures']
            elif category == 'procedures':
                related_categories = ['definitions', 'financial']
            elif category == 'financial':
                related_categories = ['procedures', 'temporal']
            elif category == 'temporal':
                related_categories = ['financial', 'general']
            else:
                related_categories = ['definitions', 'procedures']
            
            self.category_relationships[category] = related_categories
    
    async def search_similar(self, 
                           query: str, 
                           top_k: int = 5, 
                           threshold: float = 0.3,
                           categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Enhanced similarity search with category filtering and ranking"""
        if not self.is_initialized:
            return []
        
        try:
            # Determine query category
            query_category = await self._classify_query(query)
            
            # Determine which categories to search
            search_categories = categories or self._get_search_categories(query_category)
            
            all_results = []
            
            for category in search_categories:
                if category in self.indices and self.indices[category].ntotal > 0:
                    # Generate query embedding
                    query_embedding = await self._generate_enhanced_embeddings([query], category)
                    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
                    
                    # Search in category index
                    scores, indices = self.indices[category].search(
                        query_embedding.astype('float32'), 
                        min(top_k * 2, self.indices[category].ntotal)
                    )
                    
                    # Process results
                    for score, idx in zip(scores[0], indices[0]):
                        if score >= threshold and idx < len(self.texts[category]):
                            result = {
                                'text': self.texts[category][idx],
                                'similarity_score': float(score),
                                'metadata': self.chunk_metadata[category][idx],
                                'category': category,
                                'index': int(idx)
                            }
                            
                            # Enhanced scoring
                            result['enhanced_score'] = self._calculate_enhanced_score(
                                result, query, query_category
                            )
                            
                            all_results.append(result)
            
            # Sort by enhanced score and return top results
            all_results.sort(key=lambda x: x['enhanced_score'], reverse=True)
            return all_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in enhanced similarity search: {str(e)}")
            return []
    
    async def _classify_query(self, query: str) -> str:
        """Classify query to determine appropriate search strategy"""
        query_lower = query.lower()
        
        # Definition queries
        if any(word in query_lower for word in ['what is', 'define', 'definition', 'means', 'meaning']):
            return 'definitions'
        
        # Procedure queries
        if any(word in query_lower for word in ['how to', 'process', 'procedure', 'steps', 'file', 'apply']):
            return 'procedures'
        
        # Financial queries
        if any(word in query_lower for word in ['cost', 'price', 'premium', 'amount', 'fee', 'deductible', 'limit']):
            return 'financial'
        
        # Temporal queries
        if any(word in query_lower for word in ['when', 'period', 'time', 'duration', 'waiting', 'grace']):
            return 'temporal'
        
        return 'general'
    
    def _get_search_categories(self, query_category: str) -> List[str]:
        """Get categories to search based on query category"""
        if query_category in self.category_relationships:
            categories = [query_category] + self.category_relationships[query_category]
        else:
            categories = ['general']
        
        # Ensure categories exist in indices
        return [cat for cat in categories if cat in self.indices and self.indices[cat].ntotal > 0]
    
    def _calculate_enhanced_score(self, result: Dict[str, Any], query: str, query_category: str) -> float:
        """Calculate enhanced score combining multiple factors"""
        base_score = result['similarity_score']
        metadata = result['metadata']
        
        # Category match bonus
        category_bonus = 0.2 if result['category'] == query_category else 0.0
        
        # Importance score bonus
        importance_bonus = metadata.get('importance_score', 0.5) * 0.15
        
        # Entity density bonus
        entity_bonus = metadata.get('entity_density', 0.0) * 0.1
        
        # Keyword density bonus
        keyword_bonus = metadata.get('keyword_density', 0.0) * 0.05
        
        # Position bonus (earlier content might be more important)
        position_score = 1.0 - metadata.get('position_in_document', 0.5)
        position_bonus = position_score * 0.1
        
        enhanced_score = (
            base_score * 0.5 +  # Base similarity (50%)
            category_bonus +     # Category match (20%)
            importance_bonus +   # Content importance (15%)
            entity_bonus +       # Entity density (10%)
            keyword_bonus +      # Keyword density (5%)
            position_bonus       # Document position (10%)
        )
        
        return min(1.0, enhanced_score)
    
    def get_embedding_dimensions(self) -> int:
        """Get the embedding dimension"""
        return self.embedding_dim if self.embedding_dim else 0
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the indices"""
        stats = {}
        for category, index in self.indices.items():
            stats[category] = {
                'total_vectors': index.ntotal,
                'dimension': self.embedding_dim,
                'texts_count': len(self.texts.get(category, [])),
                'metadata_count': len(self.chunk_metadata.get(category, []))
            }
        return stats
    
    async def save_enhanced_index(self, filepath: str):
        """Save enhanced indices and metadata"""
        try:
            base_path = Path(filepath)
            base_path.mkdir(parents=True, exist_ok=True)
            
            # Save each index separately
            for category, index in self.indices.items():
                if index.ntotal > 0:
                    index_path = base_path / f"{category}.index"
                    faiss.write_index(index, str(index_path))
            
            # Save metadata
            metadata_path = base_path / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'texts': self.texts,
                    'chunk_metadata': self.chunk_metadata,
                    'embedding_dim': self.embedding_dim,
                    'category_relationships': getattr(self, 'category_relationships', {}),
                    'chunk_configs': self.chunk_configs
                }, f)
            
            logger.info(f"Enhanced indices saved to {base_path}")
            
        except Exception as e:
            logger.error(f"Error saving enhanced indices: {str(e)}")
            raise
    
    async def load_enhanced_index(self, filepath: str):
        """Load enhanced indices and metadata"""
        try:
            base_path = Path(filepath)
            
            # Load metadata first
            metadata_path = base_path / "metadata.pkl"
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.texts = data['texts']
                self.chunk_metadata = data['chunk_metadata']
                self.embedding_dim = data['embedding_dim']
                self.category_relationships = data.get('category_relationships', {})
                self.chunk_configs = data.get('chunk_configs', self.chunk_configs)
            
            # Load indices
            self.indices = {}
            for category in self.texts.keys():
                index_path = base_path / f"{category}.index"
                if index_path.exists():
                    self.indices[category] = faiss.read_index(str(index_path))
                else:
                    self.indices[category] = faiss.IndexFlatIP(self.embedding_dim)
            
            logger.info(f"Enhanced indices loaded from {base_path}")
            
        except Exception as e:
            logger.error(f"Error loading enhanced indices: {str(e)}")
            raise