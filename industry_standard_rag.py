"""
INDUSTRY-STANDARD RAG SYSTEM WITH RE-RANKING
Hard reset implementation to fix the 2.0/10 catastrophic retrieval failure
"""

import os
import re
import logging
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Optional ML imports with fallback
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("⚠️ Advanced models not available - using fallback implementation")

@dataclass
class Document:
    """Simple document representation"""
    content: str
    metadata: Dict = None

@dataclass 
class Chunk:
    """Document chunk with relevance scoring"""
    content: str
    source: str
    relevance_score: float = 0.0
    rerank_score: float = 0.0

class IndustryStandardRetriever:
    """Industry-standard two-stage retrieval: Initial + Re-ranking"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.embedding_model = None
        self.reranker = None
        self.setup_models()
    
    def setup_models(self):
        """Setup industry-standard models"""
        if MODELS_AVAILABLE:
            try:
                # Industry standard embedding model
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                # Industry standard cross-encoder for re-ranking
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                self.logger.info("✅ Industry-standard models loaded successfully")
            except Exception as e:
                self.logger.error(f"Model loading failed: {e}")
                MODELS_AVAILABLE = False
        
        if not MODELS_AVAILABLE:
            self.logger.info("🔄 Using keyword-based fallback system")
    
    def retrieve(self, query: str, chunks: List[Chunk], top_k: int = 3) -> List[Chunk]:
        """Two-stage retrieval: Initial broad search + Precision re-ranking"""
        if not chunks:
            return []
        
        self.logger.info(f"🔍 Industry-standard retrieval for: '{query[:50]}...'")
        
        # STAGE 1: Initial broad retrieval (cast wide net)
        initial_candidates = self._initial_retrieval(query, chunks, top_k=min(20, len(chunks)))
        self.logger.info(f"📊 Stage 1: {len(initial_candidates)} initial candidates")
        
        # STAGE 2: Precision re-ranking (focus to most relevant)
        final_results = self._precision_rerank(query, initial_candidates, top_k=top_k)
        self.logger.info(f"🎯 Stage 2: {len(final_results)} final results after re-ranking")
        
        return final_results
    
    def _initial_retrieval(self, query: str, chunks: List[Chunk], top_k: int) -> List[Chunk]:
        """Stage 1: Cast wide net with semantic + keyword search"""
        all_candidates = []
        
        # Semantic search if available
        if self.embedding_model:
            semantic_results = self._semantic_search(query, chunks, top_k)
            all_candidates.extend(semantic_results)
            self.logger.info(f"   📈 Semantic: {len(semantic_results)} candidates")
        
        # Keyword search
        keyword_results = self._keyword_search(query, chunks, top_k)
        all_candidates.extend(keyword_results)
        self.logger.info(f"   🔤 Keyword: {len(keyword_results)} candidates")
        
        # Remove duplicates
        unique_candidates = self._deduplicate(all_candidates)
        self.logger.info(f"   🔄 Deduplicated: {len(all_candidates)} → {len(unique_candidates)}")
        
        return unique_candidates[:top_k]
    
    def _semantic_search(self, query: str, chunks: List[Chunk], top_k: int) -> List[Chunk]:
        """Semantic similarity search using embeddings"""
        try:
            # Get embeddings
            query_embedding = self.embedding_model.encode([query])
            chunk_texts = [chunk.content for chunk in chunks]
            chunk_embeddings = self.embedding_model.encode(chunk_texts)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            
            # Rank by similarity
            ranked_indices = np.argsort(similarities)[::-1]
            
            # Return top chunks with scores
            results = []
            for i in ranked_indices[:top_k]:
                chunk = chunks[i]
                chunk.relevance_score = float(similarities[i])
                results.append(chunk)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    def _keyword_search(self, query: str, chunks: List[Chunk], top_k: int) -> List[Chunk]:
        """Enhanced keyword-based search"""
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        
        scored_chunks = []
        for chunk in chunks:
            content_words = set(re.findall(r'\b\w{3,}\b', chunk.content.lower()))
            
            # Basic word overlap
            overlap = len(query_words.intersection(content_words))
            
            # Boost for numbers and specific terms
            score = overlap
            if re.search(r'\d+', chunk.content):
                score += 1
            if any(term in chunk.content.lower() for term in ['rs.', 'inr', '%', 'maximum', 'minimum']):
                score += 0.5
            
            if score > 0:
                chunk.relevance_score = score
                scored_chunks.append(chunk)
        
        # Sort by score
        scored_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored_chunks[:top_k]
    
    def _precision_rerank(self, query: str, candidates: List[Chunk], top_k: int) -> List[Chunk]:
        """Stage 2: Precision re-ranking using cross-encoder"""
        if not candidates:
            return []
        
        if self.reranker:
            try:
                # Prepare query-chunk pairs
                pairs = [[query, chunk.content] for chunk in candidates]
                
                # Get precision scores
                scores = self.reranker.predict(pairs)
                
                # Apply scores and sort
                for i, chunk in enumerate(candidates):
                    chunk.rerank_score = float(scores[i])
                
                # Sort by rerank score
                candidates.sort(key=lambda x: x.rerank_score, reverse=True)
                
                self.logger.info("🎯 Cross-encoder re-ranking completed")
                
            except Exception as e:
                self.logger.error(f"Re-ranking failed: {e}")
        else:
            # Fallback re-ranking
            for chunk in candidates:
                chunk.rerank_score = chunk.relevance_score
        
        return candidates[:top_k]
    
    def _deduplicate(self, chunks: List[Chunk]) -> List[Chunk]:
        """Remove duplicate chunks"""
        seen = set()
        unique = []
        
        for chunk in chunks:
            # Normalize content for comparison
            normalized = re.sub(r'\s+', ' ', chunk.content.lower().strip())
            
            if normalized not in seen:
                seen.add(normalized)
                unique.append(chunk)
        
        return unique

class SimpleChunker:
    """Simple, reliable document chunker"""
    
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = logging.getLogger(__name__)
    
    def chunk_document(self, text: str) -> List[Chunk]:
        """Create overlapping chunks from document"""
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        chunks = []
        current_chunk = ""
        current_words = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence exceeds chunk size, create chunk
            if current_words + sentence_words > self.chunk_size and current_chunk:
                chunk = Chunk(
                    content=current_chunk.strip(),
                    source="document"
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_words = current_chunk.split()[-self.overlap:]
                current_chunk = " ".join(overlap_words) + " " + sentence + ". "
                current_words = len(current_chunk.split())
            else:
                current_chunk += sentence + ". "
                current_words += sentence_words
        
        # Add final chunk
        if current_chunk.strip():
            chunk = Chunk(
                content=current_chunk.strip(),
                source="document"
            )
            chunks.append(chunk)
        
        self.logger.info(f"✅ Created {len(chunks)} document chunks")
        return chunks

class PreciseAnswerGenerator:
    """Generate precise answers from ranked chunks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_answer(self, query: str, chunks: List[Chunk]) -> str:
        """Generate precise answer from top-ranked chunks"""
        if not chunks:
            return "The requested information is not available in the document."
        
        # Combine top chunks
        context = self._combine_chunks(chunks)
        
        # Extract precise answer
        answer = self._extract_precise_answer(query, context)
        
        return answer
    
    def _combine_chunks(self, chunks: List[Chunk]) -> str:
        """Combine chunks into coherent context"""
        combined = []
        for i, chunk in enumerate(chunks):
            # Include rank information for transparency
            combined.append(f"[Rank {i+1}, Score: {chunk.rerank_score:.3f}] {chunk.content}")
        
        return "\n\n".join(combined)
    
    def _extract_precise_answer(self, query: str, context: str) -> str:
        """Extract most relevant answer from context"""
        query_lower = query.lower()
        
        # Insurance-specific patterns
        patterns = {
            'waiting period': r'(\d+)\s*months?\s*(?:waiting|period)',
            'percentage': r'(\d+)%\s*(?:of\s*)?(?:sum\s*insured)?',
            'amount': r'rs\.?\s*([0-9,]+)',
            'days': r'(\d+)\s*days?',
            'hours': r'(\d+)\s*hours?'
        }
        
        # Try pattern matching first
        for pattern_name, pattern in patterns.items():
            if any(word in query_lower for word in pattern_name.split()):
                matches = re.findall(pattern, context, re.IGNORECASE)
                if matches:
                    # Return first match in a natural sentence
                    value = matches[0]
                    if pattern_name == 'waiting period':
                        return f"The waiting period is {value} months."
                    elif pattern_name == 'percentage':
                        return f"The coverage is {value}% of sum insured."
                    elif pattern_name == 'amount':
                        return f"The amount is Rs. {value}."
                    elif pattern_name == 'days':
                        return f"The time period is {value} days."
                    elif pattern_name == 'hours':
                        return f"The time limit is {value} hours."
        
        # Fallback: return most relevant sentence
        sentences = re.split(r'[.!?]+', context)
        query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
        
        best_sentence = None
        best_score = 0
        
        for sentence in sentences:
            if len(sentence.strip()) < 20:
                continue
            
            sentence_words = set(re.findall(r'\b\w{3,}\b', sentence.lower()))
            overlap = len(query_words.intersection(sentence_words))
            
            # Boost for specific information
            if re.search(r'\d+', sentence):
                overlap += 1
            if any(term in sentence.lower() for term in ['rs.', 'maximum', 'minimum', '%']):
                overlap += 0.5
            
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence.strip()
        
        if best_sentence and best_score >= 1:
            return best_sentence
        
        return "The specific information requested is not clearly available in the document content."

class IndustryStandardRAG:
    """Complete industry-standard RAG system"""
    
    def __init__(self):
        self.chunker = SimpleChunker()
        self.retriever = IndustryStandardRetriever()
        self.generator = PreciseAnswerGenerator()
        self.document_cache = {}
        self.logger = logging.getLogger(__name__)
    
    def process_document_and_answer(self, document_text: str, questions: List[str]) -> List[str]:
        """Process document and answer questions using industry-standard RAG"""
        # Create document hash for caching
        doc_hash = hashlib.md5(document_text.encode()).hexdigest()
        
        # Get or create chunks
        if doc_hash in self.document_cache:
            chunks = self.document_cache[doc_hash]
            self.logger.info("📋 Using cached document chunks")
        else:
            self.logger.info("📄 Creating document chunks...")
            chunks = self.chunker.chunk_document(document_text)
            self.document_cache[doc_hash] = chunks
            self.logger.info(f"✅ Cached {len(chunks)} chunks")
        
        # Answer each question
        answers = []
        for i, question in enumerate(questions):
            self.logger.info(f"🤔 [{i+1}/{len(questions)}] Processing: {question[:50]}...")
            
            # Retrieve relevant chunks
            relevant_chunks = self.retriever.retrieve(question, chunks, top_k=3)
            
            # Generate answer
            answer = self.generator.generate_answer(question, relevant_chunks)
            answers.append(answer)
            
            # Log quality indicator
            if len(answer) > 50 and "not available" not in answer.lower():
                self.logger.info(f"✅ [{i+1}] Good answer generated")
            else:
                self.logger.warning(f"⚠️ [{i+1}] Short/fallback answer")
        
        self.logger.info(f"🎉 Completed processing {len(questions)} questions")
        return answers
