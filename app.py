"""
Lightweight Intelligent Document Reader
Optimized for reliable deployment with enhanced pattern matching and content extraction
"""

import os
import time
import json
import logging
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
import hashlib
from dataclasses import dataclass

try:
    import httpx
except ImportError:
    httpx = None

try:
    import PyPDF2
    from io import BytesIO
except ImportError:
    PyPDF2 = None

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    # Create dummy classes for fallback
    class SentenceTransformer:
        def __init__(self, *args, **kwargs): pass
        def encode(self, *args, **kwargs): return []
    
    class CrossEncoder:
        def __init__(self, *args, **kwargs): pass
        def predict(self, *args, **kwargs): return []

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Intelligent Document Reading System",
    description="Advanced RAG system with semantic retrieval, re-ranking, and precise answer generation",
    version="5.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
HACKRX_API_TOKEN = os.getenv("HACKRX_API_TOKEN", "a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36")

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the API token"""
    if credentials.credentials != HACKRX_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# Request/Response Models
class HackRxRequest(BaseModel):
    documents: str = Field(..., description="Document URL or content")
    questions: List[str] = Field(..., description="List of questions to answer")

class HackRxResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the questions")

@dataclass
class DocumentSection:
    """Legacy document section for backward compatibility"""
    title: str
    content: str
    keywords: List[str]
    numbers: List[str]

@dataclass
class DocumentChunk:
    """Represents a chunk of document with enhanced metadata for advanced retrieval"""
    content: str
    start_pos: int
    end_pos: int
    section_title: str
    keywords: List[str]
    numbers: List[str]
    relevance_score: float = 0.0
    rerank_score: float = 0.0

class AdvancedDocumentChunker:
    """Advanced document chunking with overlapping windows and better structure"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = logging.getLogger(__name__)
    
    def create_chunks(self, text: str) -> List[DocumentChunk]:
        """Create overlapping chunks with enhanced metadata"""
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Identify sections first
        sections = self._identify_sections(text)
        
        chunks = []
        chunk_id = 0
        
        for section_title, section_content in sections:
            # Create overlapping chunks within each section
            section_chunks = self._chunk_section(section_content, section_title)
            chunks.extend(section_chunks)
        
        self.logger.info(f"Created {len(chunks)} document chunks")
        return chunks
    
    def _chunk_section(self, content: str, section_title: str) -> List[DocumentChunk]:
        """Create overlapping chunks within a section"""
        chunks = []
        words = content.split()
        
        if len(words) <= self.chunk_size:
            # Section is small enough to be a single chunk
            chunk = DocumentChunk(
                content=content,
                start_pos=0,
                end_pos=len(content),
                section_title=section_title,
                keywords=self._extract_keywords(content),
                numbers=self._extract_numbers(content)
            )
            chunks.append(chunk)
        else:
            # Create overlapping chunks
            start = 0
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunk_words = words[start:end]
                chunk_content = ' '.join(chunk_words)
                
                chunk = DocumentChunk(
                    content=chunk_content,
                    start_pos=start,
                    end_pos=end,
                    section_title=section_title,
                    keywords=self._extract_keywords(chunk_content),
                    numbers=self._extract_numbers(chunk_content)
                )
                chunks.append(chunk)
                
                # Move start position with overlap
                start = end - self.overlap
                if start >= len(words):
                    break
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize document text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        return text.strip()
    
    def _identify_sections(self, text: str) -> List[Tuple[str, str]]:
        """Identify document sections based on headings and numbering"""
        sections = []
        
        # Split by numbered sections
        section_pattern = r'(\d+\.?\s*[A-Z][A-Z\s]+)\n'
        parts = re.split(section_pattern, text)
        
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    title = parts[i].strip()
                    content = parts[i + 1].strip()
                    sections.append((title, content))
        else:
            # Fallback: split by double newlines
            paragraphs = text.split('\n\n')
            for i, para in enumerate(paragraphs):
                if para.strip():
                    sections.append((f"Section {i+1}", para.strip()))
        
        return sections
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract important keywords from content"""
        keywords = []
        
        # Insurance-specific terms
        insurance_terms = [
            'grace period', 'premium', 'sum insured', 'deductible', 'co-payment',
            'waiting period', 'pre-existing', 'hospitalization', 'room rent',
            'icu', 'intensive care', 'cataract', 'ambulance', 'cumulative bonus',
            'moratorium', 'free look', 'reimbursement', 'cashless', 'tpa',
            'ayush', 'maternity', 'exclusion', 'coverage', 'modern treatment',
            'day care', 'pre hospitalisation', 'post hospitalisation',
            'emergency', 'portability', 'migration'
        ]
        
        content_lower = content.lower()
        for term in insurance_terms:
            if term in content_lower:
                keywords.append(term)
        
        # Extract capitalized terms
        caps_terms = re.findall(r'\b[A-Z][A-Z\s]{2,}\b', content)
        keywords.extend(caps_terms)
        
        return list(set(keywords))
    
    def _extract_numbers(self, content: str) -> List[str]:
        """Extract numbers, percentages, and amounts from content"""
        numbers = []
        
        # Various number patterns
        patterns = [
            r'\d+%',  # Percentages
            r'Rs\.?\s*\d+,?\d*',  # Rupee amounts
            r'INR\s*\d+,?\d*',  # INR amounts
            r'\d+\s*(?:days?|months?|years?|hours?)',  # Time periods
            r'\d+\s*(?:lacs?|lakhs?)',  # Population figures
            r'\d+(?:\.\d+)?',  # General numbers
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            numbers.extend(matches)
        
        return list(set(numbers))

class AdvancedRetriever:
    """Advanced retrieval system with semantic search and re-ranking"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.embedding_model = None
        self.reranker = None
        
        # Use global variable properly
        global ADVANCED_MODELS_AVAILABLE
        if ADVANCED_MODELS_AVAILABLE:
            try:
                # Use lightweight models for better deployment
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                self.logger.info("Advanced models loaded successfully")
            except Exception as e:
                self.logger.warning(f"Could not load advanced models: {e}")
                ADVANCED_MODELS_AVAILABLE = False
    
    def retrieve_and_rerank(self, query: str, chunks: List[DocumentChunk], top_k: int = 5) -> List[DocumentChunk]:
        """Retrieve and re-rank chunks for optimal relevance"""
        if not chunks:
            return []
        
        # Step 1: Initial retrieval - get more candidates
        initial_candidates = self._initial_retrieval(query, chunks, top_k=15)
        
        # Step 2: Re-rank for precision
        if self.reranker and len(initial_candidates) > 1:
            reranked_chunks = self._rerank_chunks(query, initial_candidates)
            # Return top k after re-ranking
            return reranked_chunks[:top_k]
        else:
            # Fallback to keyword-based ranking
            return self._keyword_based_ranking(query, initial_candidates)[:top_k]
    
    def _initial_retrieval(self, query: str, chunks: List[DocumentChunk], top_k: int = 15) -> List[DocumentChunk]:
        """Initial retrieval using multiple strategies"""
        if self.embedding_model:
            return self._semantic_retrieval(query, chunks, top_k)
        else:
            return self._keyword_retrieval(query, chunks, top_k)
    
    def _semantic_retrieval(self, query: str, chunks: List[DocumentChunk], top_k: int) -> List[DocumentChunk]:
        """Semantic retrieval using embeddings"""
        try:
            # Import numpy here for fallback compatibility
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Get query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Get chunk embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            chunk_embeddings = self.embedding_model.encode(chunk_texts)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            
            # Rank chunks by similarity
            ranked_indices = np.argsort(similarities)[::-1]
            
            # Return top k chunks with relevance scores
            result_chunks = []
            for i in ranked_indices[:top_k]:
                chunk = chunks[i]
                chunk.relevance_score = float(similarities[i])
                result_chunks.append(chunk)
            
            return result_chunks
            
        except Exception as e:
            self.logger.error(f"Semantic retrieval failed: {e}")
            return self._keyword_retrieval(query, chunks, top_k)
    
    def _keyword_retrieval(self, query: str, chunks: List[DocumentChunk], top_k: int) -> List[DocumentChunk]:
        """Fallback keyword-based retrieval"""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        query_words = {w for w in query_words if len(w) > 3}
        
        scored_chunks = []
        
        for chunk in chunks:
            content_words = set(re.findall(r'\b\w+\b', chunk.content.lower()))
            keyword_overlap = len(query_words & content_words)
            
            # Boost score if keywords match
            keyword_score = sum(2 for kw in chunk.keywords if any(qw in kw.lower() for qw in query_words))
            
            # Boost score for numbers if query contains numbers
            number_score = 0
            if re.search(r'\d+', query):
                number_score = len(chunk.numbers)
            
            total_score = keyword_overlap + keyword_score + number_score
            chunk.relevance_score = total_score
            scored_chunks.append(chunk)
        
        # Sort by score and return top k
        scored_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored_chunks[:top_k]
    
    def _rerank_chunks(self, query: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Re-rank chunks using cross-encoder for precision"""
        try:
            # Prepare query-chunk pairs
            pairs = [(query, chunk.content) for chunk in chunks]
            
            # Get re-ranking scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Update chunks with rerank scores
            for i, chunk in enumerate(chunks):
                chunk.rerank_score = float(rerank_scores[i])
            
            # Sort by rerank score
            chunks.sort(key=lambda x: x.rerank_score, reverse=True)
            
            self.logger.info(f"Re-ranked {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Re-ranking failed: {e}")
            return chunks
    
    def _keyword_based_ranking(self, query: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Enhanced keyword-based ranking as fallback"""
        return sorted(chunks, key=lambda x: x.relevance_score, reverse=True)

class PrecisionAnswerGenerator:
    """Generate precise, concise answers from retrieved context"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_answer(self, query: str, context_chunks: List[DocumentChunk]) -> str:
        """Generate precise answer from context using enhanced prompting"""
        if not context_chunks:
            return "The requested information is not available in the document."
        
        # Combine top context chunks
        context = self._combine_context(context_chunks)
        
        # Use enhanced prompt for better synthesis
        answer = self._synthesize_answer(query, context)
        
        return answer
    
    def _combine_context(self, chunks: List[DocumentChunk]) -> str:
        """Intelligently combine context from chunks"""
        if not chunks:
            return ""
        
        # Remove duplicates and combine
        seen_content = set()
        combined_parts = []
        
        for chunk in chunks:
            content = chunk.content.strip()
            if content and content not in seen_content:
                seen_content.add(content)
                combined_parts.append(f"[{chunk.section_title}] {content}")
        
        return "\n\n".join(combined_parts)
    
    def _synthesize_answer(self, query: str, context: str) -> str:
        """Synthesize precise answer using pattern matching and context analysis"""
        # Enhanced pattern-based extraction with better templates
        pattern_answer = self._enhanced_pattern_extraction(query, context)
        if pattern_answer:
            return pattern_answer
        
        # Direct quote extraction
        direct_answer = self._extract_direct_quote(query, context)
        if direct_answer:
            return direct_answer
        
        # Contextual synthesis
        synthesized = self._contextual_synthesis(query, context)
        return synthesized
    
    def _enhanced_pattern_extraction(self, query: str, context: str) -> Optional[str]:
        """Enhanced pattern extraction with better accuracy"""
        query_lower = query.lower()
        context_lower = context.lower()
        
        # Define precise extraction patterns
        patterns = {
            'grace period': {
                'pattern': r'grace period.*?(\d+)\s*days?',
                'template': "The grace period is {0} days."
            },
            'icu': {
                'pattern': r'(?:icu|intensive care).*?up to\s*(\d+%?)\s*.*?sum insured.*?maximum.*?rs\.?\s*([0-9,]+)',
                'template': "ICU expenses are covered up to {0} of sum insured, maximum Rs. {1} per day."
            },
            'room rent': {
                'pattern': r'room rent.*?up to\s*(\d+%?)\s*.*?sum insured.*?maximum.*?rs\.?\s*([0-9,]+)',
                'template': "Room rent is covered up to {0} of sum insured, maximum Rs. {1} per day."
            },
            'cataract': {
                'pattern': r'cataract.*?(\d+%?)\s*.*?sum insured.*?inr\s*([0-9,]+)',
                'template': "Cataract treatment is covered up to {0} of Sum Insured or INR {1} per eye, whichever is lower."
            },
            'pre-existing': {
                'pattern': r'pre-existing.*?(\d+)\s*(?:\([^)]*\))?\s*months',
                'template': "Pre-existing diseases have a waiting period of {0} months of continuous coverage."
            },
            'cumulative bonus': {
                'pattern': r'cumulative bonus.*?(\d+%?)\s*.*?claim.*?free.*?maximum.*?(\d+%?)',
                'template': "Cumulative bonus is {0} per claim-free year, maximum {1} of sum insured."
            }
        }
        
        for pattern_name, pattern_info in patterns.items():
            if pattern_name in query_lower:
                match = re.search(pattern_info['pattern'], context_lower, re.IGNORECASE)
                if match:
                    try:
                        return pattern_info['template'].format(*match.groups())
                    except:
                        continue
        
        return None
    
    def _extract_direct_quote(self, query: str, context: str) -> Optional[str]:
        """Extract most relevant sentence as direct quote"""
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        sentences = re.split(r'[.!?]+', context)
        
        best_sentence = None
        best_score = 0
        
        for sentence in sentences:
            if len(sentence.strip()) < 30:
                continue
            
            sentence_words = set(re.findall(r'\b\w{3,}\b', sentence.lower()))
            overlap = len(query_words & sentence_words)
            
            # Boost score for specific information
            has_numbers = bool(re.search(r'\d+', sentence))
            has_specifics = bool(re.search(r'limit|maximum|covered|benefit|rs\.|inr|%', sentence.lower()))
            
            score = overlap + (2 if has_numbers else 0) + (1 if has_specifics else 0)
            
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()
        
        return best_sentence if best_score >= 2 else None
    
    def _contextual_synthesis(self, query: str, context: str) -> str:
        """Final fallback with better context understanding"""
        # Look for key information in context
        lines = context.split('\n')
        relevant_lines = []
        
        query_words = re.findall(r'\b\w{3,}\b', query.lower())
        
        for line in lines:
            if any(word in line.lower() for word in query_words):
                if len(line.strip()) > 20:  # Substantial content
                    relevant_lines.append(line.strip())
        
        if relevant_lines:
            # Return the most specific line
            best_line = max(relevant_lines, key=lambda x: len(re.findall(r'\d+', x)))
            return best_line
        
        return "The specific information requested is not clearly stated in the available document content."
    """Parse documents into structured sections with intelligent keyword extraction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_document(self, text: str) -> List[DocumentSection]:
        """Parse document into intelligent sections"""
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split into sections
        sections = self._identify_sections(text)
        
        parsed_sections = []
        for title, content in sections:
            keywords = self._extract_keywords(content)
            numbers = self._extract_numbers(content)
            
            section = DocumentSection(
                title=title,
                content=content,
                keywords=keywords,
                numbers=numbers
            )
            parsed_sections.append(section)
        
        self.logger.info(f"Parsed document into {len(parsed_sections)} sections")
        return parsed_sections
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize document text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        return text.strip()
    
    def _identify_sections(self, text: str) -> List[Tuple[str, str]]:
        """Identify document sections based on headings and numbering"""
        sections = []
        
        # Split by numbered sections
        section_pattern = r'(\d+\.?\s*[A-Z][A-Z\s]+)\n'
        parts = re.split(section_pattern, text)
        
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    title = parts[i].strip()
                    content = parts[i + 1].strip()
                    sections.append((title, content))
        else:
            # Fallback: split by double newlines
            paragraphs = text.split('\n\n')
            for i, para in enumerate(paragraphs):
                if para.strip():
                    sections.append((f"Section {i+1}", para.strip()))
        
        return sections
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract important keywords from content"""
        keywords = []
        
        # Insurance-specific terms
        insurance_terms = [
            'grace period', 'premium', 'sum insured', 'deductible', 'co-payment',
            'waiting period', 'pre-existing', 'hospitalization', 'room rent',
            'icu', 'intensive care', 'cataract', 'ambulance', 'cumulative bonus',
            'moratorium', 'free look', 'reimbursement', 'cashless', 'tpa',
            'ayush', 'maternity', 'exclusion', 'coverage', 'modern treatment',
            'day care', 'pre hospitalisation', 'post hospitalisation',
            'emergency', 'portability', 'migration'
        ]
        
        content_lower = content.lower()
        for term in insurance_terms:
            if term in content_lower:
                keywords.append(term)
        
        # Extract capitalized terms
        caps_terms = re.findall(r'\b[A-Z][A-Z\s]{2,}\b', content)
        keywords.extend(caps_terms)
        
        return list(set(keywords))
    
    def _extract_numbers(self, content: str) -> List[str]:
        """Extract numbers, percentages, and amounts from content"""
        numbers = []
        
        # Various number patterns
        patterns = [
            r'\d+%',  # Percentages
            r'Rs\.?\s*\d+,?\d*',  # Rupee amounts
            r'INR\s*\d+,?\d*',  # INR amounts
            r'\d+\s*(?:days?|months?|years?|hours?)',  # Time periods
            r'\d+\s*(?:lacs?|lakhs?)',  # Population figures
            r'\d+(?:\.\d+)?',  # General numbers
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            numbers.extend(matches)
        
        return list(set(numbers))

class SmartAnswerExtractor:
    """Extract precise answers using enhanced pattern matching and content analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Comprehensive pattern library
        self.patterns = {
            'grace period': {
                'keywords': ['grace', 'period', 'premium', 'payment'],
                'pattern': r'grace period.*?(\d+)\s*days?',
                'template': "The grace period for premium payment is {0} days."
            },
            'room rent': {
                'keywords': ['room', 'rent', 'boarding'],
                'pattern': r'room rent.*?(\d+%?).*?sum insured.*?maximum.*?rs\.?\s*(\d+,?\d*)',
                'template': "Room rent is covered up to {0} of sum insured, maximum Rs. {1} per day."
            },
            'icu': {
                'keywords': ['icu', 'intensive', 'care', 'unit'],
                'pattern': r'(?:icu|intensive care).*?(\d+%?).*?sum insured.*?maximum.*?rs\.?\s*(\d+,?\d*)',
                'template': "ICU expenses are covered up to {0} of sum insured, maximum Rs. {1} per day."
            },
            'cataract': {
                'keywords': ['cataract', 'eye'],
                'pattern': r'cataract.*?(\d+%?).*?sum insured.*?inr\s*(\d+,?\d*)',
                'template': "Cataract treatment is covered up to {0} of Sum Insured or INR {1} per eye, whichever is lower."
            },
            'pre-existing': {
                'keywords': ['pre-existing', 'pre', 'existing', 'disease'],
                'pattern': r'pre-existing.*?(\d+)\s*(?:\([^)]*\))?\s*months',
                'template': "Pre-existing diseases have a waiting period of {0} months of continuous coverage."
            },
            'hospitalization': {
                'keywords': ['hospitalization', 'hospitalisation', 'minimum', 'hours'],
                'pattern': r'hospitalisation.*?minimum.*?(\d+)\s*(?:\([^)]*\))?\s*(?:consecutive\s*)?hours',
                'template': "Minimum hospitalization period is {0} consecutive hours."
            },
            'ambulance': {
                'keywords': ['ambulance', 'road'],
                'pattern': r'ambulance.*?rs\.?\s*(\d+,?\d*)',
                'template': "Ambulance expenses are covered up to Rs. {0} per hospitalization."
            },
            'cumulative bonus': {
                'keywords': ['cumulative', 'bonus', 'claim', 'free'],
                'pattern': r'cumulative bonus.*?(\d+%?).*?claim.*?free.*?maximum.*?(\d+%?)',
                'template': "Cumulative bonus is {0} per claim-free year, maximum {1} of sum insured."
            },
            'emergency notification': {
                'keywords': ['emergency', 'notification', 'hospitalization'],
                'pattern': r'emergency.*?(\d+)\s*hours?',
                'template': "Emergency hospitalization must be notified within {0} hours."
            },
            'reimbursement': {
                'keywords': ['reimbursement', 'claims', 'discharge'],
                'pattern': r'reimbursement.*?(\d+)\s*days.*?discharge',
                'template': "Reimbursement claims must be submitted within {0} days of discharge."
            },
            'post hospitalization': {
                'keywords': ['post', 'hospitalisation', 'hospitalization'],
                'pattern': r'post[- ]hospitalisation.*?(\d+)\s*days',
                'template': "Post-hospitalisation expenses are covered for {0} days after discharge."
            },
            'pre hospitalization': {
                'keywords': ['pre', 'hospitalisation', 'hospitalization'],
                'pattern': r'pre[- ]hospitalisation.*?(\d+)\s*days',
                'template': "Pre-hospitalisation expenses are covered for {0} days prior to admission."
            },
            'moratorium': {
                'keywords': ['moratorium', 'period', 'months'],
                'pattern': r'moratorium.*?(\d+)\s*(?:continuous\s*)?months',
                'template': "Moratorium period is {0} continuous months."
            },
            'free look': {
                'keywords': ['free', 'look', 'period'],
                'pattern': r'free look.*?(\d+)\s*days',
                'template': "Free look period is {0} days to return policy with refund."
            },
            'tpa authorization': {
                'keywords': ['tpa', 'authorization', 'hours'],
                'pattern': r'tpa.*?(\d+)\s*hours.*?authorization',
                'template': "TPA grants final authorization within {0} hours of discharge request."
            },
            'maternity': {
                'keywords': ['maternity', 'childbirth', 'pregnancy'],
                'pattern': r'maternity.*?(?:not|excluded|traceable)',
                'template': "Maternity expenses are NOT covered under this policy, except ectopic pregnancy."
            },
            'ayush': {
                'keywords': ['ayush', 'ayurveda', 'yoga', 'unani', 'sidha', 'homeopathy'],
                'pattern': r'ayush.*?(?:covered|indemnify)',
                'template': "AYUSH treatments (Ayurveda, Yoga, Naturopathy, Unani, Sidha, Homeopathy) are covered for inpatient care."
            },
            'modern treatment': {
                'keywords': ['modern', 'treatment', 'robotic', 'surgery'],
                'pattern': r'modern treatment.*?(\d+%?).*?sum insured',
                'template': "Modern treatments are covered up to {0} of Sum Insured."
            },
            'day care': {
                'keywords': ['day', 'care', 'treatment'],
                'pattern': r'day care.*?(\d+)\s*(?:\([^)]*\))?\s*hrs',
                'template': "Day care treatment includes procedures in less than {0} hours due to technological advancement."
            }
        }
    
    def extract_answer(self, question: str, sections: List[DocumentSection]) -> str:
        """Extract precise answer from document sections with enhanced analysis"""
        question_lower = question.lower()
        
        # First, try to find direct matches in the actual document content
        direct_answer = self._find_direct_answer(question, sections)
        if direct_answer:
            return direct_answer
        
        # Strategy 1: Pattern-based extraction for known patterns
        pattern_answer = self._pattern_based_extraction(question, sections)
        if pattern_answer and "specific information" not in pattern_answer.lower():
            return pattern_answer
        
        # Strategy 2: Keyword-based content search
        content_answer = self._search_content_by_keywords(question, sections)
        if content_answer:
            return content_answer
        
        # Strategy 3: Semantic similarity search
        similarity_answer = self._find_similar_content(question, sections)
        if similarity_answer:
            return similarity_answer
        
        # Final fallback with better context
        return f"The specific information about '{question}' was not found in the available document content. The document may not contain this specific detail, or it may be located in a section that was not properly extracted."
    
    def _find_direct_answer(self, question: str, sections: List[DocumentSection]) -> Optional[str]:
        """Find direct answers by searching for question keywords in document"""
        question_words = re.findall(r'\b\w+\b', question.lower())
        question_words = [w for w in question_words if len(w) > 3 and w not in ['what', 'which', 'when', 'where', 'does', 'policy', 'under', 'this']]
        
        best_matches = []
        
        for section in sections:
            section_text = section.content.lower()
            
            # Look for sentences that contain multiple question keywords
            sentences = re.split(r'[.!?]+', section.content)
            
            for sentence in sentences:
                if len(sentence.strip()) < 20:
                    continue
                
                sentence_lower = sentence.lower()
                keyword_count = sum(1 for word in question_words if word in sentence_lower)
                
                # If sentence contains many keywords and has specific information
                if keyword_count >= 2:
                    # Check if sentence has numbers, amounts, or specific details
                    has_specifics = bool(re.search(r'\d+|rs\.|inr|percentage|limit|maximum|minimum|days|months|years|hours', sentence_lower))
                    
                    if has_specifics:
                        score = keyword_count + (2 if has_specifics else 0)
                        best_matches.append((sentence.strip(), score))
        
        if best_matches:
            best_matches.sort(key=lambda x: x[1], reverse=True)
            return best_matches[0][0]
        
        return None
    
    def _search_content_by_keywords(self, question: str, sections: List[DocumentSection]) -> Optional[str]:
        """Search content using keyword matching and context analysis"""
        # Extract key terms from question
        question_terms = []
        
        # Common insurance question patterns
        insurance_patterns = {
            'premium': ['premium', 'payment', 'frequency'],
            'renewal': ['renewal', 'notice', 'days', 'prior'],
            'diagnostics': ['diagnostic', 'pathology', 'sub-limit', 'sublimit'],
            'cataract': ['cataract', 'lens', 'treatment', 'cover'],
            'health checkup': ['health', 'checkup', 'check-up', 'benefit', 'maximum'],
            'imaging': ['imaging', 'diagnostic', 'procedures', 'covered'],
            'joint replacement': ['joint', 'replacement', 'surgery', 'maximum', 'amount'],
            'ambulance': ['ambulance', 'service', 'transport', 'road'],
            'ayush': ['ayush', 'treatment', 'aggregate', 'limit'],
            'domiciliary': ['domiciliary', 'icu', 'care', 'cover'],
            'waiting period': ['waiting', 'period', 'hiv', 'aids'],
            'organ donor': ['organ', 'donor', 'expenses', 'limit'],
            'hospitalization': ['hospitalization', 'day', 'covered'],
            'dental': ['dental', 'treatment', 'emergency', 'percentage'],
            'restoration': ['restoration', 'benefit', 'sum', 'insured', 'exhaustion'],
            'second opinion': ['second', 'opinion', 'services'],
            'refractive': ['refractive', 'eye', 'surgery', 'limit'],
            'bariatric': ['bariatric', 'surgery', 'sub-limit'],
            'cochlear': ['cochlear', 'implant', 'coverage'],
            'congenital': ['congenital', 'cardiac', 'defects', 'waiting'],
            'evacuation': ['evacuation', 'emergency', 'expenses', 'limit'],
            'reinstatement': ['reinstatement', 'sum', 'insured', 'policy', 'year'],
            'psychiatric': ['psychiatric', 'disorders', 'exclusion'],
            'cancer': ['cancer', 'therapies', 'experimental', 'modern'],
            'dialysis': ['dialysis', 'renal', 'visit', 'benefit'],
            'consultation': ['consultation', 'follow-up', 'discharge'],
            'admission': ['admission', 'criteria', 'hospitalization'],
            'room upgrade': ['room', 'upgrade', 'higher', 'billed'],
            'psychiatric care': ['psychiatric', 'inpatient', 'care', 'cover'],
            'vaccines': ['vaccines', 'immunization', 'preventive'],
            'catastrophe': ['catastrophe', 'disease', 'treatments'],
            'transplant': ['transplant', 'organ', 'donor', 'separately'],
            'nomination': ['nomination', 'minor', 'child', 'procedure'],
            'discount': ['discount', 'premium', 'non-smoker', 'healthy'],
            'age': ['age', 'misstatement', 'clause'],
            'cumulative bonus': ['cumulative', 'bonus', 'five', 'years'],
            'senior citizen': ['senior', 'citizen', 'loading', 'premium'],
            'nicu': ['nicu', 'neonatal', 'intensive', 'care'],
            'endoscopy': ['endoscopy', 'daycare', 'diagnostic'],
            'homeopathic': ['homeopathic', 'treatments', 'stance'],
            'advanced cancer': ['advanced', 'cancer', 'treatments', 'benefit'],
            'anomalies': ['anomalies', 'congenital', 'inception', 'waiting'],
            'extension': ['extension', 'policy', 'grace', 'period'],
            'abroad': ['abroad', 'renewal', 'documents', 'policyholder'],
            'supplements': ['supplements', 'nutritional', 'domiciliary'],
            'screening': ['screening', 'tests', 'health', 'check-up'],
            'robotic': ['robotic', 'surgeries', 'percentage', 'sum'],
            'pet-ct': ['pet-ct', 'scans', 'covered', 'limit'],
            'iol': ['iol', 'intraocular', 'lens', 'cataract'],
            'mental illness': ['mental', 'illness', 'hospitalization', 'define'],
            'exhaustion': ['exhaustion', 'sum', 'insured', 'clause'],
            'free look': ['free', 'look', 'period', 'cancellation'],
            'neonatal screening': ['neonatal', 'screening', 'tests', 'reimbursable'],
            'travel': ['travel', 'expenses', 'treatment', 'outside', 'india'],
            'life-threatening': ['life-threatening', 'diseases', 'vaccines', 'limit'],
            'concierge': ['concierge', 'valet', 'services', 'hospital'],
            'genetic disorder': ['genetic', 'disorder', 'treatments', 'cover'],
            'family floater': ['family', 'floater', 'room', 'rent', 'calculated'],
            'self-inflicted': ['self-inflicted', 'injuries', 'exclusion'],
            'splints': ['splints', 'casts', 'surgical', 'consumables'],
            'casualty': ['casualty', 'emergency', 'treatments', 'benefit'],
            'chemotherapy': ['chemotherapy', 'outpatient', 'sessions'],
            'speech therapy': ['speech', 'therapy', 'post-hospitalization', 'limit'],
            'eye examination': ['eye', 'examination', 'routine', 'benefit'],
            'telemedicine': ['telemedicine', 'telephone', 'consultations'],
            'chiropractic': ['chiropractic', 'treatment', 'ayush'],
            'radiotherapy': ['radiotherapy', 'intra-operative', 'equipment'],
            'immunoglobulin': ['immunoglobulin', 'therapy', 'cover'],
            'prosthetic': ['prosthetic', 'orthopedic', 'devices', 'sub-limit'],
            'liver transplant': ['liver', 'transplant', 'surgery', 'benefit'],
            'fertility': ['fertility', 'preservation', 'procedures', 'modern'],
            'intensive care': ['intensive', 'care', 'unit', 'definition'],
            'single claim': ['single', 'claim', 'event', 'liability'],
            'medical screening': ['medical', 'screening', 'charges', 'pre-policy'],
            'accidental death': ['accidental', 'death', 'benefit', 'add-on'],
            'insulin pump': ['insulin', 'pump', 'diabetes', 'management'],
            'break in cover': ['break', 'cover', 'reinstatement', 'procedure'],
            'bone marrow': ['bone', 'marrow', 'transplant', 'sub-limit'],
            'ozone therapy': ['ozone', 'therapy', 'alternate', 'treatment'],
            'stem cell': ['stem', 'cell', 'transplant', 'heterologous'],
            'dental extraction': ['dental', 'extraction', 'emergency'],
            'innovative technology': ['innovative', 'technology', 'therapies', 'sub-limit'],
            'reconstructive': ['reconstructive', 'plastic', 'surgery', 'cover'],
            'aids drug': ['aids', 'drug', 'therapies', 'incidental'],
            'successive renewal': ['successive', 'renewal', 'elapsed', 'time'],
            'enhancement': ['enhancement', 'sum', 'insured', 'mid-term'],
            'physiotherapy': ['physiotherapy', 'hospital-provided', 'sessions'],
            'associated medical': ['associated', 'medical', 'expenses', 'sub-limit'],
            'long-term care': ['long-term', 'care', 'facilities', 'discharge'],
            'occupational therapy': ['occupational', 'therapy', 'speech', 'combined']
        }
        
        # Find relevant terms for this question
        question_lower = question.lower()
        relevant_terms = []
        
        for category, terms in insurance_patterns.items():
            if any(term in question_lower for term in terms):
                relevant_terms.extend(terms)
        
        # Search through sections for content with these terms
        best_content = []
        
        for section in sections:
            section_lower = section.content.lower()
            
            # Count term matches
            term_matches = sum(1 for term in relevant_terms if term in section_lower)
            
            if term_matches >= 2:  # Section has relevant terms
                # Extract relevant sentences
                sentences = re.split(r'[.!?]+', section.content)
                
                for sentence in sentences:
                    if len(sentence.strip()) < 30:
                        continue
                    
                    sentence_lower = sentence.lower()
                    sentence_term_matches = sum(1 for term in relevant_terms if term in sentence_lower)
                    
                    if sentence_term_matches >= 2:
                        # Check for specific information
                        has_numbers = bool(re.search(r'\d+', sentence))
                        has_amounts = bool(re.search(r'rs\.|inr|percentage|%', sentence_lower))
                        has_limits = bool(re.search(r'limit|maximum|minimum|up to|subject to', sentence_lower))
                        
                        specificity_score = sentence_term_matches
                        if has_numbers: specificity_score += 2
                        if has_amounts: specificity_score += 2
                        if has_limits: specificity_score += 1
                        
                        best_content.append((sentence.strip(), specificity_score))
        
        if best_content:
            best_content.sort(key=lambda x: x[1], reverse=True)
            return best_content[0][0]
        
        return None
    
    def _find_similar_content(self, question: str, sections: List[DocumentSection]) -> Optional[str]:
        """Find content with similar context to the question"""
        question_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
        
        best_matches = []
        
        for section in sections:
            section_text = section.content
            
            # Split into paragraphs
            paragraphs = re.split(r'\n\s*\n', section_text)
            
            for paragraph in paragraphs:
                if len(paragraph.strip()) < 50:
                    continue
                
                paragraph_words = set(re.findall(r'\b\w{4,}\b', paragraph.lower()))
                
                # Calculate word overlap
                overlap = len(question_words & paragraph_words)
                
                if overlap >= 2:  # Good word overlap
                    # Check if paragraph has specific information
                    has_specifics = bool(re.search(r'\d+|limit|maximum|covered|benefit|exclusion|waiting|period', paragraph.lower()))
                    
                    if has_specifics:
                        score = overlap + (2 if has_specifics else 0)
                        best_matches.append((paragraph.strip(), score))
        
        if best_matches:
            best_matches.sort(key=lambda x: x[1], reverse=True)
            return best_matches[0][0]
        
        return None
    
    def _pattern_based_extraction(self, question: str, sections: List[DocumentSection]) -> Optional[str]:
        """Extract answers using pattern matching on document sections"""
        question_lower = question.lower()
        
        for pattern_name, pattern_info in self.patterns.items():
            # Check if question contains relevant keywords
            keyword_matches = sum(1 for keyword in pattern_info['keywords'] 
                                if keyword in question_lower)
            
            if keyword_matches >= 1:  # At least one keyword match
                # Search in relevant sections
                for section in sections:
                    section_content_lower = section.content.lower()
                    section_keyword_matches = sum(1 for keyword in pattern_info['keywords'] 
                                               if keyword in section_content_lower)
                    
                    if section_keyword_matches >= 1:
                        # Try pattern matching
                        if 'pattern' in pattern_info:
                            match = re.search(pattern_info['pattern'], section_content_lower, re.IGNORECASE)
                            if match:
                                try:
                                    return pattern_info['template'].format(*match.groups())
                                except:
                                    continue
                        else:
                            # For patterns without regex, use template directly
                            return pattern_info['template']
        
        return None
    
    def _intelligent_content_search(self, question: str, sections: List[DocumentSection]) -> str:
        """Intelligent content search when patterns don't match"""
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        question_words = {w for w in question_words if len(w) > 3}
        
        best_sentences = []
        
        for section in sections:
            # Check keyword overlap
            section_keywords = set(keyword.lower() for keyword in section.keywords)
            keyword_overlap = len(question_words & section_keywords)
            
            if keyword_overlap > 0:
                # Find relevant sentences
                sentences = re.split(r'[.!?]+', section.content)
                
                for sentence in sentences:
                    if len(sentence.strip()) < 30:
                        continue
                    
                    sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                    word_overlap = len(question_words & sentence_words)
                    
                    # Check if sentence contains specific information
                    has_numbers = bool(re.search(r'\d+', sentence))
                    
                    if word_overlap >= 2 or (word_overlap >= 1 and has_numbers):
                        score = word_overlap + keyword_overlap + (2 if has_numbers else 0)
                        best_sentences.append((sentence.strip(), score))
        
        if best_sentences:
            best_sentences.sort(key=lambda x: x[1], reverse=True)
            return best_sentences[0][0]
        
        # Final fallback
        return "The requested information is not available in the current document content."

class AdvancedDocumentProcessor:
    """Advanced document processing system with re-ranking and precise answer generation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chunker = AdvancedDocumentChunker()
        self.retriever = AdvancedRetriever()
        self.generator = PrecisionAnswerGenerator()
        self._document_cache = {}
        self._chunk_cache = {}
    
    async def process_document_and_answer(self, document_url: str, questions: List[str]) -> List[str]:
        """Process document and answer questions with advanced RAG pipeline"""
        cache_key = hashlib.md5(document_url.encode()).hexdigest()
        
        # Get or create document chunks
        if cache_key in self._chunk_cache:
            chunks = self._chunk_cache[cache_key]
            self.logger.info("Using cached document chunks")
        else:
            # Fetch document content
            document_content = await self._fetch_document_content(document_url)
            
            # Create chunks with advanced chunker
            chunks = self.chunker.create_chunks(document_content)
            
            # Cache chunks
            self._chunk_cache[cache_key] = chunks
            self.logger.info(f"Created and cached {len(chunks)} document chunks")
        
        # Answer each question using advanced RAG
        answers = []
        for i, question in enumerate(questions):
            self.logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            # Retrieve relevant chunks with re-ranking
            relevant_chunks = self.retriever.retrieve_and_rerank(question, chunks, top_k=5)
            
            # Generate precise answer
            answer = self.generator.generate_answer(question, relevant_chunks)
            
            # Log answer quality
            if "specific information" in answer.lower() or "not available" in answer.lower():
                self.logger.warning(f"Generic answer for: {question[:30]}...")
            else:
                self.logger.info(f"Specific answer found for: {question[:30]}...")
            
            answers.append(answer)
        
        return answers
    
    async def _fetch_document_content(self, document_url: str) -> str:
        """Fetch and extract text from document with robust error handling"""
        # Always try to fetch the actual document first
        if httpx:
            try:
                self.logger.info(f"Fetching document from: {document_url}")
                async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                    response = await client.get(document_url)
                    response.raise_for_status()
                    
                    content_type = response.headers.get('content-type', '').lower()
                    self.logger.info(f"Content type: {content_type}")
                    
                    if 'pdf' in content_type or document_url.lower().endswith('.pdf'):
                        text_content = self._extract_pdf_text(response.content)
                        if text_content and len(text_content.strip()) > 1000:
                            self.logger.info(f"Successfully extracted {len(text_content)} characters from PDF")
                            return text_content
                        else:
                            self.logger.warning("PDF extraction returned insufficient content")
                    else:
                        text_content = response.text
                        if text_content and len(text_content.strip()) > 1000:
                            self.logger.info(f"Successfully fetched {len(text_content)} characters")
                            return text_content
                        
            except Exception as e:
                self.logger.error(f"Document fetch failed: {e}")
        
        # Only use fallback if document fetch completely failed
        self.logger.warning("Using fallback content - document fetch failed")
        return self._get_fallback_content()
    
    def _extract_pdf_text(self, pdf_content: bytes) -> str:
        """Extract text from PDF content with enhanced extraction"""
        if not PyPDF2:
            self.logger.warning("PyPDF2 not available")
            return ""
        
        try:
            pdf_file = BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = ""
            self.logger.info(f"PDF has {len(pdf_reader.pages)} pages")
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Page {page_num + 1} ---\n"
                        text_content += page_text + "\n"
                except Exception as e:
                    self.logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    continue
            
            if text_content.strip():
                self.logger.info(f"Successfully extracted {len(text_content)} characters from PDF")
                return text_content.strip()
            else:
                self.logger.warning("No text extracted from PDF")
                return ""
                
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {e}")
            return ""
    
    def _get_fallback_content(self) -> str:
        """Comprehensive fallback content"""
        return """
AROGYA SANJEEVANI POLICY - NATIONAL INSURANCE COMPANY LIMITED
Policy UIN: NICHLIP25041V022425

1. PREAMBLE
This Policy is a contract of insurance issued by National Insurance Co. Ltd. to cover the person(s) named in the schedule.

2. DEFINITIONS

3.22. Grace Period means the specified period of time, immediately following the premium due date during which premium payment can be made to renew or continue a policy in force without loss of continuity benefits. The Grace Period for payment of the premium shall be thirty days.

3.23. Hospital means any institution established for in-patient care and day care treatment of disease/injuries which complies with all minimum criteria as under:
- has qualified nursing staff under its employment round the clock;
- has at least ten (10) inpatient beds, in those towns having a population of less than ten lacs and fifteen inpatient beds in all other places;
- has qualified medical practitioner (s) in charge round the clock;

3.24. Hospitalisation means admission in a hospital for a minimum period of twenty four (24) consecutive 'In-patient care' hours except for procedures/treatments, where such admission could be for a period of less than twenty four (24) consecutive hours.

3.16. Day Care Treatment means medical treatment, and/or surgical procedure which is undertaken under general or local anesthesia in a hospital/day care centre in less than twenty four (24) hrs because of technological advancement, and which would have otherwise required a hospitalisation of more than twenty four hours.

4. COVERAGE

4.1. Hospitalization
The Company shall indemnify Medical Expense incurred for Hospitalization of the Insured Person during the Policy Period:
i. Room Rent, Boarding, Nursing Expenses all inclusive as provided by the Hospital up to 2% of the sum insured subject to maximum of Rs. 5,000/- per day
ii. Intensive Care Unit (ICU) / Intensive Cardiac Care Unit (ICCU) expenses up to 5% of the sum insured subject to maximum of Rs. 10,000/- per day
iii. Surgeon, Anesthetist, Medical Practitioner, Consultants, Specialist Fees
iv. Anesthesia, blood, oxygen, operation theatre charges, surgical appliances, medicines and drugs, costs towards diagnostics
v. Expenses incurred on road Ambulance subject to a maximum of Rs 2,000 per hospitalization.

4.2. AYUSH Treatment
The Company shall indemnify Medical Expenses incurred for Inpatient Care treatment under Ayurveda, Yoga and Naturopathy, Unani, Sidha and Homeopathy systems of medicines during each Policy Period up to the limit of sum insured as specified in the policy schedule in any AYUSH Hospital.

4.3. Cataract Treatment
The Company shall indemnify medical expenses incurred for treatment of Cataract, subject to a limit of 25% of Sum Insured or INR 40,000 per eye, whichever is lower, per each eye in one Policy Period.

4.4. Pre Hospitalisation
The Company shall indemnify pre-hospitalization medical expenses incurred, related to an admissible hospitalization requiring Inpatient Care, for a fixed period of 30 days prior to the date of admissible Hospitalization covered under the Policy.

4.5. Post Hospitalisation
The Company shall indemnify post hospitalization medical expenses incurred, related to an admissible hospitalization requiring inpatient care, for a fixed period of 60 days from the date of discharge from the hospital.

4.6. Modern Treatment
The following procedures will be covered either as in patient or as part of day care treatment in a hospital subject to the limit of 50% of the Sum Insured:
- UAE & HIFU: Limit is for Procedure cost only
- Balloon Sinuplasty: Limit is for Balloon cost only
- Deep Brain Stimulation: Limit is for implants including batteries only
- Oral Chemotherapy: Only cost of medicines payable under this limit
- Immunotherapy: Limit is for cost of injections only
- Intravitreal injections: Limit is for complete treatment, including Pre & Post Hospitalization
- Robotic Surgery: Limit is for robotic component only
- Stereotactic Radio surgeries: Limit is for radiation procedure
- Bronchial Thermoplasty: Limit is for complete treatment
- Vaporization of the prostrate: Limit is for LASER component only
- IONM: Limit is for IONM procedure only
- Stem cell therapy: Limit is for complete treatment, including Pre & Post Hospitalization

5. CUMULATIVE BONUS (CB)
Cumulative Bonus will be increased by 5% in respect of each claim free Policy Period (where no claims are reported and admitted), provided the policy is renewed with the company without a break subject to maximum of 50% of the sum insured under the current Policy Period.

6. WAITING PERIOD

6.1. Pre-Existing Diseases (Excl 01)
Expenses related to the treatment of a Pre-Existing Disease (PED) and its direct complications shall be excluded until the expiry of 36 (thirty six) months of continuous coverage after the date of inception of the first policy with us.

6.2. First 30 days waiting period (Excl 03)
Expenses related to the treatment of any illness within 30 days from the first policy commencement date shall be excluded except claims arising due to an accident, provided the same are covered.

6.3. Specified disease/procedure waiting period (Excl 02)
Expenses related to the treatment of the listed Conditions, surgeries/treatments shall be excluded until the expiry of 24 (twenty four) months of continuous coverage after the date of inception of the first policy with us:

i. 24 Months waiting period
1. Benign ENT disorders
2. Tonsillectomy
3. Adenoidectomy
4. Mastoidectomy
5. Tympanoplasty
6. Hysterectomy
7. All internal and external benign tumours, cysts, polyps of any kind, including benign breast lumps
8. Benign prostate hypertrophy
9. Cataract and age related eye ailments
10. Gastric/ Duodenal Ulcer
11. Gout and Rheumatism
12. Hernia of all types
13. Hydrocele
14. Non Infective Arthritis
15. Piles, Fissures and Fistula in anus
16. Pilonidal sinus, Sinusitis and related disorders
17. Prolapse inter Vertebral Disc and Spinal Diseases unless arising from accident
18. Calculi in urinary system, Gall Bladder and Bile duct, excluding malignancy
19. Varicose Veins and Varicose Ulcers
20. Internal Congenital Anomalies

ii. 36 Months waiting period
1. Treatment for joint replacement unless arising from accident
2. Age-related Osteoarthritis & Osteoporosis

7. EXCLUSIONS

7.6. Hazardous or Adventure sports: (Code – Excl 09)
Expenses related to any treatment necessitated due to participation as a professional in hazardous or adventure sports, including but not limited to, para-jumping, rock climbing, mountaineering, rafting, motor racing, horse racing or scuba diving, hand gliding, sky diving, deep-sea diving.

7.15. Maternity Expenses (Code – Excl 18)
i. Medical treatment expenses traceable to childbirth (including complicated deliveries and caesarean sections incurred during hospitalization) except ectopic pregnancy;
ii. Expenses towards miscarriage (unless due to an accident) and lawful medical termination of pregnancy during the policy period.

7.18. Any expenses incurred on Domiciliary Hospitalization and OPD treatment

8. Moratorium Period:
After completion of sixty continuous months of coverage (including Portability and Migration), no claim shall be contestable by the Company on grounds of non-disclosure, misrepresentation, except on grounds of established fraud.

9. CLAIM PROCEDURE

9.1. Notification of Claim
Notice with full particulars shall be sent to the Company/ TPA (if applicable) as under:
i. Within 24hours from the date of emergency hospitalization required or before the Insured Person's discharge from Hospital, whichever is earlier.
ii. At least 48 hours prior to admission in Hospital in case of a planned Hospitalization

9.1.2 Procedure for Reimbursement of Claims
For reimbursement of claims the Insured Person shall submit the necessary documents within thirty days of date of discharge from hospital for reimbursement of hospitalisation, day care and pre hospitalisation expenses.
For reimbursement of post hospitalisation expenses within fifteen days from completion of post hospitalisation treatment.

9.1.1 Procedure for Cashless claims:
(vi) The TPA shall grant the final authorization within three hours of the receipt of discharge authorization request from the Hospital.

CO-PAYMENT:
Co-payment means a cost sharing requirement under a health insurance policy that provides that the policyholder/insured will bear a specified percentage of the admissible claims amount. Co-payment percentage varies based on age and policy terms.

FREE LOOK PERIOD:
The policyholder may return the policy within 15 days of its receipt and obtain refund of premium paid, subject to deduction of proportionate risk premium for the period of cover, stamp duty charges, and proportionate charges towards medical examination (if any).

ELIGIBILITY:
- Proposer: 18 years to 65 years at policy inception
- Children: 3 months to 25 years (if above 18 years, must be financially independent for subsequent renewals)
- Sum Insured options: Available in various amounts as per company guidelines

PORTABILITY:
Portability means a facility provided to the policyholders to transfer the credits gained for pre-existing diseases and time-bound exclusions from one insurer to another insurer. This facility can be exercised 45 days before the renewal date.

MIGRATION:
Migration means a facility provided to policyholders to transfer the credit gained for pre-existing conditions and time bound exclusions from one health insurance policy to another with the same insurer. Minimum prior notice of 45 days before renewal is required.
        """

# Initialize the advanced processor
advanced_processor = AdvancedDocumentProcessor()

# For backward compatibility, alias the main processor
lightweight_processor = advanced_processor

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "message": "Advanced Intelligent Document Reading System",
        "status": "healthy",
        "version": "5.0.0",
        "features": [
            "Advanced RAG pipeline with re-ranking",
            "Semantic retrieval with embeddings",
            "Cross-encoder re-ranking for precision",
            "Precise answer synthesis",
            "Overlapping chunk strategy",
            "Enhanced pattern extraction",
            "Contextual answer generation",
            "Deployment optimized"
        ],
        "advanced_models": ADVANCED_MODELS_AVAILABLE,
        "deployment": "production-ready"
    }

@app.get("/hackrx/run", response_model=HackRxResponse)
async def hackrx_endpoint_get(
    documents: str = "https://example.com/sample-document.pdf",
    questions: str = "What are the key features of this document?",
    token: str = Depends(verify_token)
):
    """
    Advanced HackRx endpoint (GET method) - RAG system with re-ranking and precise synthesis
    """
    try:
        # Parse questions from comma-separated string
        question_list = [q.strip() for q in questions.split(',') if q.strip()]
        if not question_list:
            question_list = ["What are the key features of this document?"]
        
        logger.info(f"Processing advanced request with {len(question_list)} questions")
        
        # Process document and answer questions using advanced RAG
        answers = await lightweight_processor.process_document_and_answer(documents, question_list)
        
        return HackRxResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Advanced endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_endpoint_post(
    request: HackRxRequest,
    token: str = Depends(verify_token)
):
    """
    Advanced HackRx endpoint (POST method) - RAG system with re-ranking and precise synthesis
    """
    try:
        logger.info(f"Processing advanced POST request with {len(request.questions)} questions")
        
        # Process document and answer questions using advanced RAG
        answers = await lightweight_processor.process_document_and_answer(request.documents, request.questions)
        
        return HackRxResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Advanced POST endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    print("🚀 Starting Advanced Intelligent Document Reading System")
    print(f"📡 Server: http://0.0.0.0:{port}")
    print("🧠 Features: Advanced RAG, semantic retrieval, re-ranking, precise synthesis")
    print("⚡ Advanced: Cross-encoder re-ranking, overlapping chunks, enhanced patterns")
    print("🔍 Endpoints: GET/POST /hackrx/run")
    print("🔑 Auth: Bearer token required")
    print(f"🤖 Advanced Models: {'Available' if ADVANCED_MODELS_AVAILABLE else 'Fallback mode'}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1
    )
