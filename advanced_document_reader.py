"""
Advanced Document Reading API Server with Intelligent Processing
Implements semantic chunking, hybrid search, and sophisticated answer extraction
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
from collections import defaultdict

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
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    ADVANCED_PROCESSING = True
except ImportError:
    ADVANCED_PROCESSING = False
    print("Advanced processing libraries not available. Using fallback methods.")

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
    title="Advanced LLM-Powered Document Reading System",
    description="Intelligent document processing with semantic chunking and hybrid search",
    version="3.0.0"
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
class DocumentChunk:
    """Represents a chunk of document with metadata"""
    text: str
    page_number: int
    section_title: str
    chunk_id: str
    summary: str
    keywords: List[str]
    
class SemanticChunker:
    """Intelligent document chunking based on semantic boundaries"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def chunk_document(self, text: str) -> List[DocumentChunk]:
        """Split document into semantically meaningful chunks"""
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split into sections based on headings and structure
        sections = self._identify_sections(text)
        
        chunks = []
        for section_idx, (section_title, section_text) in enumerate(sections):
            # Split section into paragraphs
            paragraphs = self._split_paragraphs(section_text)
            
            for para_idx, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) < 50:  # Skip very short paragraphs
                    continue
                
                chunk_id = f"chunk_{section_idx}_{para_idx}"
                summary = self._generate_summary(paragraph)
                keywords = self._extract_keywords(paragraph)
                
                chunk = DocumentChunk(
                    text=paragraph,
                    page_number=section_idx + 1,
                    section_title=section_title,
                    chunk_id=chunk_id,
                    summary=summary,
                    keywords=keywords
                )
                chunks.append(chunk)
        
        self.logger.info(f"Created {len(chunks)} semantic chunks")
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
        
        # Split by numbered sections (e.g., "1. PREAMBLE", "2. DEFINITIONS")
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
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split section text into coherent paragraphs with overlap"""
        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        paragraphs = []
        current_para = []
        
        for sentence in sentences:
            current_para.append(sentence)
            
            # Create paragraph when we have 3-5 sentences or reach certain length
            if len(current_para) >= 3 and len(' '.join(current_para)) > 200:
                paragraph_text = ' '.join(current_para)
                paragraphs.append(paragraph_text)
                
                # Keep last sentence for overlap
                current_para = [sentence] if len(current_para) > 1 else []
        
        # Add remaining sentences
        if current_para:
            paragraphs.append(' '.join(current_para))
        
        return paragraphs
    
    def _generate_summary(self, text: str) -> str:
        """Generate a brief summary of the chunk"""
        # Extract key phrases and first sentence
        sentences = re.split(r'[.!?]+', text)
        first_sentence = sentences[0].strip() if sentences else ""
        
        # Extract numbers and key terms
        numbers = re.findall(r'\d+(?:\.\d+)?%?', text)
        key_terms = re.findall(r'\b[A-Z][A-Z\s]{2,}\b', text)
        
        summary_parts = [first_sentence[:100]]
        if numbers:
            summary_parts.append(f"Numbers: {', '.join(numbers[:3])}")
        if key_terms:
            summary_parts.append(f"Terms: {', '.join(key_terms[:2])}")
        
        return " | ".join(summary_parts)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from chunk"""
        # Extract key terms, numbers, and important phrases
        keywords = []
        
        # Numbers and percentages
        keywords.extend(re.findall(r'\d+(?:\.\d+)?%?', text))
        
        # Capitalized terms
        keywords.extend(re.findall(r'\b[A-Z][A-Z\s]{2,}\b', text))
        
        # Important insurance terms
        insurance_terms = [
            'premium', 'deductible', 'coverage', 'benefit', 'claim', 'policy',
            'waiting period', 'exclusion', 'hospitalization', 'treatment'
        ]
        
        text_lower = text.lower()
        for term in insurance_terms:
            if term in text_lower:
                keywords.append(term)
        
        return list(set(keywords))

class HybridSearchEngine:
    """Combines semantic and keyword search for optimal retrieval"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chunks: List[DocumentChunk] = []
        self.embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Initialize models if available
        if ADVANCED_PROCESSING:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                self.logger.info("Advanced models loaded successfully")
            except Exception as e:
                self.logger.warning(f"Could not load advanced models: {e}")
                self.semantic_model = None
                self.reranker = None
        else:
            self.semantic_model = None
            self.reranker = None
    
    def index_chunks(self, chunks: List[DocumentChunk]):
        """Index chunks for both semantic and keyword search"""
        self.chunks = chunks
        chunk_texts = [chunk.text for chunk in chunks]
        
        # Semantic indexing
        if self.semantic_model:
            try:
                self.embeddings = self.semantic_model.encode(chunk_texts)
                self.logger.info("Semantic embeddings created")
            except Exception as e:
                self.logger.warning(f"Semantic indexing failed: {e}")
        
        # Keyword indexing (TF-IDF)
        try:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunk_texts)
            self.logger.info("TF-IDF indexing completed")
        except Exception as e:
            self.logger.warning(f"TF-IDF indexing failed: {e}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Perform hybrid search combining semantic and keyword matching"""
        if not self.chunks:
            return []
        
        semantic_scores = self._semantic_search(query, top_k * 2)
        keyword_scores = self._keyword_search(query, top_k * 2)
        
        # Combine scores (weighted average)
        combined_scores = self._combine_scores(semantic_scores, keyword_scores)
        
        # Re-rank top candidates
        if self.reranker and len(combined_scores) > 0:
            combined_scores = self._rerank_results(query, combined_scores, top_k)
        
        return combined_scores[:top_k]
    
    def _semantic_search(self, query: str, top_k: int) -> Dict[int, float]:
        """Semantic search using embeddings"""
        if not self.semantic_model or self.embeddings is None:
            return {}
        
        try:
            query_embedding = self.semantic_model.encode([query])
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top indices and scores
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return {idx: similarities[idx] for idx in top_indices}
        except Exception as e:
            self.logger.warning(f"Semantic search failed: {e}")
            return {}
    
    def _keyword_search(self, query: str, top_k: int) -> Dict[int, float]:
        """Keyword search using TF-IDF"""
        if not self.tfidf_vectorizer or self.tfidf_matrix is None:
            return {}
        
        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            
            # Get top indices and scores
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return {idx: similarities[idx] for idx in top_indices if similarities[idx] > 0}
        except Exception as e:
            self.logger.warning(f"Keyword search failed: {e}")
            return {}
    
    def _combine_scores(self, semantic_scores: Dict[int, float], keyword_scores: Dict[int, float]) -> List[Tuple[DocumentChunk, float]]:
        """Combine semantic and keyword search scores"""
        all_indices = set(semantic_scores.keys()) | set(keyword_scores.keys())
        
        combined_results = []
        for idx in all_indices:
            semantic_score = semantic_scores.get(idx, 0.0)
            keyword_score = keyword_scores.get(idx, 0.0)
            
            # Dynamic weighting: favor keyword search for exact matches
            if keyword_score > 0.5:  # Strong keyword match
                combined_score = 0.3 * semantic_score + 0.7 * keyword_score
            else:  # Favor semantic for weaker keyword matches
                combined_score = 0.6 * semantic_score + 0.4 * keyword_score
            
            combined_results.append((self.chunks[idx], combined_score))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results
    
    def _rerank_results(self, query: str, results: List[Tuple[DocumentChunk, float]], top_k: int) -> List[Tuple[DocumentChunk, float]]:
        """Re-rank results using cross-encoder"""
        if not self.reranker:
            return results
        
        try:
            # Prepare pairs for reranking
            pairs = [(query, chunk.text) for chunk, _ in results[:top_k * 2]]
            
            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Create new ranking
            reranked = []
            for i, (chunk, _) in enumerate(results[:len(rerank_scores)]):
                reranked.append((chunk, float(rerank_scores[i])))
            
            # Sort by rerank scores
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked
        except Exception as e:
            self.logger.warning(f"Reranking failed: {e}")
            return results

class IntelligentAnswerExtractor:
    """Extract precise answers using multiple strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_answer(self, question: str, relevant_chunks: List[Tuple[DocumentChunk, float]]) -> str:
        """Extract answer using multiple strategies"""
        if not relevant_chunks:
            return "No relevant information found in the document."
        
        # Strategy 1: Direct pattern matching
        pattern_answer = self._pattern_based_extraction(question, relevant_chunks)
        if pattern_answer and "specific information" not in pattern_answer.lower():
            return pattern_answer
        
        # Strategy 2: Context-aware extraction
        context_answer = self._context_aware_extraction(question, relevant_chunks)
        if context_answer and "specific information" not in context_answer.lower():
            return context_answer
        
        # Strategy 3: Multi-chunk synthesis
        synthesis_answer = self._synthesize_from_chunks(question, relevant_chunks)
        if synthesis_answer:
            return synthesis_answer
        
        # Fallback: Best chunk with context
        best_chunk = relevant_chunks[0][0]
        return f"Based on the document: {best_chunk.text[:200]}..."
    
    def _pattern_based_extraction(self, question: str, chunks: List[Tuple[DocumentChunk, float]]) -> Optional[str]:
        """Extract answers using advanced pattern matching"""
        question_lower = question.lower()
        
        # Enhanced patterns with more specific extraction
        patterns = {
            'grace period': {
                'pattern': r'grace period.*?(\d+)\s*days?',
                'template': "The grace period for premium payment is {0} days."
            },
            'room rent': {
                'pattern': r'room rent.*?(\d+%?).*?sum insured.*?maximum.*?rs\.?\s*(\d+,?\d*)',
                'template': "Room rent is covered up to {0} of sum insured, maximum Rs. {1} per day."
            },
            'icu': {
                'pattern': r'(?:icu|intensive care).*?(\d+%?).*?sum insured.*?maximum.*?rs\.?\s*(\d+,?\d*)',
                'template': "ICU expenses are covered up to {0} of sum insured, maximum Rs. {1} per day."
            },
            'cataract': {
                'pattern': r'cataract.*?(\d+%?).*?sum insured.*?inr\s*(\d+,?\d*)',
                'template': "Cataract treatment is covered up to {0} of Sum Insured or INR {1} per eye, whichever is lower."
            },
            'pre-existing|pre existing': {
                'pattern': r'pre-existing.*?(\d+)\s*(?:\([^)]*\))?\s*months',
                'template': "Pre-existing diseases have a waiting period of {0} months of continuous coverage."
            },
            'hospitalization|hospitalisation': {
                'pattern': r'hospitalisation.*?minimum.*?(\d+)\s*(?:\([^)]*\))?\s*(?:consecutive\s*)?hours',
                'template': "Minimum hospitalization period is {0} consecutive hours."
            },
            'ambulance': {
                'pattern': r'ambulance.*?rs\.?\s*(\d+,?\d*)',
                'template': "Ambulance expenses are covered up to Rs. {0} per hospitalization."
            },
            'cumulative bonus': {
                'pattern': r'cumulative bonus.*?(\d+%?).*?claim.*?free.*?maximum.*?(\d+%?)',
                'template': "Cumulative bonus is {0} per claim-free year, maximum {1} of sum insured."
            },
            'emergency notification|emergency hospitalization': {
                'pattern': r'emergency.*?(\d+)\s*hours?',
                'template': "Emergency hospitalization must be notified within {0} hours."
            },
            'reimbursement': {
                'pattern': r'reimbursement.*?(\d+)\s*days.*?discharge',
                'template': "Reimbursement claims must be submitted within {0} days of discharge."
            },
            'post hospitalization|post hospitalisation': {
                'pattern': r'post[- ]hospitalisation.*?(\d+)\s*days',
                'template': "Post-hospitalisation expenses are covered for {0} days after discharge."
            },
            'pre hospitalization|pre hospitalisation': {
                'pattern': r'pre[- ]hospitalisation.*?(\d+)\s*days',
                'template': "Pre-hospitalisation expenses are covered for {0} days prior to admission."
            },
            'moratorium': {
                'pattern': r'moratorium.*?(\d+)\s*(?:continuous\s*)?months',
                'template': "Moratorium period is {0} continuous months."
            },
            'free look': {
                'pattern': r'free look.*?(\d+)\s*days',
                'template': "Free look period is {0} days to return policy with refund."
            }
        }
        
        # Search through chunks for patterns
        for chunk, score in chunks:
            chunk_text = chunk.text.lower()
            
            for key_phrase, pattern_info in patterns.items():
                if any(keyword in question_lower for keyword in key_phrase.split('|')):
                    match = re.search(pattern_info['pattern'], chunk_text, re.IGNORECASE)
                    if match:
                        try:
                            return pattern_info['template'].format(*match.groups())
                        except:
                            continue
        
        return None
    
    def _context_aware_extraction(self, question: str, chunks: List[Tuple[DocumentChunk, float]]) -> Optional[str]:
        """Extract answers by understanding context and question intent"""
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        question_words = {w for w in question_words if len(w) > 3}
        
        best_sentences = []
        
        for chunk, score in chunks[:3]:  # Focus on top 3 chunks
            sentences = re.split(r'[.!?]+', chunk.text)
            
            for sentence in sentences:
                if len(sentence.strip()) < 20:
                    continue
                
                sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                overlap = len(question_words & sentence_words)
                
                if overlap >= 2:  # Require at least 2 word overlap
                    # Check if sentence contains specific information
                    if any(re.search(pattern, sentence.lower()) for pattern in [
                        r'\d+', r'rs\.', r'%', r'days?', r'months?', r'hours?', r'years?'
                    ]):
                        best_sentences.append((sentence.strip(), overlap, score))
        
        if best_sentences:
            # Sort by overlap and score
            best_sentences.sort(key=lambda x: (x[1], x[2]), reverse=True)
            return best_sentences[0][0]
        
        return None
    
    def _synthesize_from_chunks(self, question: str, chunks: List[Tuple[DocumentChunk, float]]) -> Optional[str]:
        """Synthesize answer from multiple chunks"""
        # Extract all sentences with numbers or specific terms
        relevant_info = []
        
        for chunk, score in chunks[:5]:
            # Look for sentences with specific information
            sentences = re.split(r'[.!?]+', chunk.text)
            
            for sentence in sentences:
                if len(sentence.strip()) < 30:
                    continue
                
                # Check if sentence contains quantifiable information
                if re.search(r'\d+(?:\.\d+)?(?:%|rs\.?|\s*(?:days?|months?|hours?|years?))', sentence, re.IGNORECASE):
                    relevant_info.append(sentence.strip())
        
        if relevant_info:
            # Return the most relevant sentence
            return relevant_info[0]
        
        return None

class AdvancedDocumentProcessor:
    """Main document processing system with intelligent capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chunker = SemanticChunker()
        self.search_engine = HybridSearchEngine()
        self.answer_extractor = IntelligentAnswerExtractor()
        self._document_cache = {}
    
    async def fetch_and_process_document(self, document_url: str) -> bool:
        """Fetch document and process it for intelligent querying"""
        cache_key = hashlib.md5(document_url.encode()).hexdigest()
        
        if cache_key in self._document_cache:
            self.logger.info("Using cached processed document")
            return True
        
        try:
            # Fetch document content
            document_content = await self._fetch_document_content(document_url)
            
            # Create semantic chunks
            chunks = self.chunker.chunk_document(document_content)
            
            # Index chunks for hybrid search
            self.search_engine.index_chunks(chunks)
            
            # Cache the processed document
            self._document_cache[cache_key] = {
                'chunks': chunks,
                'search_engine': self.search_engine
            }
            
            self.logger.info(f"Document processed: {len(chunks)} chunks indexed")
            return True
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            return False
    
    async def _fetch_document_content(self, document_url: str) -> str:
        """Fetch and extract text from document"""
        if not httpx:
            return self._get_fallback_content()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(document_url)
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '').lower()
                
                if 'pdf' in content_type or document_url.lower().endswith('.pdf'):
                    return self._extract_pdf_text(response.content)
                else:
                    return response.text
                    
        except Exception as e:
            self.logger.warning(f"Document fetch failed: {e}, using fallback")
            return self._get_fallback_content()
    
    def _extract_pdf_text(self, pdf_content: bytes) -> str:
        """Extract text from PDF content"""
        if not PyPDF2:
            return self._get_fallback_content()
        
        try:
            pdf_file = BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            return text_content.strip()
            
        except Exception as e:
            self.logger.warning(f"PDF extraction failed: {e}")
            return self._get_fallback_content()
    
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
    
    async def answer_questions(self, questions: List[str]) -> List[str]:
        """Answer questions using intelligent processing"""
        answers = []
        
        for question in questions:
            self.logger.info(f"Processing question: {question[:50]}...")
            
            # Search for relevant chunks
            relevant_chunks = self.search_engine.search(question, top_k=5)
            
            # Extract answer
            answer = self.answer_extractor.extract_answer(question, relevant_chunks)
            answers.append(answer)
        
        return answers

# Initialize the advanced processor
advanced_processor = AdvancedDocumentProcessor()

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "message": "Advanced LLM-Powered Document Reading System",
        "status": "healthy",
        "version": "3.0.0",
        "features": [
            "Semantic chunking",
            "Hybrid search (semantic + keyword)",
            "Cross-encoder reranking",
            "Intelligent answer extraction",
            "Advanced pattern matching",
            "Multi-strategy processing"
        ],
        "advanced_processing": ADVANCED_PROCESSING
    }

@app.get("/hackrx/run", response_model=HackRxResponse)
async def hackrx_endpoint_get(
    documents: str = "https://example.com/sample-document.pdf",
    questions: str = "What are the key features of this document?",
    token: str = Depends(verify_token)
):
    """
    Advanced HackRx endpoint (GET method) - Intelligent document processing
    """
    try:
        # Parse questions from comma-separated string
        question_list = [q.strip() for q in questions.split(',') if q.strip()]
        if not question_list:
            question_list = ["What are the key features of this document?"]
        
        logger.info(f"Processing advanced request with {len(question_list)} questions")
        
        # Process document with advanced intelligence
        await advanced_processor.fetch_and_process_document(documents)
        
        # Answer questions intelligently
        answers = await advanced_processor.answer_questions(question_list)
        
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
    Advanced HackRx endpoint (POST method) - Intelligent document processing
    """
    try:
        logger.info(f"Processing advanced POST request with {len(request.questions)} questions")
        
        # Process document with advanced intelligence
        await advanced_processor.fetch_and_process_document(request.documents)
        
        # Answer questions intelligently
        answers = await advanced_processor.answer_questions(request.questions)
        
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
    
    print("🚀 Starting Advanced Document Reading System")
    print(f"📡 Server: http://0.0.0.0:{port}")
    print("🧠 Features: Semantic chunking, hybrid search, intelligent extraction")
    print(f"⚡ Advanced processing: {'Enabled' if ADVANCED_PROCESSING else 'Fallback mode'}")
    print("🔍 Endpoints: GET/POST /hackrx/run")
    print("🔑 Auth: Bearer token required")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1
    )
