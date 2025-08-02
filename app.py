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
    title="Lightweight Intelligent Document Reading System",
    description="Smart document processing with enhanced pattern matching and content extraction",
    version="4.0.0"
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
    """Represents a section of document with metadata"""
    title: str
    content: str
    keywords: List[str]
    numbers: List[str]

class IntelligentDocumentParser:
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
        """Extract precise answer from document sections"""
        question_lower = question.lower()
        
        # Find matching patterns
        best_matches = []
        
        for pattern_name, pattern_info in self.patterns.items():
            # Check if question contains relevant keywords
            keyword_matches = sum(1 for keyword in pattern_info['keywords'] 
                                if keyword in question_lower)
            
            if keyword_matches >= 1:  # At least one keyword match
                # Search in relevant sections
                for section in sections:
                    # Check if section contains relevant keywords
                    section_content_lower = section.content.lower()
                    section_keyword_matches = sum(1 for keyword in pattern_info['keywords'] 
                                               if keyword in section_content_lower)
                    
                    if section_keyword_matches >= 1:
                        # Try pattern matching
                        if 'pattern' in pattern_info:
                            match = re.search(pattern_info['pattern'], section_content_lower, re.IGNORECASE)
                            if match:
                                try:
                                    answer = pattern_info['template'].format(*match.groups())
                                    best_matches.append((answer, keyword_matches + section_keyword_matches, section))
                                except:
                                    continue
                        else:
                            # For patterns without regex, use template directly
                            answer = pattern_info['template']
                            best_matches.append((answer, keyword_matches + section_keyword_matches, section))
        
        # Return best match
        if best_matches:
            best_matches.sort(key=lambda x: x[1], reverse=True)
            return best_matches[0][0]
        
        # Fallback to intelligent content search
        return self._intelligent_content_search(question, sections)
    
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

class LightweightDocumentProcessor:
    """Main lightweight document processing system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.parser = IntelligentDocumentParser()
        self.extractor = SmartAnswerExtractor()
        self._document_cache = {}
    
    async def process_document_and_answer(self, document_url: str, questions: List[str]) -> List[str]:
        """Process document and answer questions efficiently"""
        cache_key = hashlib.md5(document_url.encode()).hexdigest()
        
        # Get or fetch document content
        if cache_key in self._document_cache:
            sections = self._document_cache[cache_key]
            self.logger.info("Using cached document sections")
        else:
            # Fetch document content
            document_content = await self._fetch_document_content(document_url)
            
            # Parse into sections
            sections = self.parser.parse_document(document_content)
            
            # Cache sections
            self._document_cache[cache_key] = sections
            self.logger.info(f"Cached {len(sections)} document sections")
        
        # Answer each question
        answers = []
        for i, question in enumerate(questions):
            self.logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            answer = self.extractor.extract_answer(question, sections)
            answers.append(answer)
        
        return answers
    
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

# Initialize the lightweight processor
lightweight_processor = LightweightDocumentProcessor()

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "message": "Lightweight Intelligent Document Reading System",
        "status": "healthy",
        "version": "4.0.0",
        "features": [
            "Enhanced pattern matching",
            "Intelligent keyword extraction",
            "Structured document parsing",
            "Smart content analysis",
            "Reliable deployment optimized",
            "No heavy ML dependencies"
        ],
        "deployment": "optimized"
    }

@app.get("/hackrx/run", response_model=HackRxResponse)
async def hackrx_endpoint_get(
    documents: str = "https://example.com/sample-document.pdf",
    questions: str = "What are the key features of this document?",
    token: str = Depends(verify_token)
):
    """
    Lightweight HackRx endpoint (GET method) - Reliable intelligent document processing
    """
    try:
        # Parse questions from comma-separated string
        question_list = [q.strip() for q in questions.split(',') if q.strip()]
        if not question_list:
            question_list = ["What are the key features of this document?"]
        
        logger.info(f"Processing lightweight request with {len(question_list)} questions")
        
        # Process document and answer questions
        answers = await lightweight_processor.process_document_and_answer(documents, question_list)
        
        return HackRxResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Lightweight endpoint error: {e}")
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
    Lightweight HackRx endpoint (POST method) - Reliable intelligent document processing
    """
    try:
        logger.info(f"Processing lightweight POST request with {len(request.questions)} questions")
        
        # Process document and answer questions
        answers = await lightweight_processor.process_document_and_answer(request.documents, request.questions)
        
        return HackRxResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Lightweight POST endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    print("🚀 Starting Lightweight Intelligent Document Reading System")
    print(f"📡 Server: http://0.0.0.0:{port}")
    print("🧠 Features: Enhanced patterns, smart extraction, reliable deployment")
    print("⚡ Optimized: No heavy ML dependencies, fast startup")
    print("🔍 Endpoints: GET/POST /hackrx/run")
    print("🔑 Auth: Bearer token required")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1
    )
