"""
TEST THE FIXED CHUNKER - Does it eliminate the Annexure-A problem?
"""

import os
import re
import logging
from typing import List
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    content: str
    source: str
    relevance_score: float = 0.0

class FixedDocumentChunker:
    """Fixed chunker with annexure handling"""
    
    def __init__(self, chunk_size: int = 200, overlap: int = 30):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = logging.getLogger(__name__)
    
    def create_chunks(self, text: str) -> List[DocumentChunk]:
        """Create overlapping chunks with special handling for annexures and lists"""
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # CRITICAL FIX: Handle Annexure-A separately to prevent giant chunks
        if "ANNEXURE-A" in text or "List I – List of which coverage is not available" in text:
            return self._create_smart_chunks_with_annexure_handling(text)
        
        # Regular sentence-based chunking for normal content
        return self._create_sentence_chunks(text)
    
    def _create_smart_chunks_with_annexure_handling(self, text: str) -> List[DocumentChunk]:
        """Handle document with annexure lists intelligently"""
        chunks = []
        
        # Split document at major sections
        sections = re.split(r'(ANNEXURE-A|List I –|List II –)', text)
        
        for i, section in enumerate(sections):
            section = section.strip()
            if len(section) < 20:
                continue
            
            # Handle annexure lists differently
            if "ANNEXURE" in section or "List I –" in section or "List II –" in section:
                # Skip the problematic annexure sections entirely
                self.logger.info(f"🚫 Skipping annexure section to prevent noise")
                continue
            else:
                # Process normal sections with sentence chunking
                section_chunks = self._create_sentence_chunks(section)
                chunks.extend(section_chunks)
        
        self.logger.info(f"✅ Created {len(chunks)} chunks (annexures filtered)")
        return chunks
    
    def _create_sentence_chunks(self, text: str) -> List[DocumentChunk]:
        """Create sentence-based chunks for normal content"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        chunks = []
        current_chunk = ""
        current_words = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence exceeds chunk size, create chunk
            if current_words + sentence_words > self.chunk_size and current_chunk:
                chunk = DocumentChunk(
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
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                source="document"
            )
            chunks.append(chunk)
        
        return chunks

def get_test_document():
    """Get the same document content the API uses"""
    return """
AROGYA SANJEEVANI POLICY - NATIONAL INSURANCE COMPANY LIMITED
Policy UIN: NICHLIP25041V022425

4. COVERAGE

4.1. Hospitalization
The Company shall indemnify Medical Expense incurred for Hospitalization of the Insured Person during the Policy Period:
i. Room Rent, Boarding, Nursing Expenses all inclusive as provided by the Hospital up to 2% of the sum insured subject to maximum of Rs. 5,000/- per day
ii. Intensive Care Unit (ICU) / Intensive Cardiac Care Unit (ICCU) expenses up to 5% of the sum insured subject to maximum of Rs. 10,000/- per day
iii. Surgeon, Anesthetist, Medical Practitioner, Consultants, Specialist Fees
iv. Anesthesia, blood, oxygen, operation theatre charges, surgical appliances, medicines and drugs, costs towards diagnostics
v. Expenses incurred on road Ambulance subject to a maximum of Rs 2,000 per hospitalization.

5. CUMULATIVE BONUS (CB)
Cumulative Bonus will be increased by 5% in respect of each claim free Policy Period (where no claims are reported and admitted), provided the policy is renewed with the company without a break subject to maximum of 50% of the sum insured under the current Policy Period.

8. Moratorium Period:
After completion of sixty continuous months of coverage (including Portability and Migration), no claim shall be contestable by the Company on grounds of non-disclosure, misrepresentation, except on grounds of established fraud.

ANNEXURE-A
List I – List of which coverage is not available in the policy 
Sl Item 
1 BABY FOOD 
2 BABY UTILITIES CHARGES 
3 BEAUTY SERVICES 
49 AMBULANCE COLLAR 
50 AMBULANCE EQUIPMENT 
51 ABDOMINAL BINDER 
52 PRIVATE NURSES CHARGES - SPECIAL NURSING CHARGES 
53 SUGAR FREE Tablets 
67 VASOFIX SAFETY 

List II – Items that are to be subsumed into Room Charges 
Sl Item 
1 BABY CHARGES (UNLESS SPECIFIED/INDICATED) 
2 HAND WASH 
35 INCIDENTAL EXPENSES / MISC
    """

def test_fixed_chunker():
    print("🔧 TESTING FIXED CHUNKER")
    print("="*50)
    
    # Get document and test both chunkers
    document = get_test_document()
    
    # Test fixed chunker
    fixed_chunker = FixedDocumentChunker()
    fixed_chunks = fixed_chunker.create_chunks(document)
    
    print(f"Fixed chunker created {len(fixed_chunks)} chunks")
    
    # Check if any chunks contain annexure content
    annexure_chunks = 0
    for i, chunk in enumerate(fixed_chunks):
        if "BABY FOOD" in chunk.content or "AMBULANCE COLLAR" in chunk.content:
            annexure_chunks += 1
            print(f"⚠️ Chunk {i+1} contains annexure content!")
        else:
            print(f"✅ Chunk {i+1}: {chunk.content[:80]}...")
    
    print(f"\n📊 Summary:")
    print(f"Total chunks: {len(fixed_chunks)}")
    print(f"Annexure chunks: {annexure_chunks}")
    
    if annexure_chunks == 0:
        print("🎉 SUCCESS: No annexure chunks found!")
    else:
        print("❌ FAILED: Annexure chunks still present")

if __name__ == "__main__":
    test_fixed_chunker()
