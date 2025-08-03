"""
CRITICAL DEBUG: Isolate the broken retrieval system
This will show us exactly what chunks are being returned for basic questions
"""

import os
import re
import logging
from typing import List
from dataclasses import dataclass

# Optional advanced dependencies
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    content: str
    source: str
    relevance_score: float = 0.0

def get_test_document():
    """Get the same document content the API uses"""
    return """
AROGYA SANJEEVANI POLICY - NATIONAL INSURANCE COMPANY LIMITED
Policy UIN: NICHLIP25041V022425

1. DEFINITIONS

3.22. Grace Period means the specified period of time, immediately following the premium due date during which premium payment can be made to renew or continue a policy in force without loss of continuity benefits. The Grace Period for payment of the premium shall be thirty days.

3.23. Hospital means any institution established for in-patient care and day care treatment of disease/injuries which has qualified nursing staff under its employment round the clock; has at least ten (10) inpatient beds, in those towns having a population of less than ten lacs and fifteen inpatient beds in all other places; has qualified medical practitioner (s) in charge round the clock.

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
The following procedures will be covered either as in patient or as part of day care treatment in a hospital subject to the limit of 50% of the Sum Insured.

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
9. Cataract and age related eye ailments

ii. 36 Months waiting period
1. Treatment for joint replacement unless arising from accident
2. Age-related Osteoarthritis & Osteoporosis

7. EXCLUSIONS

7.15. Maternity Expenses (Code – Excl 18)
i. Medical treatment expenses traceable to childbirth (including complicated deliveries and caesarean sections incurred during hospitalization) except ectopic pregnancy;
ii. Expenses towards miscarriage (unless due to an accident) and lawful medical termination of pregnancy during the policy period.

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

FREE LOOK PERIOD:
The policyholder may return the policy within 15 days of its receipt and obtain refund of premium paid, subject to deduction of proportionate risk premium for the period of cover, stamp duty charges, and proportionate charges towards medical examination (if any).

Pre-Existing Disease means any condition, ailment or injury or related condition(s) for which medical advice or treatment was received from a physician within 48 months prior to the effective date of policy.

ANNEXURE-A
List I – List of which coverage is not available in the policy 
Sl Item 
1 BABY FOOD 
2 BABY UTILITIES CHARGES 
3 BEAUTY SERVICES 
4 BELTS/ BRACES 
5 BUDS 
6 COLD PACK/HOT PACK 
7 CARRY BAGS 
8 EMAIL / INTERNET CHARGES 
9 FOOD CHARGES (OTHER THAN PATIENT's DIET PROVIDED BY HOSPITAL) 
10 LEGGINGS 
11 LAUNDRY CHARGES 
12 MINERAL WATER 
13 SANITARY PAD 
14 TELEPHONE CHARGES 
15 GUEST SERVICES 
16 CREPE BANDAGE 
17 DIAPER OF ANY TYPE 
18 EYELET COLLAR 
19 SLINGS 
20 BLOOD GROUPING AND CROSS MATCHING OF DONORS SAMPLES 
21 SERVICE CHARGES WHERE NURSING CHARGE ALSO CHARGED 
22 Television Charges 
23 SURCHARGES 
24 ATTENDANT CHARGES 
25 EXTRA DIET OF PATIENT (OTHER THAN THAT WHICH FORMS PART OF BED CHARGE) 
26 BIRTH CERTIFICATE 
27 CERTIFICATE CHARGES 
28 COURIER CHARGES 
29 CONVEYANCE CHARGES 
30 MEDICAL CERTIFICATE 
31 MEDICAL RECORDS 
32 PHOTOCOPIES CHARGES 
33 MORTUARY CHARGES 
34 WALKING AIDS CHARGES 
35 OXYGEN CYLINDER (FOR USAGE OUTSIDE THE HOSPITAL) 
36 SPACER 
37 SPIROMETRE 
38 NEBULIZER KIT 
39 STEAM INHALER 
40 ARMSLING 
41 THERMOMETER 
42 CERVICAL COLLAR 
43 SPLINT 
44 DIABETIC FOOT WEAR 
45 KNEE BRACES (LONG/ SHORT/ HINGED) 
46 KNEE IMMOBILIZER/SHOULDER IMMOBILIZER 
47 LUMBO SACRAL BELT 
48 NIMBUS BED OR WATER OR AIR BED CHARGES 
49 AMBULANCE COLLAR 
50 AMBULANCE EQUIPMENT 
51 ABDOMINAL BINDER 
52 PRIVATE NURSES CHARGES - SPECIAL NURSING CHARGES 
53 SUGAR FREE Tablets 
54 CREAMS POWDERS LOTIONS (Toiletries are not payable, only prescribed medical pharmaceuticals payable) 
55 ECG ELECTRODES 
56 GLOVES 
57 NEBULISATION KIT 
58 ANY KIT WITH NO DETAILS MENTIONED [DELIVERY KIT, ORTHOKIT, RECOVERY KIT, ETC] 
59 KIDNEY TRAY 
60 MASK 
61 OUNCE GLASS 
62 OXYGEN MASK 
63 PELVIC TRACTION BELT 
64 PAN CAN 
65 TROLLY COVER 
66 UROMETER, URINE JUG 
67 VASOFIX SAFETY 

List II – Items that are to be subsumed into Room Charges 
Sl Item 
1 BABY CHARGES (UNLESS SPECIFIED/INDICATED) 
2 HAND WASH 
3 SHOE COVER 
4 CAPS 
5 CRADLE CHARGES 
6 COMB 
7 EAU -DE-COLOGNE / ROOM FRESHNERS 
8 FOOT COVER 
9 GOWN 
10 SLIPPERS 
11 TISSUE PAPER 
12 TOOTH PASTE 
13 TOOTH BRUSH 
14 BED PAN 
15 FACE MASK 
16 FLEXI MASK 
17 HAND HOLDER 
18 SPUTUM CUP 
19 DISINFECTANT LOTIONS 
20 LUXURY TAX 
21 HVAC 
22 HOUSE KEEPING CHARGES 
23 AIR CONDITIONER CHARGES 
24 IM IV INJECTION CHARGES 
25 CLEAN SHEET 
26 BLANKET/WARMER BLANKET 
27 ADMISSION KIT 
28 DIABETIC CHART CHARGES 
29 DOCUMENTATION CHARGES / ADMINISTRATIVE EXPENSES 
30 DISCHARGE PROCEDURE CHARGES 
31 DAILY CHART CHARGES 
32 ENTRANCE PASS / VISITORS PASS CHARGES 
33 EXPENSES RELATED TO PRESCRIPTION ON DISCHARGE 
34 FILE OPENING CHARGES 
35 INCIDENTAL EXPENSES / MISC
    """

def create_simple_chunks(text: str) -> List[DocumentChunk]:
    """Create simple sentence-based chunks"""
    # Split by periods and clean
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if len(sentence) > 20:  # Skip very short fragments
            chunk = DocumentChunk(
                content=sentence,
                source=f"chunk_{i}",
                relevance_score=0.0
            )
            chunks.append(chunk)
    
    print(f"✅ Created {len(chunks)} simple chunks")
    return chunks

def debug_keyword_search(query: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Debug keyword search - show what's happening"""
    print(f"\n🔍 DEBUGGING KEYWORD SEARCH for: '{query}'")
    
    query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
    print(f"Query words: {query_words}")
    
    scored_chunks = []
    
    for i, chunk in enumerate(chunks):
        content_words = set(re.findall(r'\b\w{3,}\b', chunk.content.lower()))
        overlap = len(query_words.intersection(content_words))
        
        if overlap > 0:
            chunk.relevance_score = overlap
            scored_chunks.append(chunk)
            
            # Show first few matches for debugging
            if len(scored_chunks) <= 5:
                print(f"\nMatch {len(scored_chunks)}: Score={overlap}")
                print(f"Content: {chunk.content[:100]}...")
                print(f"Overlapping words: {query_words.intersection(content_words)}")
    
    # Sort by score
    scored_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
    
    print(f"\n📊 Found {len(scored_chunks)} chunks with keyword matches")
    return scored_chunks[:5]

def debug_semantic_search(query: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Debug semantic search if available"""
    if not ADVANCED_MODELS_AVAILABLE:
        print("❌ Semantic search not available - missing dependencies")
        return []
    
    print(f"\n🧠 DEBUGGING SEMANTIC SEARCH for: '{query}'")
    
    try:
        # Load model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Model loaded")
        
        # Get embeddings
        query_embedding = model.encode([query])
        chunk_texts = [chunk.content for chunk in chunks]
        chunk_embeddings = model.encode(chunk_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        # Get top 5
        top_indices = np.argsort(similarities)[::-1][:5]
        
        results = []
        for i, idx in enumerate(top_indices):
            chunk = chunks[idx]
            chunk.relevance_score = float(similarities[idx])
            results.append(chunk)
            
            print(f"\nSemantic Match {i+1}: Score={similarities[idx]:.3f}")
            print(f"Content: {chunk.content[:100]}...")
        
        return results
        
    except Exception as e:
        print(f"❌ Semantic search failed: {e}")
        return []

def test_question(question: str, chunks: List[DocumentChunk]):
    """Test one question and show detailed results"""
    print(f"\n{'='*60}")
    print(f"TESTING QUESTION: {question}")
    print(f"{'='*60}")
    
    # Test keyword search
    keyword_results = debug_keyword_search(question, chunks)
    
    # Test semantic search
    semantic_results = debug_semantic_search(question, chunks)
    
    print(f"\n📋 SUMMARY FOR: '{question}'")
    print(f"Keyword matches: {len(keyword_results)}")
    print(f"Semantic matches: {len(semantic_results)}")
    
    if keyword_results:
        print(f"\n🏆 TOP KEYWORD RESULT:")
        print(f"Score: {keyword_results[0].relevance_score}")
        print(f"Content: {keyword_results[0].content}")
        
        # Check if it's the problematic annexure content
        if "ANNEXURE" in keyword_results[0].content.upper() or "BABY FOOD" in keyword_results[0].content:
            print("🚨 WARNING: This is the problematic Annexure-A content!")
    
    if semantic_results:
        print(f"\n🧠 TOP SEMANTIC RESULT:")
        print(f"Score: {semantic_results[0].relevance_score:.3f}")
        print(f"Content: {semantic_results[0].content}")
        
        if "ANNEXURE" in semantic_results[0].content.upper() or "BABY FOOD" in semantic_results[0].content:
            print("🚨 WARNING: Semantic search also returning Annexure-A!")

def main():
    """Main debug function"""
    print("🐛 CRITICAL RETRIEVAL DEBUG")
    print("="*50)
    
    # Get document and create chunks
    document = get_test_document()
    chunks = create_simple_chunks(document)
    
    # Test with the exact question that should work
    test_questions = [
        "What is the moratorium period?",
        "What is the cumulative bonus percentage?",
        "What is the room rent limit?",
        "What is the ambulance coverage amount?"
    ]
    
    for question in test_questions:
        test_question(question, chunks)
    
    print(f"\n{'='*60}")
    print("DEBUG COMPLETE")
    print("Check if Annexure-A content is dominating the results!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
