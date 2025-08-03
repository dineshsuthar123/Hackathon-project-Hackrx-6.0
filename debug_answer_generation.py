#!/usr/bin/env python3
"""
ANSWER GENERATION DEBUG SCRIPT
Debug the answer generation step since retrieval is working perfectly
"""

import os
import sys
from typing import List

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

# Import our components
from app_new import (
    DocumentChunk, 
    SimpleDocumentChunker, 
    IndustryStandardRetriever,
    PrecisionAnswerGenerator
)

class AnswerGenerationDebugger:
    """Debug the answer generation step in isolation"""
    
    def __init__(self):
        self.chunker = SimpleDocumentChunker()
        self.retriever = IndustryStandardRetriever()
        self.generator = PrecisionAnswerGenerator()
        
        # Test document content (same as retrieval debug)
        self.test_document = """
        WAITING PERIODS
        
        Pre-existing diseases are subject to a waiting period of 3 years from the date of first enrollment.
        
        Specific conditions waiting periods:
        - Cataract: 24 months
        - Joint replacement: 48 months  
        - Gout and Rheumatism: 36 months
        - Hernia, Hydrocele, Congenital internal diseases, stones in urinary system and gall bladder: 24 months
        
        AMBULANCE COVERAGE
        Expenses incurred on road ambulance subject to maximum of Rs. 2,000/- per hospitalization are payable.
        
        ROOM RENT COVERAGE
        Room rent, boarding and nursing expenses are covered up to 2% of sum insured per day.
        
        ICU COVERAGE
        Intensive Care Unit (ICU/ICCU) expenses are covered up to 5% of sum insured per day.
        
        CUMULATIVE BONUS
        Cumulative Bonus @ 5% for each claim free year up to maximum of 50% of the sum insured.
        
        GRACE PERIOD
        There shall be a grace period of thirty days for payment of renewal premium.
        
        MORATORIUM PERIOD
        After completion of sixty continuous months, no health insurance claim shall be contestable.
        
        DEPENDENT CHILDREN AGE LIMIT
        Dependent children are covered from 3 months to 25 years of age.
        
        AYUSH HOSPITALS
        AYUSH hospitals must have minimum 5 in-patient beds and round the clock availability of qualified AYUSH Medical Practitioner.
        
        PLANNED HOSPITALIZATION NOTIFICATION
        For planned hospitalization, the insured person or attendant must notify the Company at least 48 hours in advance.
        
        EXCLUSIONS
        The following are not covered:
        - Treatment arising from breach of law with criminal intent
        - Dietary supplements and vitamins that can be purchased without prescription
        
        DEFINITIONS
        Chronic Condition means a disease, illness, or injury that has ongoing monitoring requirements.
        Migration means a facility to transfer credits from one policy to another with the same insurer.
        Day Care Centre means a facility for carrying out procedures without in-patient services.
        """
    
    def debug_answer_generation(self):
        """Debug answer generation with known good retrieval results"""
        
        print("🔧 ANSWER GENERATION DEBUG SESSION")
        print("=" * 60)
        
        # Create chunks from test document
        chunks = self.chunker.create_chunks(self.test_document)
        print(f"📄 Created {len(chunks)} chunks from test document")
        print()
        
        # Test cases that we know have perfect retrieval
        test_cases = [
            {
                "query": "What is the waiting period for Gout and Rheumatism?",
                "expected": "36 months",
                "correct_chunk_content": "Gout and Rheumatism: 36 months"
            },
            {
                "query": "What is the waiting period for cataract treatment?", 
                "expected": "24 months",
                "correct_chunk_content": "Cataract: 24 months"
            },
            {
                "query": "What is the ambulance coverage amount?",
                "expected": "Rs. 2,000",
                "correct_chunk_content": "Rs. 2,000/- per hospitalization"
            },
            {
                "query": "What is the grace period for premium payment?",
                "expected": "30 days",
                "correct_chunk_content": "grace period of thirty days"
            },
            {
                "query": "What is the age limit for dependent children?",
                "expected": "3 months to 25 years",
                "correct_chunk_content": "from 3 months to 25 years of age"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"🔍 TEST {i}: {test_case['query']}")
            print(f"📋 Expected: {test_case['expected']}")
            print(f"🎯 Should find: {test_case['correct_chunk_content']}")
            print("-" * 50)
            
            # Step 1: Get retrieval results (we know these work)
            retrieved_chunks = self.retriever.retrieve(
                query=test_case['query'],
                chunks=chunks,
                top_k=3
            )
            
            print(f"📊 Retrieved {len(retrieved_chunks)} chunks")
            
            # Step 2: Check if correct content is in top chunk
            if retrieved_chunks:
                top_chunk = retrieved_chunks[0]
                has_correct_content = test_case['correct_chunk_content'].lower() in top_chunk.content.lower()
                print(f"✅ Top chunk has correct content: {has_correct_content}")
                if has_correct_content:
                    print(f"📝 Top chunk: {top_chunk.content[:200]}...")
                else:
                    print(f"❌ Top chunk: {top_chunk.content[:200]}...")
                print()
            
            # Step 3: Generate answer
            print("🤖 Generating answer...")
            generated_answer = self.generator.generate_answer(test_case['query'], retrieved_chunks)
            
            # Step 4: Compare results
            print(f"📤 Generated: {generated_answer}")
            print(f"📋 Expected:  {test_case['expected']}")
            
            # Simple correctness check
            expected_lower = test_case['expected'].lower()
            generated_lower = generated_answer.lower()
            
            is_correct = False
            if expected_lower in generated_lower:
                is_correct = True
            elif "36 months" in expected_lower and ("36" in generated_lower and ("month" in generated_lower or "months" in generated_lower)):
                is_correct = True
            elif "24 months" in expected_lower and ("24" in generated_lower and ("month" in generated_lower or "months" in generated_lower)):
                is_correct = True
            elif "2,000" in expected_lower and "2,000" in generated_lower:
                is_correct = True
            elif "30 days" in expected_lower and ("30" in generated_lower or "thirty" in generated_lower) and "day" in generated_lower:
                is_correct = True
            elif "3 months to 25 years" in expected_lower and "3" in generated_lower and "25" in generated_lower:
                is_correct = True
            
            status = "✅ CORRECT" if is_correct else "❌ INCORRECT" 
            print(f"🎯 Result: {status}")
            print()
            
            if not is_correct:
                print("🔧 DEBUGGING ANSWER GENERATION:")
                # Debug the answer extraction methods
                context = self.generator._combine_chunks(retrieved_chunks[:3])
                print(f"📄 Combined context length: {len(context)}")
                print(f"📝 Context preview: {context[:300]}...")
                
                # Test individual extraction methods
                insurance_answer = self.generator._extract_insurance_answer(test_case['query'].lower(), context)
                print(f"🏥 Insurance extraction: {insurance_answer}")
                
                sentence_answer = self.generator._extract_best_sentence_from_context(test_case['query'].lower(), context)
                print(f"📝 Sentence extraction: {sentence_answer}")
                
                fallback_answer = self.generator._extract_fallback_answer(context)
                print(f"⚠️ Fallback extraction: {fallback_answer}")
                print()
            
            print("="*60)
            print()

def main():
    """Main debugging function"""
    
    debugger = AnswerGenerationDebugger()
    debugger.debug_answer_generation()

if __name__ == "__main__":
    main()
