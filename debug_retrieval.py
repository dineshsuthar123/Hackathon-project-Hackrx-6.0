#!/usr/bin/env python3
"""
RETRIEVAL DEBUG SCRIPT
Isolate and debug the core retrieval problem that's causing incorrect answers
"""

import os
import sys
import re
import hashlib
from typing import List, Optional
from dataclasses import dataclass

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

# Import our components
from app_new import (
    DocumentChunk, 
    SimpleDocumentChunker, 
    IndustryStandardRetriever,
    PRODUCTION_MODE,
    ADVANCED_MODELS_AVAILABLE
)

@dataclass
class DebugResult:
    query: str
    expected_answer: str
    chunks_found: List[DocumentChunk]
    retrieval_successful: bool
    error_analysis: str

class RetrievalDebugger:
    """Debug the retrieval system in isolation"""
    
    def __init__(self):
        self.chunker = SimpleDocumentChunker()
        self.retriever = IndustryStandardRetriever()
        
        # Test document content (known good content)
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
    
    def debug_retrieval(self) -> List[DebugResult]:
        """Debug retrieval for known failing questions"""
        
        print("🔧 RETRIEVAL DEBUG SESSION")
        print("=" * 60)
        print(f"📊 Production Mode: {PRODUCTION_MODE}")
        print(f"🧠 Advanced Models: {ADVANCED_MODELS_AVAILABLE}")
        print()
        
        # Create chunks from test document
        chunks = self.chunker.create_chunks(self.test_document)
        print(f"📄 Created {len(chunks)} chunks from test document")
        print()
        
        # Test cases that are known to fail
        test_cases = [
            {
                "query": "What is the waiting period for Gout and Rheumatism?",
                "expected": "36 months",
                "should_find": ["gout", "rheumatism", "36", "months"]
            },
            {
                "query": "What is the waiting period for cataract treatment?", 
                "expected": "24 months",
                "should_find": ["cataract", "24", "months"]
            },
            {
                "query": "What is the ambulance coverage amount?",
                "expected": "Rs. 2,000",
                "should_find": ["ambulance", "rs", "2,000"]
            },
            {
                "query": "What is the room rent limit?",
                "expected": "2% of sum insured per day", 
                "should_find": ["room rent", "2%", "sum insured"]
            },
            {
                "query": "What is the grace period for premium payment?",
                "expected": "30 days",
                "should_find": ["grace period", "thirty days", "premium"]
            },
            {
                "query": "What is the age limit for dependent children?",
                "expected": "3 months to 25 years",
                "should_find": ["dependent children", "3 months", "25 years"]
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"🔍 TEST {i}: {test_case['query']}")
            print(f"📋 Expected: {test_case['expected']}")
            print("-" * 50)
            
            # Perform retrieval
            retrieved_chunks = self.retriever.retrieve(
                query=test_case['query'],
                chunks=chunks,
                top_k=3
            )
            
            # Analyze results
            debug_result = self._analyze_retrieval(
                test_case['query'],
                test_case['expected'], 
                test_case['should_find'],
                retrieved_chunks
            )
            
            results.append(debug_result)
            
            # Print detailed analysis
            self._print_retrieval_analysis(debug_result)
            print("\n" + "="*60 + "\n")
        
        return results
    
    def _analyze_retrieval(self, query: str, expected: str, should_find: List[str], 
                          chunks: List[DocumentChunk]) -> DebugResult:
        """Analyze if retrieval found correct content"""
        
        retrieval_successful = False
        error_analysis = ""
        
        if not chunks:
            error_analysis = "❌ NO CHUNKS RETURNED - Complete retrieval failure"
        else:
            # Check if any chunk contains the expected content
            found_content = False
            for chunk in chunks:
                content_lower = chunk.content.lower()
                matches = sum(1 for term in should_find if term.lower() in content_lower)
                
                if matches >= len(should_find) * 0.7:  # 70% of expected terms
                    found_content = True
                    retrieval_successful = True
                    break
            
            if not found_content:
                error_analysis = f"❌ WRONG CONTENT RETRIEVED - None of the {len(chunks)} chunks contain expected terms: {should_find}"
            else:
                error_analysis = "✅ CORRECT CONTENT FOUND"
        
        return DebugResult(
            query=query,
            expected_answer=expected,
            chunks_found=chunks,
            retrieval_successful=retrieval_successful,
            error_analysis=error_analysis
        )
    
    def _print_retrieval_analysis(self, result: DebugResult):
        """Print detailed analysis of retrieval results"""
        
        print(f"🎯 Result: {result.error_analysis}")
        print(f"📊 Retrieved {len(result.chunks_found)} chunks")
        print()
        
        if result.chunks_found:
            for i, chunk in enumerate(result.chunks_found, 1):
                print(f"📄 CHUNK {i} (Score: {chunk.relevance_score:.2f}, Rerank: {chunk.rerank_score:.2f})")
                print(f"📍 Source: {chunk.source}")
                print(f"📝 Content: {chunk.content[:200]}...")
                
                # Check if this chunk contains expected terms
                content_lower = chunk.content.lower()
                query_lower = result.query.lower()
                
                query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
                chunk_words = set(re.findall(r'\b\w{3,}\b', content_lower))
                overlap = query_words.intersection(chunk_words)
                
                print(f"🔤 Word overlap: {overlap}")
                print(f"📈 Relevance: {len(overlap)}/{len(query_words)} query words found")
                print()
        else:
            print("❌ NO CHUNKS TO ANALYZE")
    
    def run_comprehensive_debug(self):
        """Run comprehensive debugging session"""
        
        print("🚀 Starting comprehensive retrieval debugging...")
        print()
        
        # Run retrieval tests
        results = self.debug_retrieval()
        
        # Summary analysis
        print("📊 DEBUGGING SUMMARY")
        print("=" * 60)
        
        successful = sum(1 for r in results if r.retrieval_successful)
        total = len(results)
        
        print(f"✅ Successful retrievals: {successful}/{total}")
        print(f"❌ Failed retrievals: {total - successful}/{total}")
        print(f"📈 Success rate: {(successful/total)*100:.1f}%")
        print()
        
        if successful < total:
            print("🔧 DEBUGGING RECOMMENDATIONS:")
            print("1. Check keyword search algorithms")
            print("2. Verify chunk creation preserves important content") 
            print("3. Examine scoring mechanisms")
            print("4. Test pattern matching for insurance terms")
            print("5. Validate chunk deduplication logic")
        else:
            print("🎉 All retrievals successful! The issue may be in answer generation.")

def main():
    """Main debugging function"""
    
    debugger = RetrievalDebugger()
    debugger.run_comprehensive_debug()

if __name__ == "__main__":
    main()
