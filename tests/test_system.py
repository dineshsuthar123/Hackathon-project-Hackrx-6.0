"""
Test script for the LLM-Powered Query-Retrieval System
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from document_processor import DocumentProcessor
from embedding_engine import EmbeddingEngine
from llm_handler import LLMHandler
from clause_matcher import ClauseMatcher
from response_generator import ResponseGenerator

async def test_system():
    """Test the complete system with sample data"""
    print("🚀 Starting LLM-Powered Query-Retrieval System Test")
    print("=" * 60)
    
    try:
        # Initialize components
        print("1. Initializing components...")
        document_processor = DocumentProcessor()
        embedding_engine = EmbeddingEngine()
        await embedding_engine.initialize()
        llm_handler = LLMHandler()
        clause_matcher = ClauseMatcher(embedding_engine)
        response_generator = ResponseGenerator(llm_handler)
        print("✅ Components initialized successfully")
        
        # Test document processing (you can replace with actual document URL)
        print("\n2. Testing document processing...")
        
        # For testing, create a sample document
        sample_document_content = {
            "text": """
            NATIONAL PARIVAR MEDICLAIM PLUS POLICY
            
            COVERAGE:
            This policy provides medical insurance coverage for the insured and family members.
            
            GRACE PERIOD:
            A grace period of thirty (30) days is provided for premium payment after the due date 
            to renew or continue the policy without losing continuity benefits.
            
            WAITING PERIOD FOR PRE-EXISTING DISEASES:
            There is a waiting period of thirty-six (36) months of continuous coverage from the 
            first policy inception for pre-existing diseases and their direct complications to be covered.
            
            MATERNITY COVERAGE:
            The policy covers maternity expenses, including childbirth and lawful medical termination 
            of pregnancy. To be eligible, the female insured person must have been continuously 
            covered for at least 24 months. The benefit is limited to two deliveries or terminations 
            during the policy period.
            
            CATARACT SURGERY:
            The policy has a specific waiting period of two (2) years for cataract surgery.
            
            ORGAN DONOR COVERAGE:
            The policy indemnifies the medical expenses for the organ donor's hospitalization for 
            the purpose of harvesting the organ, provided the organ is for an insured person and 
            the donation complies with the Transplantation of Human Organs Act, 1994.
            
            NO CLAIM DISCOUNT:
            A No Claim Discount of 5% on the base premium is offered on renewal for a one-year 
            policy term if no claims were made in the preceding year. The maximum aggregate NCD 
            is capped at 5% of the total base premium.
            """,
            "sections": {
                "COVERAGE": "This policy provides medical insurance coverage for the insured and family members.",
                "GRACE PERIOD": "A grace period of thirty (30) days is provided for premium payment...",
                "WAITING PERIOD": "There is a waiting period of thirty-six (36) months...",
                "MATERNITY": "The policy covers maternity expenses...",
                "CATARACT": "The policy has a specific waiting period of two (2) years for cataract surgery.",
                "ORGAN DONOR": "The policy indemnifies the medical expenses for the organ donor...",
                "NO CLAIM DISCOUNT": "A No Claim Discount of 5% on the base premium..."
            },
            "metadata": {
                "source": "test_document",
                "file_type": ".pdf",
                "processed_at": "test_time"
            }
        }
        
        print("✅ Sample document created for testing")
        
        # Test embedding and indexing
        print("\n3. Testing embedding and indexing...")
        indexed_content = await embedding_engine.index_content(sample_document_content)
        print(f"✅ Indexed {indexed_content['total_chunks']} text chunks")
        
        # Test queries
        test_questions = [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?",
            "Does this policy cover maternity expenses?",
            "What is the waiting period for cataract surgery?",
            "Are organ donor expenses covered?",
            "What is the No Claim Discount offered?"
        ]
        
        print("\n4. Testing query processing...")
        answers = []
        
        for i, question in enumerate(test_questions, 1):
            print(f"\nProcessing Question {i}: {question}")
            
            # Find relevant clauses
            relevant_clauses = await clause_matcher.find_relevant_clauses(
                question, indexed_content
            )
            print(f"   Found {len(relevant_clauses)} relevant clauses")
            
            # Generate response
            answer, explanation = await response_generator.generate_response(
                question, relevant_clauses, sample_document_content
            )
            
            answers.append(answer)
            print(f"   Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
        
        print("\n" + "=" * 60)
        print("🎉 TEST COMPLETED SUCCESSFULLY!")
        print(f"✅ Processed {len(test_questions)} questions")
        print(f"✅ Generated {len(answers)} answers")
        
        # Display results
        print("\n📋 FINAL RESULTS:")
        print("-" * 40)
        for i, (question, answer) in enumerate(zip(test_questions, answers), 1):
            print(f"\nQ{i}: {question}")
            print(f"A{i}: {answer}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_format():
    """Test the API response format"""
    print("\n🔧 Testing API Response Format...")
    
    sample_response = {
        "answers": [
            "A grace period of thirty days is provided for premium payment after the due date.",
            "There is a waiting period of thirty-six (36) months for pre-existing diseases.",
            "Yes, the policy covers maternity expenses with 24 months continuous coverage requirement."
        ],
        "metadata": {
            "total_questions": 3,
            "processing_timestamp": "test_time",
            "explanations": [
                {"confidence_level": "high", "sources_used": 2},
                {"confidence_level": "high", "sources_used": 1},
                {"confidence_level": "medium", "sources_used": 2}
            ]
        }
    }
    
    print("✅ Sample API response format:")
    print(json.dumps(sample_response, indent=2))
    return True

if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("   Set your OpenAI API key in the .env file to test with actual LLM")
        print("   The test will run with mock responses for demonstration")
    
    # Run tests
    asyncio.run(test_system())
    asyncio.run(test_api_format())
