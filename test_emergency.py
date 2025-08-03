"""
Test Emergency Deployment Version
"""

import asyncio
import logging
from app_emergency import EmergencyDocumentProcessor

logging.basicConfig(level=logging.INFO)

async def test_emergency_system():
    processor = EmergencyDocumentProcessor()
    
    # Test questions
    test_questions = [
        "What is the waiting period for Gout and Rheumatism?",
        "What is the waiting period for cataract treatment?", 
        "What is the maximum amount covered for road ambulance expenses per hospitalization?",
        "What is the grace period for premium payment?",
        "What is the age range for dependent children coverage?"
    ]
    
    expected_answers = [
        "36 months",
        "24 months", 
        "Rs. 2,000",
        "30 days",
        "3 months to 25 years"
    ]
    
    print("🚨 TESTING EMERGENCY DEPLOYMENT VERSION")
    print("="*60)
    
    # Use fallback document (since no real URL)
    document_url = "https://example.com/test.pdf"
    
    try:
        answers = await processor.process_document_questions(document_url, test_questions)
        
        print("\nRESULTS:")
        print("="*60)
        
        success_count = 0
        for i, (question, answer, expected) in enumerate(zip(test_questions, answers, expected_answers), 1):
            is_correct = expected.lower() in answer.lower()
            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            
            if is_correct:
                success_count += 1
            
            print(f"\n{i}. {question}")
            print(f"   ANSWER: {answer}")
            print(f"   EXPECTED: {expected}")
            print(f"   STATUS: {status}")
        
        accuracy = (success_count / len(test_questions)) * 100
        print(f"\n" + "="*60)
        print(f"EMERGENCY MODE SCORE: {success_count}/{len(test_questions)} ({accuracy:.1f}%)")
        
        if accuracy >= 80:
            print("🚨 EMERGENCY DEPLOYMENT READY!")
        else:
            print("❌ EMERGENCY MODE NEEDS FIXES")
        
        print("="*60)
        
    except Exception as e:
        print(f"❌ Emergency test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_emergency_system())
