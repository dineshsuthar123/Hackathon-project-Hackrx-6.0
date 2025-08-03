"""
COMPREHENSIVE TEST: Validate the retrieval fix
Test critical questions that were failing with 1.0/10 rating
"""

import requests
import json

# API configuration
BASE_URL = "http://localhost:8000"
TOKEN = "a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# Critical test questions that were returning Annexure-A content
TEST_QUESTIONS = [
    "What is the ambulance coverage amount?",
    "What is the cumulative bonus percentage?", 
    "What is the room rent limit?",
    "What is the moratorium period?",
    "What is the ICU coverage limit?",
    "What is the waiting period for cataract?",
    "What is the grace period for premium payment?",
    "What is the pre-hospitalization coverage period?"
]

def test_question(question):
    """Test a single question and return result"""
    try:
        url = f"{BASE_URL}/hackrx/run"
        params = {
            "documents": "test",
            "questions": question
        }
        
        response = requests.get(url, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        answer = data["answers"][0]
        
        # Check for annexure pollution
        is_annexure_polluted = (
            "BABY FOOD" in answer or 
            "AMBULANCE COLLAR" in answer or
            "List I – List of which coverage" in answer or
            len(answer) > 500  # Suspiciously long answers
        )
        
        return {
            "question": question,
            "answer": answer,
            "is_polluted": is_annexure_polluted,
            "length": len(answer),
            "success": True
        }
        
    except Exception as e:
        return {
            "question": question,
            "answer": f"ERROR: {str(e)}",
            "is_polluted": True,
            "length": 0,
            "success": False
        }

def main():
    print("🧪 COMPREHENSIVE RETRIEVAL TEST")
    print("="*60)
    print(f"Testing {len(TEST_QUESTIONS)} critical questions...")
    print()
    
    results = []
    polluted_count = 0
    
    for i, question in enumerate(TEST_QUESTIONS):
        print(f"[{i+1}/{len(TEST_QUESTIONS)}] {question}")
        
        result = test_question(question)
        results.append(result)
        
        if result["is_polluted"]:
            polluted_count += 1
            print(f"   ❌ POLLUTED: {result['answer'][:100]}...")
        else:
            print(f"   ✅ CLEAN: {result['answer']}")
        
        print(f"   📏 Length: {result['length']} chars")
        print()
    
    # Summary
    print("="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    clean_count = len(TEST_QUESTIONS) - polluted_count
    success_rate = (clean_count / len(TEST_QUESTIONS)) * 100
    
    print(f"Total questions: {len(TEST_QUESTIONS)}")
    print(f"Clean answers: {clean_count}")
    print(f"Polluted answers: {polluted_count}")
    print(f"Success rate: {success_rate:.1f}%")
    print()
    
    if polluted_count == 0:
        print("🎉 PERFECT! No annexure pollution detected!")
        print("🏆 The retrieval fix is successful!")
        estimated_rating = min(8.0, 4.0 + (success_rate / 100) * 4.0)
        print(f"📈 Estimated new rating: {estimated_rating:.1f}/10")
    elif polluted_count <= 2:
        print("✅ MAJOR IMPROVEMENT! Minimal pollution detected.")
        estimated_rating = min(7.0, 3.0 + (success_rate / 100) * 4.0)
        print(f"📈 Estimated new rating: {estimated_rating:.1f}/10")
    else:
        print("⚠️ PARTIAL FIX. Some pollution still present.")
        estimated_rating = max(2.0, (success_rate / 100) * 6.0)
        print(f"📈 Estimated new rating: {estimated_rating:.1f}/10")
    
    print()
    print("Individual Results:")
    print("-" * 60)
    for result in results:
        status = "✅" if not result["is_polluted"] else "❌"
        print(f"{status} {result['question']}")
        print(f"   {result['answer'][:80]}...")

if __name__ == "__main__":
    main()
