#!/usr/bin/env python3
"""
Test script for the improved answer generation system
Tests various question types and answer quality
"""

import asyncio
import httpx
import json
import time
from typing import List, Dict

BASE_URL = "http://localhost:8002"
API_TOKEN = "a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36"

# Enhanced test questions covering different categories
TEST_QUESTIONS = [
    # Company information
    "What is the name of the insurance company?",
    "Who is the insurance provider?",
    
    # Contact information
    "What is the contact number of the company?",
    "How can I contact the insurance company?",
    "What is the email address of the company?",
    
    # Policy information
    "What is the policy number or UIN?",
    "What is the registration number of the company?",
    
    # Coverage questions
    "Does the policy cover maternity expenses?",
    "Is AYUSH treatment covered under this policy?",
    "Are pre-existing diseases covered?",
    
    # Technical details
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?",
    "What is the waiting period for cataract surgery?",
    
    # Location and address
    "Where is the company office located?",
    "What is the address of the insurance company?",
    
    # Benefits and features
    "What is the no claim discount offered?",
    "Are health check-ups covered?",
    "What are the room rent limits?"
]

async def test_improved_answers():
    """Test the improved answer generation system"""
    print("🧪 Testing Improved Answer Generation System")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Test with actual insurance document
        test_payload = {
            "documents": "https://www.nationalinsurance.nic.co.in/sites/default/files/2024-11/National%20Parivar%20Mediclaim%20Plus%20Policy%20Wording.pdf",
            "questions": TEST_QUESTIONS
        }
        
        headers = {
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        print(f"📤 Sending {len(TEST_QUESTIONS)} test questions...")
        print(f"📄 Document: National Parivar Mediclaim Plus Policy")
        print()
        
        start_time = time.time()
        
        try:
            response = await client.post(
                f"{BASE_URL}/hackrx/run",
                json=test_payload,
                headers=headers
            )
            
            response_time = time.time() - start_time
            print(f"⏱️  Response time: {response_time:.2f} seconds")
            print(f"📊 Status code: {response.status_code}")
            print()
            
            if response.status_code == 200:
                result = response.json()
                answers = result.get('answers', [])
                
                print("📝 Question and Answer Analysis:")
                print("=" * 60)
                
                for i, (question, answer) in enumerate(zip(TEST_QUESTIONS, answers), 1):
                    print(f"\n🔍 Question {i}: {question}")
                    print(f"💡 Answer: {answer}")
                    
                    # Analyze answer quality
                    quality_score = analyze_answer_quality(question, answer)
                    print(f"⭐ Quality Score: {quality_score}/5")
                    
                    if i % 5 == 0:  # Add separator every 5 questions
                        print("\n" + "-" * 40)
                
                # Overall statistics
                print("\n" + "=" * 60)
                print("📊 OVERALL STATISTICS")
                print("=" * 60)
                
                total_questions = len(TEST_QUESTIONS)
                total_answers = len(answers)
                
                print(f"Total Questions: {total_questions}")
                print(f"Total Answers: {total_answers}")
                print(f"Response Rate: {(total_answers/total_questions)*100:.1f}%")
                print(f"Average Response Time: {response_time/total_questions:.2f}s per question")
                
                # Calculate quality metrics
                quality_scores = [analyze_answer_quality(q, a) for q, a in zip(TEST_QUESTIONS, answers)]
                avg_quality = sum(quality_scores) / len(quality_scores)
                print(f"Average Quality Score: {avg_quality:.2f}/5")
                
                specific_answers = sum(1 for score in quality_scores if score >= 4)
                print(f"High Quality Answers: {specific_answers}/{total_questions} ({(specific_answers/total_questions)*100:.1f}%)")
                
                # Save detailed results
                save_test_results(TEST_QUESTIONS, answers, quality_scores, response_time)
                
            else:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error during testing: {str(e)}")

def analyze_answer_quality(question: str, answer: str) -> int:
    """Analyze the quality of an answer on a scale of 1-5"""
    if not answer or len(answer.strip()) < 10:
        return 1
    
    question_lower = question.lower()
    answer_lower = answer.lower()
    
    # Check for specific information types
    score = 2  # Base score for having an answer
    
    # Company name questions
    if any(word in question_lower for word in ['company', 'provider', 'insurer']):
        if 'national insurance' in answer_lower:
            score = 5
        elif 'company' in answer_lower:
            score = 4
    
    # Contact information
    elif any(word in question_lower for word in ['contact', 'phone', 'email']):
        if re.search(r'[\d-]{8,}|@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', answer):
            score = 5
        elif any(word in answer_lower for word in ['contact', 'phone', 'email']):
            score = 4
    
    # Yes/No questions
    elif question_lower.startswith(('does', 'is', 'are', 'can', 'will')):
        if any(word in answer_lower for word in ['yes', 'no', 'covered', 'not covered']):
            score = 5
        elif 'policy' in answer_lower:
            score = 4
    
    # Specific policy details
    elif any(word in question_lower for word in ['period', 'waiting', 'grace', 'uin', 'number']):
        if re.search(r'\d+', answer):
            score = 5
        elif any(word in answer_lower for word in ['period', 'months', 'days', 'years']):
            score = 4
    
    # Location questions
    elif any(word in question_lower for word in ['where', 'address', 'location']):
        if re.search(r'\d{6}|road|building|premises', answer_lower):
            score = 5
        elif 'address' in answer_lower:
            score = 4
    
    # Generic fallback penalty
    if 'cannot find' in answer_lower or 'error' in answer_lower:
        score = max(1, score - 2)
    
    return min(5, score)

def save_test_results(questions: List[str], answers: List[str], quality_scores: List[int], response_time: float):
    """Save test results to a JSON file"""
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(questions),
        "response_time_seconds": response_time,
        "average_quality_score": sum(quality_scores) / len(quality_scores),
        "test_results": [
            {
                "question": q,
                "answer": a,
                "quality_score": s
            }
            for q, a, s in zip(questions, answers, quality_scores)
        ]
    }
    
    with open("improved_answer_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: improved_answer_test_results.json")

if __name__ == "__main__":
    import re
    asyncio.run(test_improved_answers())
