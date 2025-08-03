"""
Test Phase 0 Clean Room Protocol API endpoint
Verify no corruption in API responses
"""

import requests
import json

def test_cleanroom_api():
    """Test the /hackrx/run endpoint with Phase 0 Clean Room Protocol"""
    
    print("🧹 TESTING PHASE 0 CLEAN ROOM API")
    print("=" * 50)
    
    url = "http://localhost:8002/hackrx/run"
    
    headers = {
        "Authorization": "Bearer a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36",
        "Content-Type": "application/json"
    }
    
    # Test questions that previously caused corruption
    payload = {
        "documents": "https://www.careinsurance.com/upload/brochures/Arogya%20Sanjeevani%20Policy%20-%20National%20(ASP-N).pdf",
        "questions": [
            "What is the waiting period for Gout and Rheumatism?",
            "What is the grace period for premium payment?"
        ]
    }
    
    try:
        print(f"📡 Testing endpoint: {url}")
        print(f"❓ Questions: {len(payload['questions'])}")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"\n📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            answers = data.get("answers", [])
            
            print(f"✅ Phase 0 Clean Room API working!")
            print(f"📝 Received {len(answers)} answers")
            
            # Check for corruption in each answer
            corruption_indicators = ['/Title', '/Author', '/Producer', 'endobj', '<<', '>>', '/StructParent']
            
            for i, answer in enumerate(answers, 1):
                print(f"\n❓ Question {i}: {payload['questions'][i-1]}")
                print(f"✅ Answer {i}: {answer}")
                
                # Check for corruption
                has_corruption = any(indicator in answer for indicator in corruption_indicators)
                if has_corruption:
                    print(f"❌ CORRUPTION DETECTED in answer {i}!")
                else:
                    print(f"✅ Answer {i} is CLEAN (no corruption)")
            
            # Overall assessment
            clean_count = sum(1 for answer in answers if not any(
                indicator in answer for indicator in corruption_indicators
            ))
            
            print(f"\n🎯 PHASE 0 API RESULTS:")
            print(f"   Clean answers: {clean_count}/{len(answers)}")
            print(f"   Success rate: {(clean_count/len(answers))*100:.1f}%" if answers else "0%")
            
            if clean_count == len(answers) and answers:
                print("🎉 PHASE 0 CLEAN ROOM API: SUCCESS!")
                print("   All API responses are corruption-free!")
            else:
                print("❌ PHASE 0 CLEAN ROOM API: FAILED!")
                print("   Some responses still contain corruption")
                
        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ API test failed: {e}")

if __name__ == "__main__":
    test_cleanroom_api()
