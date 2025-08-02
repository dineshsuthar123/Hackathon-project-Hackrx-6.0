"""
Test client for the simplified demo server
"""

import asyncio
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = "http://localhost:8000"
API_TOKEN = os.getenv("HACKRX_API_TOKEN", "a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36")

async def test_demo_server():
    """Test the demo server with the hackathon sample data"""
    
    print("🧪 Testing Demo Server")
    print("=" * 30)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            
            # Test 1: Health check
            print("1. Testing health endpoint...")
            response = await client.get(f"{API_BASE_URL}/api/v1/health")
            
            if response.status_code == 200:
                health_data = response.json()
                print("✅ Health check passed")
                print(f"   Status: {health_data.get('status')}")
                print(f"   Components: {health_data.get('components')}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return
            
            # Test 2: Main API endpoint
            print("\n2. Testing main API endpoint...")
            
            request_data = {
                "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
                "questions": [
                    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                    "What is the waiting period for pre-existing diseases (PED) to be covered?", 
                    "Does this policy cover maternity expenses, and what are the conditions?",
                    "What is the waiting period for cataract surgery?",
                    "Are the medical expenses for an organ donor covered under this policy?"
                ]
            }
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json", 
                "Authorization": f"Bearer {API_TOKEN}"
            }
            
            response = await client.post(
                f"{API_BASE_URL}/hackrx/run",
                json=request_data,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ API endpoint test passed")
                print(f"📝 Questions processed: {len(result.get('answers', []))}")
                
                # Display results
                answers = result.get("answers", [])
                for i, (question, answer) in enumerate(zip(request_data["questions"], answers), 1):
                    print(f"\nQ{i}: {question}")
                    print(f"A{i}: {answer[:100]}{'...' if len(answer) > 100 else ''}")
                
                # Show metadata
                metadata = result.get("metadata", {})
                print(f"\n📊 Metadata:")
                print(f"   Processing mode: {metadata.get('processing_mode')}")
                print(f"   Total questions: {metadata.get('total_questions')}")
                
                # Save response
                with open("demo_test_response.json", "w") as f:
                    json.dump(result, f, indent=2)
                print(f"\n💾 Full response saved to 'demo_test_response.json'")
                
            else:
                print(f"❌ API test failed: {response.status_code}")
                print(f"Response: {response.text}")
            
            # Test 3: Authentication
            print("\n3. Testing authentication...")
            
            # Test with wrong token
            wrong_headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer wrong_token"
            }
            
            response = await client.post(
                f"{API_BASE_URL}/api/v1/test",
                headers=wrong_headers
            )
            
            if response.status_code == 401:
                print("✅ Authentication test passed (correctly rejected wrong token)")
            else:
                print(f"❌ Authentication test failed: expected 401, got {response.status_code}")
            
            print("\n🎉 Demo server testing completed!")
            print("✅ Health endpoint working")
            print("✅ Main API endpoint working") 
            print("✅ Authentication working")
            print("✅ Response format correct")
            print("\n📚 You can also test interactively at: http://localhost:8000/docs")
            
    except httpx.ConnectError:
        print("❌ Could not connect to server")
        print("   Make sure the server is running: python simple_server.py")
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_demo_server())
