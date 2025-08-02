"""
Example client script to test the API endpoint
"""

import asyncio
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = "http://localhost:8000"
API_TOKEN = os.getenv("HACKRX_API_TOKEN", "a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36")

async def test_hackrx_endpoint():
    """Test the /hackrx/run endpoint with sample data"""
    
    # Sample request data (as provided in the hackathon requirements)
    request_data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    
    print("🚀 Testing HackRx API Endpoint")
    print("=" * 50)
    print(f"📡 Endpoint: {API_BASE_URL}/api/v1/hackrx/run")
    print(f"📋 Questions: {len(request_data['questions'])}")
    print(f"📄 Document: {request_data['documents'][:50]}...")
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            print("\n⏳ Sending request...")
            
            response = await client.post(
                f"{API_BASE_URL}/api/v1/hackrx/run",
                json=request_data,
                headers=headers
            )
            
            print(f"📊 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                print("✅ SUCCESS! Response received:")
                print("-" * 40)
                
                answers = result.get("answers", [])
                metadata = result.get("metadata", {})
                
                print(f"📝 Total Answers: {len(answers)}")
                print(f"🔧 Processing Details: {metadata.get('processing_details', {})}")
                
                # Display first few answers
                for i, answer in enumerate(answers[:3], 1):
                    print(f"\nQ{i}: {request_data['questions'][i-1]}")
                    print(f"A{i}: {answer}")
                
                if len(answers) > 3:
                    print(f"\n... and {len(answers) - 3} more answers")
                
                # Save full response
                with open("api_test_response.json", "w") as f:
                    json.dump(result, f, indent=2)
                print("\n💾 Full response saved to 'api_test_response.json'")
                
            else:
                print(f"❌ ERROR: {response.status_code}")
                print(f"Response: {response.text}")
                
    except httpx.TimeoutException:
        print("⏰ Request timed out - the document processing might take longer")
    except httpx.ConnectError:
        print("🔌 Connection error - make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

async def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{API_BASE_URL}/api/v1/health")
            
            if response.status_code == 200:
                health_data = response.json()
                print("🏥 Health Check:")
                print(f"   Status: {health_data.get('status')}")
                print(f"   Components: {health_data.get('components')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

async def main():
    """Main test function"""
    print("🧪 API Client Test Script")
    print("=" * 30)
    
    # Test health endpoint first
    print("\n1. Testing health endpoint...")
    health_ok = await test_health_endpoint()
    
    if health_ok:
        print("\n2. Testing main API endpoint...")
        await test_hackrx_endpoint()
    else:
        print("\n⚠️  Server appears to be down. Start the server with:")
        print("   python main.py")
        print("   or")
        print("   uvicorn main:app --host 0.0.0.0 --port 8000 --reload")

if __name__ == "__main__":
    asyncio.run(main())
