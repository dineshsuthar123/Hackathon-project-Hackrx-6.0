"""
Test the corrected /hackrx/run endpoint
"""

import httpx
import asyncio
import json

async def test_hackrx_run():
    url = "http://localhost:8000/hackrx/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36"
    }
    
    data = {
        "documents": "https://example.com/test.pdf",
        "questions": [
            "What is the waiting period for Gout and Rheumatism?",
            "What is the grace period for premium payment?"
        ]
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")
            
            if response.status_code == 200:
                print("✅ /hackrx/run endpoint working correctly!")
            else:
                print(f"❌ Error: {response.status_code}")
                
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    asyncio.run(test_hackrx_run())
