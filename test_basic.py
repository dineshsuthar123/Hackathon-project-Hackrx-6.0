"""
Simplified test script to verify the system works
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_basic_setup():
    """Test basic setup without complex dependencies"""
    print("🎯 LLM-Powered Query-Retrieval System - Basic Test")
    print("=" * 50)
    
    # Check environment
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✅ OpenAI API Key found")
        print(f"   Key starts with: {api_key[:15]}...")
    else:
        print("❌ OpenAI API Key not found")
        return False
    
    # Test OpenAI connection
    try:
        print("\n🔧 Testing OpenAI connection...")
        import openai
        
        client = openai.AsyncOpenAI(api_key=api_key)
        
        # Simple test
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # Use cheaper model for testing
            messages=[
                {"role": "user", "content": "Say 'Hello, the system is working!'"}
            ],
            max_tokens=50
        )
        
        result = response.choices[0].message.content
        print(f"✅ OpenAI Response: {result}")
        
    except Exception as e:
        print(f"❌ OpenAI connection failed: {str(e)}")
        return False
    
    # Test FastAPI
    try:
        print("\n🚀 Testing FastAPI...")
        from fastapi import FastAPI
        app = FastAPI()
        
        @app.get("/")
        def test_endpoint():
            return {"status": "working", "message": "FastAPI is running!"}
        
        print("✅ FastAPI app created successfully")
        
    except Exception as e:
        print(f"❌ FastAPI test failed: {str(e)}")
        return False
    
    print("\n🎉 Basic system test completed successfully!")
    print("✅ Environment variables loaded")
    print("✅ OpenAI API connection working")
    print("✅ FastAPI framework ready")
    print("\nYou can now run the full server with:")
    print("   python simple_server.py")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_basic_setup())
