"""
PROTOCOL 7.1/7.2 DEBUG ENDPOINT
Quick test for the Happy Family Floater document
"""

from fastapi import HTTPException
import time

async def test_happy_family_processing():
    """Test endpoint for debugging the Happy Family Floater document"""
    
    test_url = "https://hackrx.blob.core.windows.net/assets/Happy%20Family%20Floater%20-%202024%20OICHLIP25046V062425%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A24%3A30Z&se=2026-08-01T17%3A24%3A00Z&sr=b&sp=r&sig=VNMTTQUjdXGYb2F4Di4P0zNvmM2rTBoEHr%2BnkUXIqpQ%3D"
    test_question = "What is the full name of the insurance company issuing the 'EASY HEALTH' policy?"
    
    try:
        from app_groq_ultimate import groq_processor
        
        print("🧪 TESTING: Happy Family Floater Document Processing")
        print("=" * 60)
        
        # Step 1: Test document type detection
        print("🔍 Step 1: Document Type Detection")
        is_known = groq_processor._is_known_target(test_url)
        print(f"Result: {'KNOWN' if is_known else 'UNKNOWN'} target")
        
        # Step 2: Test document loading
        print("\n📥 Step 2: Document Loading")
        start_time = time.time()
        document_content = await groq_processor._get_clean_document_content(test_url)
        load_time = (time.time() - start_time) * 1000
        print(f"Content length: {len(document_content)} characters")
        print(f"Load time: {load_time:.1f}ms")
        
        if len(document_content) < 100:
            print("❌ ERROR: Document content too short")
            return
        
        # Step 3: Test single question processing
        print("\n🎯 Step 3: Single Question Processing")
        start_time = time.time()
        answer = await groq_processor._process_single_question_optimized(
            test_url, test_question, document_content
        )
        process_time = (time.time() - start_time) * 1000
        
        print(f"Answer: {answer}")
        print(f"Process time: {process_time:.1f}ms")
        
        print("\n✅ Test completed successfully")
        return answer
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise e

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_happy_family_processing())
