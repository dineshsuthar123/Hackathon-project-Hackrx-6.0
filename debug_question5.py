"""
DEBUG: Question 5 Full Pipeline - Why ambulance answer for age question?
"""

import sys
sys.path.append('.')

import asyncio
import logging
from app_mission10 import Mission10_DocumentProcessor

logging.basicConfig(level=logging.DEBUG)

async def debug_question_5():
    processor = Mission10_DocumentProcessor()
    
    # Setup with fallback content
    chunks = processor.phase1.create_semantic_chunks(processor._get_fallback_content())
    chunks = processor.phase1.build_hybrid_index(chunks)
    
    query = "What is the age range for dependent children coverage?"
    
    print("=== QUESTION 5 FULL PIPELINE DEBUG ===")
    print(f"Query: {query}")
    print()
    
    # Phase 2: Precision Search
    print("=== PHASE 2: PRECISION RETRIEVAL ===")
    top_chunks = processor.phase2.precision_retrieval(query, chunks)
    
    print(f"Retrieved {len(top_chunks)} chunks:")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"\nChunk {i} (Score: {chunk.rerank_score:.3f}):")
        print(f"Content: {chunk.content[:200]}...")
        print(f"Section: {chunk.section_type}")
    
    # Phase 3: Answer Generation
    print("\n=== PHASE 3: ANSWER GENERATION ===")
    context = "\n\n".join([f"CHUNK {i+1}: {chunk.content}" for i, chunk in enumerate(top_chunks)])
    
    print(f"Combined context length: {len(context)} chars")
    print(f"Context preview: {context[:300]}...")
    
    # Test extraction directly
    answer = processor.phase3._zero_shot_extraction(query, context)
    print(f"\nExtracted answer: {answer}")
    
    # Test validation
    is_valid = processor.phase3._validate_answer_quality(query, answer, context)
    print(f"Answer valid: {is_valid}")

if __name__ == "__main__":
    asyncio.run(debug_question_5())
