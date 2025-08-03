"""
MISSION 10/10 SYSTEM TEST
Testing all three phases of the precision protocol
"""

import asyncio
import sys
import logging
from typing import List

# Test the mission system
sys.path.append('.')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_mission_10():
    """Test the complete 10/10 mission system"""
    try:
        from app_mission10 import Mission10_DocumentProcessor
        
        processor = Mission10_DocumentProcessor()
        
        # Test questions from the hackathon
        test_questions = [
            "What is the waiting period for Gout and Rheumatism?",
            "What is the waiting period for cataract treatment?", 
            "What is the maximum amount covered for road ambulance expenses per hospitalization?",
            "What is the grace period for premium payment?",
            "What is the age range for dependent children coverage?"
        ]
        
        # Expected answers for validation
        expected_answers = [
            "36 months",
            "24 months", 
            "Rs. 2,000",
            "30 days",
            "3 months to 25 years"
        ]
        
        logger.info("🚀 MISSION 10/10 TEST: Starting complete protocol test")
        
        # Use fallback document for testing
        document_url = "https://example.com/test.pdf"  # Will use fallback content
        
        # Execute mission
        answers = await processor.execute_mission(document_url, test_questions)
        
        # Validate results
        logger.info("\n" + "="*80)
        logger.info("MISSION 10/10 RESULTS")
        logger.info("="*80)
        
        success_count = 0
        for i, (question, answer, expected) in enumerate(zip(test_questions, answers, expected_answers), 1):
            is_correct = expected.lower() in answer.lower()
            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            
            if is_correct:
                success_count += 1
            
            logger.info(f"\nQUESTION {i}: {question}")
            logger.info(f"ANSWER: {answer}")
            logger.info(f"EXPECTED: {expected}")
            logger.info(f"STATUS: {status}")
        
        # Final score
        accuracy = (success_count / len(test_questions)) * 100
        logger.info(f"\n" + "="*80)
        logger.info(f"MISSION SCORE: {success_count}/{len(test_questions)} ({accuracy:.1f}%)")
        
        if accuracy >= 90:
            logger.info("🎯 MISSION SUCCESS: 10/10 ACHIEVED!")
        elif accuracy >= 70:
            logger.info("⚠️ MISSION PARTIAL: Need optimization")
        else:
            logger.info("❌ MISSION FAILED: Protocol needs revision")
        
        logger.info("="*80)
        
        return answers
        
    except Exception as e:
        logger.error(f"❌ MISSION TEST FAILED: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(test_mission_10())
