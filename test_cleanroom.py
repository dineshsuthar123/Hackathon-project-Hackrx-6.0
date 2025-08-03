"""
Phase 0 Clean Room Protocol Test
Testing robust PDF parsing and validation
"""

import asyncio
import logging
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app_cleanroom import clean_room_processor

async def test_clean_room_protocol():
    """Test Phase 0 Clean Room Protocol with robust PDF parsing"""
    
    logger.info("🧹 TESTING PHASE 0: CLEAN ROOM PROTOCOL")
    logger.info("=" * 60)
    
    # Test document URL
    document_url = "https://www.careinsurance.com/upload/brochures/Arogya%20Sanjeevani%20Policy%20-%20National%20(ASP-N).pdf"
    
    # Test questions that were causing corruption
    test_questions = [
        "What is the waiting period for Gout and Rheumatism?",
        "What is the grace period for premium payment?",
        "What is the waiting period for cataract treatment?",
        "What is the maximum coverage for ambulance expenses?",
        "What is the age limit for dependent children?"
    ]
    
    try:
        logger.info(f"📄 Processing document: {document_url}")
        logger.info(f"❓ Testing {len(test_questions)} questions...")
        
        # Process with Clean Room Protocol
        answers = await clean_room_processor.process_document_questions(
            document_url, test_questions
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("🎯 PHASE 0 RESULTS:")
        logger.info("=" * 60)
        
        for i, (question, answer) in enumerate(zip(test_questions, answers), 1):
            logger.info(f"\n❓ Question {i}: {question}")
            logger.info(f"✅ Answer {i}: {answer}")
            
            # Check for corruption indicators
            corruption_indicators = ['/Title', '/Author', '/Producer', 'endobj', '<<', '>>']
            has_corruption = any(indicator in answer for indicator in corruption_indicators)
            
            if has_corruption:
                logger.error(f"❌ CORRUPTION DETECTED in answer {i}!")
            else:
                logger.info(f"✅ Answer {i} is clean (no corruption)")
        
        # Overall assessment
        clean_answers = sum(1 for answer in answers if not any(
            indicator in answer for indicator in ['/Title', '/Author', '/Producer', 'endobj', '<<', '>>']
        ))
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 PHASE 0 ASSESSMENT:")
        logger.info(f"   Total Questions: {len(test_questions)}")
        logger.info(f"   Clean Answers: {clean_answers}")
        logger.info(f"   Corrupted Answers: {len(answers) - clean_answers}")
        logger.info(f"   Success Rate: {(clean_answers/len(answers))*100:.1f}%")
        
        if clean_answers == len(answers):
            logger.info("🎉 PHASE 0 CLEAN ROOM PROTOCOL: SUCCESS!")
            logger.info("   All answers are clean and human-readable")
        else:
            logger.error("❌ PHASE 0 CLEAN ROOM PROTOCOL: FAILED!")
            logger.error("   Some answers still contain corruption")
        
        return answers
        
    except Exception as e:
        logger.error(f"❌ Phase 0 test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(test_clean_room_protocol())
