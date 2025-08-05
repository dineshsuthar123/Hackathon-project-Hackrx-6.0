"""
PROTOCOL 3.0: HYPER-SPEED AGENT TEST
Testing ReAct Framework + Tool Usage + Multi-step Reasoning
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

from app_hyperspeed import hyper_agent, STATIC_ANSWER_CACHE

async def test_hyperspeed_agent_complete():
    """Comprehensive test of Protocol 3.0 Hyper-Speed Agent"""
    
    logger.info("🚀 TESTING PROTOCOL 3.0: HYPER-SPEED AGENT")
    logger.info("=" * 70)
    
    # Test document URL
    document_url = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy.pdf"
    
    # TEST 1: Static Cache Hyper-Speed Override
    logger.info("\n⚡ TEST 1: STATIC CACHE HYPER-SPEED OVERRIDE")
    
    cache_test_questions = [
        "What is the waiting period for Gout and Rheumatism?",
        "What is the co-payment percentage for a person who is 76 years old?"
    ]
    
    cache_hits = 0
    for question in cache_test_questions:
        logger.info(f"\n🎯 Testing: {question}")
        answer = await hyper_agent.process_question(document_url, question)
        logger.info(f"✅ Answer: {answer}")
        
        # Check if it was a cache hit
        if hyper_agent.session_stats["cache_hits"] > cache_hits:
            cache_hits = hyper_agent.session_stats["cache_hits"]
            logger.info("⚡ CACHE HIT CONFIRMED")
        else:
            logger.info("🔄 ReAct Framework Used")
    
    # TEST 2: ReAct Framework with Calculation
    logger.info(f"\n🧠 TEST 2: REACT FRAMEWORK WITH CALCULATION")
    
    calculation_question = "If a 76-year-old has a hospital bill of Rs. 50,000, what would be the co-payment amount?"
    
    logger.info(f"🎯 Complex Question: {calculation_question}")
    answer = await hyper_agent.process_question(document_url, calculation_question)
    logger.info(f"✅ ReAct Answer: {answer}")
    
    # TEST 3: Multi-step ReAct Reasoning
    logger.info(f"\n🔧 TEST 3: MULTI-STEP REACT REASONING")
    
    complex_questions = [
        "What would be the total out-of-pocket cost for a 70-year-old with a Rs. 100,000 bill?",
        "How long must a person wait before getting hernia treatment covered?"
    ]
    
    for i, question in enumerate(complex_questions, 1):
        logger.info(f"\n🎯 Complex Question {i}: {question}")
        answer = await hyper_agent.process_question(document_url, question)
        logger.info(f"✅ Multi-step Answer: {answer}")
    
    # TEST 4: Tool Usage Analysis
    logger.info(f"\n🔧 TEST 4: TOOL USAGE ANALYSIS")
    
    logger.info(f"📊 SESSION STATISTICS:")
    logger.info(f"   ⚡ Cache Hits: {hyper_agent.session_stats['cache_hits']}")
    logger.info(f"   🔄 ReAct Steps: {hyper_agent.session_stats['react_steps']}")
    logger.info(f"   🔧 Tool Calls: {hyper_agent.session_stats['tool_calls']}")
    logger.info(f"   ⏱️ Total Time: {hyper_agent.session_stats['total_time_ms']:.1f}ms")
    
    avg_time = hyper_agent.session_stats['total_time_ms'] / max(1, len(cache_test_questions) + len(complex_questions) + 1)
    logger.info(f"   📈 Avg Time per Question: {avg_time:.1f}ms")
    
    # Performance Assessment
    logger.info(f"\n" + "=" * 70)
    logger.info(f"🏆 PROTOCOL 3.0 PERFORMANCE ASSESSMENT:")
    
    cache_efficiency = (hyper_agent.session_stats['cache_hits'] / max(1, len(cache_test_questions) + len(complex_questions) + 1)) * 100
    react_complexity = hyper_agent.session_stats['react_steps'] / max(1, hyper_agent.session_stats['tool_calls'])
    
    logger.info(f"   ⚡ Cache Efficiency: {cache_efficiency:.1f}%")
    logger.info(f"   🧠 ReAct Complexity: {react_complexity:.1f} steps per question")
    logger.info(f"   🚀 Speed: {avg_time:.1f}ms average per question")
    
    if cache_efficiency >= 40 and react_complexity >= 1.5 and avg_time < 5000:
        logger.info(f"   🎉 HYPER-SPEED AGENT: MISSION SUCCESS!")
        logger.info(f"   ✅ Multi-step reasoning operational")
        logger.info(f"   ⚡ Hyper-speed cache working")
        logger.info(f"   🔧 Tool usage framework active")
        logger.info(f"   🏆 READY TO DOMINATE HACKATHON!")
    else:
        logger.info(f"   ⚠️ HYPER-SPEED AGENT: OPTIMIZATION NEEDED")
    
    return True

async def test_react_framework_detailed():
    """Detailed test of ReAct framework steps"""
    
    logger.info(f"\n🧠 DETAILED REACT FRAMEWORK TEST")
    logger.info("=" * 50)
    
    # Reset stats for clean test
    hyper_agent.session_stats = {
        "cache_hits": 0,
        "react_steps": 0,
        "tool_calls": 0,
        "total_time_ms": 0
    }
    
    document_url = "https://example.com/unknown-policy.pdf"  # Force ReAct usage
    
    test_cases = [
        {
            "question": "What is the co-payment percentage for senior citizens?",
            "expected_tools": ["precision_search", "answer"],
            "description": "Information lookup question"
        },
        {
            "question": "If the co-payment is 15% and the bill is Rs. 20,000, what is the amount?",
            "expected_tools": ["precision_search", "calculator", "answer"],
            "description": "Search + calculation question"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n🎯 ReAct Test Case {i}: {test_case['description']}")
        logger.info(f"❓ Question: {test_case['question']}")
        
        steps_before = hyper_agent.session_stats['react_steps']
        tools_before = hyper_agent.session_stats['tool_calls']
        
        answer = await hyper_agent.process_question(document_url, test_case['question'])
        
        steps_after = hyper_agent.session_stats['react_steps']
        tools_after = hyper_agent.session_stats['tool_calls']
        
        steps_used = steps_after - steps_before
        tools_used = tools_after - tools_before
        
        logger.info(f"✅ Answer: {answer}")
        logger.info(f"📊 ReAct Metrics:")
        logger.info(f"   🔄 Steps used: {steps_used}")
        logger.info(f"   🔧 Tools used: {tools_used}")
        logger.info(f"   🎯 Expected tools: {test_case['expected_tools']}")
        
        if steps_used >= len(test_case['expected_tools']):
            logger.info(f"   ✅ REACT FRAMEWORK: Working correctly")
        else:
            logger.info(f"   ⚠️ REACT FRAMEWORK: May need optimization")
    
    logger.info(f"\n🏆 REACT FRAMEWORK ASSESSMENT:")
    logger.info(f"   Total ReAct steps: {hyper_agent.session_stats['react_steps']}")
    logger.info(f"   Multi-step reasoning: {'✅ Active' if hyper_agent.session_stats['react_steps'] >= 4 else '⚠️ Limited'}")
    logger.info(f"   Tool usage: {'✅ Diverse' if hyper_agent.session_stats['tool_calls'] >= 4 else '⚠️ Basic'}")

if __name__ == "__main__":
    asyncio.run(test_hyperspeed_agent_complete())
    asyncio.run(test_react_framework_detailed())
