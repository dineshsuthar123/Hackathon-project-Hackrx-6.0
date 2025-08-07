"""
PROTOCOL 7.0: ReAct FRAMEWORK TEST SUITE
Demonstration of multi-step reasoning capabilities
"""

import asyncio
import logging
from react_reasoning import ReActReasoningEngine, PrecisionSearchTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample insurance document content
SAMPLE_DOCUMENT = """
CARE HEALTH INSURANCE LIMITED - POLICY DOCUMENT

SECTION 1: DEPENDENT ELIGIBILITY
1.1 Dependent children are covered from 3 months to 25 years of age.
1.2 Female child can be covered until she gets married, regardless of age.
1.3 Male child can be covered until 25 years of age.
1.4 Financially dependent children pursuing education can be covered beyond 25 years with proof.

SECTION 2: DENTAL COVERAGE
2.1 Dental treatment is covered only if it is due to an accident.
2.2 Routine dental care and cosmetic procedures are excluded.
2.3 Pre-approval is required for dental surgeries above Rs. 10,000.
2.4 Dental implants are covered up to Rs. 50,000 per policy year.

SECTION 3: CLAIM PROCEDURES
3.1 All claims must be submitted within 30 days of discharge.
3.2 For planned procedures, pre-authorization is required 48 hours in advance.
3.3 Emergency procedures must be reported within 24 hours.
3.4 Claims above Rs. 1 lakh require original bills and medical reports.

SECTION 4: PERSONAL DETAILS UPDATE
4.1 Name changes must be supported by legal documents (marriage certificate, gazette notification).
4.2 Address changes can be updated online or through customer service.
4.3 All updates require policy holder's written consent.
4.4 Changes take effect from the next policy renewal date.

SECTION 5: GRIEVANCE REDRESSAL
5.1 For complaints and grievances, contact our customer service.
5.2 E-mail: csd@orientalinsurance.co.in
5.3 Phone: 1800-11-8485 (toll-free)
5.4 Written complaints should be addressed to the Regional Manager.
5.5 If not satisfied, approach Insurance Ombudsman within 1 year.

SECTION 6: POLICY TERMS
6.1 Policy period is 12 months from the start date.
6.2 Grace period for premium payment is 30 days.
6.3 Policy can be renewed for lifetime.
6.4 Sum insured options: Rs. 1 lakh to Rs. 10 lakhs.
"""

async def test_react_framework():
    """Test the ReAct framework with complex queries"""
    
    logger.info("🧪 TESTING PROTOCOL 7.0: ReAct FRAMEWORK")
    logger.info("="*60)
    
    # Mock Groq client for testing
    class MockGroqClient:
        async def chat_completions_create(self, **kwargs):
            # Mock response for testing
            class MockResponse:
                def __init__(self, content):
                    self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': content})()})]
            return MockResponse("Mock reasoning response")
    
    # Initialize ReAct engine
    mock_client = MockGroqClient()
    react_engine = ReActReasoningEngine(mock_client, logger)
    
    # Test complex query
    complex_query = """While checking the process for submitting a dental claim for a 23-year-old 
    financially dependent daughter (who recently married), also confirm the process for updating 
    her last name and provide the company's grievance redressal email."""
    
    logger.info(f"🎯 COMPLEX QUERY: {complex_query}")
    logger.info("-"*60)
    
    try:
        # Test the ReAct reasoning process
        result = await react_engine.reason_and_act(SAMPLE_DOCUMENT, complex_query)
        
        logger.info(f"✅ REACT RESULT: {result}")
        logger.info("="*60)
        
        # Test individual components
        logger.info("🔧 TESTING INDIVIDUAL COMPONENTS:")
        logger.info("-"*40)
        
        # Test precision search tool
        search_tool = PrecisionSearchTool(SAMPLE_DOCUMENT, logger)
        
        test_queries = [
            "eligibility criteria for dependent daughter",
            "dental claim process", 
            "grievance redressal email",
            "name update process"
        ]
        
        for query in test_queries:
            search_result = await search_tool.search(query)
            logger.info(f"🔍 SEARCH '{query}': {search_result[:100]}...")
        
        logger.info("="*60)
        logger.info("🎉 PROTOCOL 7.0 TESTING COMPLETE")
        
    except Exception as e:
        logger.error(f"❌ TEST FAILED: {e}")

async def test_complexity_detection():
    """Test query complexity detection"""
    
    logger.info("🧪 TESTING COMPLEXITY DETECTION")
    logger.info("-"*40)
    
    # Mock engine for testing
    class MockEngine:
        def __init__(self):
            self.logger = logger
        
        def _detect_complex_query(self, question: str) -> bool:
            complexity_indicators = [
                'and', 'also', 'while', 'if', 'process for', 'both', 'first', 'then'
            ]
            
            question_lower = question.lower()
            complexity_score = sum(1 for indicator in complexity_indicators if indicator in question_lower)
            
            if len(question.split()) > 15:
                complexity_score += 1
            if question.count(',') > 1:
                complexity_score += 1
                
            return complexity_score >= 2
    
    engine = MockEngine()
    
    test_cases = [
        ("What is the waiting period for cataract?", False),
        ("What is the co-payment for a person aged 76?", False),
        ("While checking the dental claim process, also confirm the grievance email.", True),
        ("What are the eligibility criteria for dependent children and the claim submission process?", True),
        ("Given that a daughter is married, what is the process for updating her name and submitting claims?", True),
        ("Simple question about premium.", False)
    ]
    
    for query, expected in test_cases:
        result = engine._detect_complex_query(query)
        status = "✅" if result == expected else "❌"
        logger.info(f"{status} '{query[:50]}...' -> Complex: {result} (Expected: {expected})")
    
    logger.info("="*60)

if __name__ == "__main__":
    async def main():
        await test_complexity_detection()
        await test_react_framework()
    
    asyncio.run(main())
