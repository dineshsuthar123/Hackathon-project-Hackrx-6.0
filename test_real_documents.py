#!/usr/bin/env python3
"""
Real Document Testing Suite
Tests the system with actual PDF, DOCX, and online documents
"""

import asyncio
import httpx
import json
import time
import os
from typing import List, Dict, Any
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8002"  # Update if using different port
API_TOKEN = "a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36"

# Real document URLs for testing
TEST_DOCUMENTS = {
    "insurance_policy": {
        "url": "https://www.nationalinsurance.nic.co.in/sites/default/files/2024-11/National%20Parivar%20Mediclaim%20Plus%20Policy%20Wording.pdf",
        "type": "Insurance Policy",
        "questions": [
            "What is the grace period for premium payment?",
            "Does this policy cover maternity expenses?",
            "What is the waiting period for pre-existing diseases?",
            "Are AYUSH treatments covered?",
            "What is the maximum sum insured available?",
            "How do I file a claim?",
            "What is excluded from coverage?",
            "Is there a no-claim discount?",
            "What documents are required for claims?",
            "Can I renew this policy online?"
        ]
    },
    "legal_document": {
        "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019323000077/aapl-20230930.htm",
        "type": "SEC Filing (Apple)",
        "questions": [
            "What was Apple's total revenue?",
            "What are the main risk factors?",
            "What is the company's cash position?",
            "Who are the key executives?",
            "What are the major business segments?"
        ]
    },
    "research_paper": {
        "url": "https://arxiv.org/pdf/2005.14165.pdf",
        "type": "Research Paper (GPT-3)",
        "questions": [
            "What is the main contribution of this paper?",
            "How many parameters does GPT-3 have?",
            "What datasets were used for training?",
            "What are the key findings?",
            "How does it compare to previous models?"
        ]
    }
}

class DocumentTester:
    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url
        self.api_token = api_token
        self.results = []
    
    async def test_document(self, doc_key: str, doc_info: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single document with its questions"""
        print(f"\n📄 Testing: {doc_info['type']}")
        print(f"🔗 URL: {doc_info['url'][:80]}...")
        print(f"❓ Questions: {len(doc_info['questions'])}")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "documents": doc_info["url"],
                "questions": doc_info["questions"]
            }
            
            start_time = time.time()
            
            try:
                response = await client.post(
                    f"{self.base_url}/hackrx/run",
                    json=payload,
                    headers=headers
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    answers = result.get("answers", [])
                    
                    print(f"✅ Success! Got {len(answers)} answers in {response_time:.2f}s")
                    
                    # Analyze answer quality
                    quality_scores = [self._analyze_answer_quality(q, a) for q, a in zip(doc_info["questions"], answers)]
                    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
                    
                    test_result = {
                        "document_type": doc_info["type"],
                        "url": doc_info["url"],
                        "status": "success",
                        "response_time": response_time,
                        "questions_count": len(doc_info["questions"]),
                        "answers_count": len(answers),
                        "average_quality": avg_quality,
                        "qa_pairs": [
                            {
                                "question": q,
                                "answer": a,
                                "quality_score": s
                            }
                            for q, a, s in zip(doc_info["questions"], answers, quality_scores)
                        ]
                    }
                    
                    # Display sample Q&A
                    print(f"📊 Average Quality: {avg_quality:.2f}/5")
                    print("\n🔍 Sample Q&A:")
                    for i, (q, a) in enumerate(zip(doc_info["questions"][:3], answers[:3]), 1):
                        print(f"  Q{i}: {q}")
                        print(f"  A{i}: {a[:100]}{'...' if len(a) > 100 else ''}")
                        print()
                    
                    return test_result
                    
                else:
                    print(f"❌ Failed with status {response.status_code}")
                    print(f"Response: {response.text}")
                    return {
                        "document_type": doc_info["type"],
                        "url": doc_info["url"],
                        "status": "failed",
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "response_time": response_time
                    }
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                return {
                    "document_type": doc_info["type"],
                    "url": doc_info["url"],
                    "status": "error",
                    "error": str(e),
                    "response_time": time.time() - start_time
                }
    
    def _analyze_answer_quality(self, question: str, answer: str) -> float:
        """Analyze answer quality on a scale of 1-5"""
        if not answer or len(answer.strip()) < 10:
            return 1.0
        
        # Check for generic/error responses
        if any(phrase in answer.lower() for phrase in [
            "cannot find", "not available", "error", "unable to", "please refer"
        ]):
            return 2.0
        
        # Check for specific information
        score = 3.0  # Base score for meaningful answer
        
        # Boost score for specific information
        if any(indicator in answer.lower() for indicator in [
            "yes", "no", "₹", "$", "%", "days", "months", "years", 
            "covered", "excluded", "required", "available"
        ]):
            score += 1.0
        
        # Boost for detailed answers
        if len(answer) > 100:
            score += 0.5
        
        return min(5.0, score)
    
    async def run_comprehensive_test(self):
        """Run tests on all document types"""
        print("🧪 COMPREHENSIVE DOCUMENT TESTING SUITE")
        print("=" * 60)
        print(f"🎯 Testing {len(TEST_DOCUMENTS)} different document types")
        print(f"🚀 API Endpoint: {self.base_url}/hackrx/run")
        print()
        
        overall_start = time.time()
        
        for doc_key, doc_info in TEST_DOCUMENTS.items():
            result = await self.test_document(doc_key, doc_info)
            self.results.append(result)
            
            # Add delay between tests
            await asyncio.sleep(2)
        
        overall_time = time.time() - overall_start
        
        # Generate comprehensive report
        self._generate_report(overall_time)
        
        # Save detailed results
        self._save_results()
    
    def _generate_report(self, total_time: float):
        """Generate a comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        
        successful_tests = [r for r in self.results if r.get("status") == "success"]
        failed_tests = [r for r in self.results if r.get("status") != "success"]
        
        print(f"📈 Overall Statistics:")
        print(f"  • Total Documents Tested: {len(self.results)}")
        print(f"  • Successful Tests: {len(successful_tests)}")
        print(f"  • Failed Tests: {len(failed_tests)}")
        print(f"  • Success Rate: {(len(successful_tests)/len(self.results)*100):.1f}%")
        print(f"  • Total Test Time: {total_time:.2f} seconds")
        
        if successful_tests:
            avg_response_time = sum(r.get("response_time", 0) for r in successful_tests) / len(successful_tests)
            avg_quality = sum(r.get("average_quality", 0) for r in successful_tests) / len(successful_tests)
            total_questions = sum(r.get("questions_count", 0) for r in successful_tests)
            total_answers = sum(r.get("answers_count", 0) for r in successful_tests)
            
            print(f"  • Average Response Time: {avg_response_time:.2f}s")
            print(f"  • Average Answer Quality: {avg_quality:.2f}/5")
            print(f"  • Total Questions Processed: {total_questions}")
            print(f"  • Total Answers Generated: {total_answers}")
        
        print(f"\n📋 Detailed Results by Document Type:")
        for result in self.results:
            status_icon = "✅" if result.get("status") == "success" else "❌"
            print(f"  {status_icon} {result['document_type']}")
            if result.get("status") == "success":
                print(f"     Response Time: {result.get('response_time', 0):.2f}s")
                print(f"     Quality Score: {result.get('average_quality', 0):.2f}/5")
                print(f"     Q&A Pairs: {result.get('questions_count', 0)}")
            else:
                print(f"     Error: {result.get('error', 'Unknown error')}")
        
        print(f"\n💡 Recommendations:")
        if len(successful_tests) == len(self.results):
            print("  🎉 Excellent! All document types processed successfully")
            print("  🚀 System is ready for production deployment")
        elif len(successful_tests) > 0:
            print("  ⚠️  Some document types failed - check network connectivity")
            print("  🔧 Consider adding retry logic for failed documents")
        else:
            print("  🚨 All tests failed - check server status and configuration")
        
        print(f"\n📁 Detailed results saved to: document_test_results.json")
    
    def _save_results(self):
        """Save detailed test results to JSON file"""
        output = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_summary": {
                "total_documents": len(self.results),
                "successful_tests": len([r for r in self.results if r.get("status") == "success"]),
                "failed_tests": len([r for r in self.results if r.get("status") != "success"])
            },
            "detailed_results": self.results
        }
        
        with open("document_test_results.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

async def test_local_documents():
    """Test with local documents if available"""
    print("\n📁 TESTING LOCAL DOCUMENTS")
    print("-" * 40)
    
    local_docs = []
    extensions = ['.pdf', '.docx', '.txt']
    
    # Look for documents in common directories
    search_dirs = [
        Path.cwd(),
        Path.cwd() / "documents",
        Path.cwd() / "test_documents",
        Path.home() / "Documents",
        Path.home() / "Downloads"
    ]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            for ext in extensions:
                docs = list(search_dir.glob(f"*{ext}"))
                local_docs.extend(docs[:3])  # Limit to 3 per directory
    
    if local_docs:
        print(f"Found {len(local_docs)} local documents:")
        for doc in local_docs[:5]:  # Show first 5
            print(f"  📄 {doc.name}")
        print("\n💡 To test local documents:")
        print("   1. Upload to a web server or cloud storage")
        print("   2. Use the public URL in your test")
        print("   3. Or modify the test to use local file paths")
    else:
        print("No local documents found in common directories")
        print("💡 Add PDF/DOCX files to test with local documents")

async def main():
    """Main test function"""
    print("🎯 REAL DOCUMENT TESTING SUITE")
    print("Testing the LLM-Powered Query-Retrieval System")
    print("=" * 60)
    
    # Initialize tester
    tester = DocumentTester(BASE_URL, API_TOKEN)
    
    # Test server connectivity first
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{BASE_URL}/")
            if response.status_code == 200:
                print("✅ Server is running and accessible")
            else:
                print(f"⚠️  Server responded with status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure the server is running with: python production_server.py")
        return
    
    # Run comprehensive tests
    await tester.run_comprehensive_test()
    
    # Test local documents
    await test_local_documents()
    
    print(f"\n🎉 Testing completed! Check document_test_results.json for details.")

if __name__ == "__main__":
    asyncio.run(main())
