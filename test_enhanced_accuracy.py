#!/usr/bin/env python3
"""
Enhanced Accuracy Testing Suite
Comprehensive testing of the improved system accuracy
"""

import asyncio
import httpx
import json
import time
import statistics
from typing import List, Dict, Any
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
API_TOKEN = "a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36"

# Enhanced test questions with expected answer types
ENHANCED_TEST_QUESTIONS = {
    "insurance_policy": {
        "url": "https://www.nationalinsurance.nic.co.in/sites/default/files/2024-11/National%20Parivar%20Mediclaim%20Plus%20Policy%20Wording.pdf",
        "questions": [
            {
                "question": "What is the exact grace period for premium payment?",
                "expected_type": "temporal",
                "expected_specificity": "high",
                "keywords": ["30 days", "thirty days", "grace period"]
            },
            {
                "question": "Does this policy cover maternity expenses and what are the specific conditions?",
                "expected_type": "coverage",
                "expected_specificity": "high",
                "keywords": ["maternity", "24 months", "continuous coverage"]
            },
            {
                "question": "What is the waiting period for pre-existing diseases?",
                "expected_type": "temporal",
                "expected_specificity": "high",
                "keywords": ["36 months", "thirty-six months", "pre-existing"]
            },
            {
                "question": "Are AYUSH treatments covered under this policy?",
                "expected_type": "coverage",
                "expected_specificity": "medium",
                "keywords": ["AYUSH", "covered", "Ayurveda", "Yoga", "Naturopathy"]
            },
            {
                "question": "What is the no claim discount percentage offered?",
                "expected_type": "financial",
                "expected_specificity": "high",
                "keywords": ["5%", "no claim discount", "NCD"]
            },
            {
                "question": "What are the room rent limits for Plan A?",
                "expected_type": "financial",
                "expected_specificity": "high",
                "keywords": ["1%", "room rent", "Plan A", "sum insured"]
            },
            {
                "question": "How is a hospital defined in this policy?",
                "expected_type": "definition",
                "expected_specificity": "high",
                "keywords": ["10 beds", "15 beds", "qualified nursing", "operation theatre"]
            },
            {
                "question": "What is the waiting period for cataract surgery?",
                "expected_type": "temporal",
                "expected_specificity": "high",
                "keywords": ["2 years", "two years", "cataract"]
            },
            {
                "question": "Are organ donor expenses covered?",
                "expected_type": "coverage",
                "expected_specificity": "medium",
                "keywords": ["organ donor", "covered", "harvesting"]
            },
            {
                "question": "What documents are required for claim filing?",
                "expected_type": "procedure",
                "expected_specificity": "medium",
                "keywords": ["documents", "claim", "required"]
            }
        ]
    }
}

class EnhancedAccuracyTester:
    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url
        self.api_token = api_token
        self.results = []
        
    async def test_enhanced_accuracy(self):
        """Run comprehensive enhanced accuracy tests"""
        print("🧪 ENHANCED ACCURACY TESTING SUITE")
        print("=" * 60)
        print("🎯 Testing maximum accuracy features:")
        print("  • Advanced document processing")
        print("  • Enhanced semantic search")
        print("  • Sophisticated query analysis")
        print("  • Domain-specific prompting")
        print("  • Response validation")
        print()
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            
            # Test 1: System health and enhanced features
            await self._test_enhanced_features(client)
            
            # Test 2: Enhanced accuracy with real documents
            await self._test_document_accuracy(client)
            
            # Test 3: Query analysis accuracy
            await self._test_query_analysis_accuracy(client)
            
            # Test 4: Response quality metrics
            await self._test_response_quality(client)
            
            # Generate comprehensive report
            self._generate_accuracy_report()
    
    async def _test_enhanced_features(self, client: httpx.AsyncClient):
        """Test enhanced features availability"""
        print("1. Testing Enhanced Features Availability")
        print("-" * 40)
        
        try:
            response = await client.get(f"{self.base_url}/api/v1/health")
            
            if response.status_code == 200:
                health_data = response.json()
                enhanced_features = health_data.get('enhanced_features', False)
                accuracy_features = health_data.get('accuracy_features', {})
                
                print(f"✅ Enhanced features enabled: {enhanced_features}")
                
                for feature, status in accuracy_features.items():
                    status_icon = "✅" if status else "❌"
                    print(f"  {status_icon} {feature.replace('_', ' ').title()}: {status}")
                
                if enhanced_features:
                    print("🎉 All enhanced accuracy features are active!")
                else:
                    print("⚠️  Enhanced features not fully available")
                
                return enhanced_features
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Enhanced features test failed: {e}")
            return False
    
    async def _test_document_accuracy(self, client: httpx.AsyncClient):
        """Test document processing accuracy"""
        print("\n2. Testing Document Processing Accuracy")
        print("-" * 40)
        
        for doc_type, doc_info in ENHANCED_TEST_QUESTIONS.items():
            print(f"\n📄 Testing: {doc_type.replace('_', ' ').title()}")
            
            # Test with enhanced accuracy mode
            await self._test_accuracy_mode(client, doc_info, "maximum")
            
            # Compare with standard mode
            await self._test_accuracy_mode(client, doc_info, "standard")
    
    async def _test_accuracy_mode(self, client: httpx.AsyncClient, doc_info: Dict[str, Any], accuracy_mode: str):
        """Test specific accuracy mode"""
        print(f"\n🔍 Testing {accuracy_mode} accuracy mode...")
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "documents": doc_info["url"],
            "questions": [q["question"] for q in doc_info["questions"]],
            "accuracy_mode": accuracy_mode,
            "domain_hint": "insurance"
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
                metadata = result.get("metadata", {})
                accuracy_metrics = result.get("accuracy_metrics", {})
                
                print(f"✅ {accuracy_mode.title()} mode successful!")
                print(f"   Response time: {response_time:.2f}s")
                print(f"   Questions processed: {len(answers)}")
                
                # Analyze answer quality
                quality_scores = []
                for i, (question_info, answer) in enumerate(zip(doc_info["questions"], answers)):
                    quality_score = self._analyze_answer_quality(question_info, answer)
                    quality_scores.append(quality_score)
                    
                    if quality_score >= 4:
                        print(f"   Q{i+1}: ⭐ High quality ({quality_score}/5)")
                    elif quality_score >= 3:
                        print(f"   Q{i+1}: ✅ Good quality ({quality_score}/5)")
                    else:
                        print(f"   Q{i+1}: ⚠️  Needs improvement ({quality_score}/5)")
                
                avg_quality = statistics.mean(quality_scores) if quality_scores else 0
                print(f"   Average quality: {avg_quality:.2f}/5")
                
                # Store results for comparison
                test_result = {
                    "accuracy_mode": accuracy_mode,
                    "response_time": response_time,
                    "questions_count": len(answers),
                    "average_quality": avg_quality,
                    "quality_scores": quality_scores,
                    "accuracy_metrics": accuracy_metrics,
                    "enhanced_features_used": metadata.get("enhanced_features_used", False)
                }
                
                self.results.append(test_result)
                
            else:
                print(f"❌ {accuracy_mode.title()} mode failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ {accuracy_mode.title()} mode error: {e}")
    
    def _analyze_answer_quality(self, question_info: Dict[str, Any], answer: str) -> float:
        """Analyze answer quality based on expected criteria"""
        if not answer or len(answer.strip()) < 10:
            return 1.0
        
        score = 2.0  # Base score
        answer_lower = answer.lower()
        
        # Check for expected keywords
        expected_keywords = question_info.get("keywords", [])
        keyword_matches = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
        
        if keyword_matches > 0:
            score += min(2.0, keyword_matches * 0.5)
        
        # Check specificity
        expected_specificity = question_info.get("expected_specificity", "medium")
        
        # High specificity questions should have numbers, percentages, or specific terms
        if expected_specificity == "high":
            if any(char.isdigit() for char in answer) or '%' in answer or any(
                term in answer_lower for term in ["days", "months", "years", "inr", "rupees"]
            ):
                score += 0.5
        
        # Check answer type alignment
        expected_type = question_info.get("expected_type", "general")
        
        if expected_type == "temporal" and any(
            term in answer_lower for term in ["days", "months", "years", "period", "time"]
        ):
            score += 0.3
        elif expected_type == "financial" and any(
            term in answer_lower for term in ["inr", "rupees", "%", "amount", "cost", "premium"]
        ):
            score += 0.3
        elif expected_type == "coverage" and any(
            term in answer_lower for term in ["covered", "coverage", "yes", "no", "benefit"]
        ):
            score += 0.3
        elif expected_type == "definition" and any(
            term in answer_lower for term in ["defined", "means", "refers to", "is"]
        ):
            score += 0.3
        
        # Penalty for generic responses
        if any(phrase in answer_lower for phrase in [
            "cannot find", "not available", "please refer", "contact", "consult"
        ]):
            score -= 1.0
        
        return min(5.0, max(1.0, score))
    
    async def _test_query_analysis_accuracy(self, client: httpx.AsyncClient):
        """Test query analysis accuracy"""
        print("\n3. Testing Query Analysis Accuracy")
        print("-" * 40)
        
        # This would require access to the query analysis endpoint
        # For now, we'll analyze based on response metadata
        print("📊 Query analysis accuracy assessed through response metadata")
        
        if self.results:
            enhanced_results = [r for r in self.results if r["accuracy_mode"] == "maximum"]
            if enhanced_results:
                avg_metrics = enhanced_results[0].get("accuracy_metrics", {})
                per_question_metrics = avg_metrics.get("per_question_metrics", [])
                
                if per_question_metrics:
                    query_confidences = [
                        m.get("query_analysis", {}).get("confidence", 0) 
                        for m in per_question_metrics
                    ]
                    
                    if query_confidences:
                        avg_confidence = statistics.mean(query_confidences)
                        print(f"✅ Average query analysis confidence: {avg_confidence:.2f}")
                        
                        high_confidence_queries = sum(1 for c in query_confidences if c > 0.8)
                        print(f"✅ High confidence queries: {high_confidence_queries}/{len(query_confidences)}")
    
    async def _test_response_quality(self, client: httpx.AsyncClient):
        """Test response quality metrics"""
        print("\n4. Testing Response Quality Metrics")
        print("-" * 40)
        
        if not self.results:
            print("❌ No results available for quality analysis")
            return
        
        # Compare enhanced vs standard modes
        enhanced_results = [r for r in self.results if r["accuracy_mode"] == "maximum"]
        standard_results = [r for r in self.results if r["accuracy_mode"] == "standard"]
        
        if enhanced_results and standard_results:
            enhanced_quality = statistics.mean([r["average_quality"] for r in enhanced_results])
            standard_quality = statistics.mean([r["average_quality"] for r in standard_results])
            
            improvement = ((enhanced_quality - standard_quality) / standard_quality) * 100
            
            print(f"📊 Quality Comparison:")
            print(f"   Enhanced mode: {enhanced_quality:.2f}/5")
            print(f"   Standard mode: {standard_quality:.2f}/5")
            print(f"   Improvement: {improvement:+.1f}%")
            
            if improvement > 10:
                print("🎉 Significant quality improvement achieved!")
            elif improvement > 5:
                print("✅ Noticeable quality improvement")
            else:
                print("⚠️  Marginal quality difference")
        
        # Response time analysis
        enhanced_times = [r["response_time"] for r in enhanced_results]
        standard_times = [r["response_time"] for r in standard_results]
        
        if enhanced_times and standard_times:
            enhanced_avg_time = statistics.mean(enhanced_times)
            standard_avg_time = statistics.mean(standard_times)
            
            print(f"\n⏱️  Performance Comparison:")
            print(f"   Enhanced mode: {enhanced_avg_time:.2f}s")
            print(f"   Standard mode: {standard_avg_time:.2f}s")
            
            if enhanced_avg_time < standard_avg_time * 1.5:
                print("✅ Enhanced mode maintains good performance")
            else:
                print("⚠️  Enhanced mode is slower but more accurate")
    
    def _generate_accuracy_report(self):
        """Generate comprehensive accuracy report"""
        print("\n" + "=" * 60)
        print("📊 ENHANCED ACCURACY TEST REPORT")
        print("=" * 60)
        
        if not self.results:
            print("❌ No test results available")
            return
        
        # Overall statistics
        total_tests = len(self.results)
        enhanced_tests = len([r for r in self.results if r["accuracy_mode"] == "maximum"])
        standard_tests = len([r for r in self.results if r["accuracy_mode"] == "standard"])
        
        print(f"📈 Test Summary:")
        print(f"  • Total tests run: {total_tests}")
        print(f"  • Enhanced mode tests: {enhanced_tests}")
        print(f"  • Standard mode tests: {standard_tests}")
        
        # Quality analysis
        if self.results:
            all_quality_scores = []
            for result in self.results:
                all_quality_scores.extend(result.get("quality_scores", []))
            
            if all_quality_scores:
                avg_quality = statistics.mean(all_quality_scores)
                high_quality_count = sum(1 for score in all_quality_scores if score >= 4)
                
                print(f"\n📊 Quality Metrics:")
                print(f"  • Average answer quality: {avg_quality:.2f}/5")
                print(f"  • High quality answers: {high_quality_count}/{len(all_quality_scores)} ({(high_quality_count/len(all_quality_scores)*100):.1f}%)")
        
        # Performance metrics
        response_times = [r["response_time"] for r in self.results]
        if response_times:
            avg_time = statistics.mean(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            print(f"\n⏱️  Performance Metrics:")
            print(f"  • Average response time: {avg_time:.2f}s")
            print(f"  • Fastest response: {min_time:.2f}s")
            print(f"  • Slowest response: {max_time:.2f}s")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        enhanced_results = [r for r in self.results if r["accuracy_mode"] == "maximum"]
        if enhanced_results:
            avg_enhanced_quality = statistics.mean([r["average_quality"] for r in enhanced_results])
            
            if avg_enhanced_quality >= 4.0:
                print("  🎉 Excellent! Enhanced mode provides high-quality responses")
                print("  🚀 System is ready for production with maximum accuracy")
            elif avg_enhanced_quality >= 3.5:
                print("  ✅ Good performance with enhanced mode")
                print("  🔧 Consider fine-tuning for specific domains")
            else:
                print("  ⚠️  Enhanced mode needs improvement")
                print("  🔧 Review document processing and query analysis")
        
        # Save detailed results
        self._save_detailed_results()
    
    def _save_detailed_results(self):
        """Save detailed test results"""
        output = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_type": "enhanced_accuracy_testing",
            "total_tests": len(self.results),
            "results": self.results,
            "summary": {
                "enhanced_mode_tests": len([r for r in self.results if r["accuracy_mode"] == "maximum"]),
                "standard_mode_tests": len([r for r in self.results if r["accuracy_mode"] == "standard"]),
                "average_quality": statistics.mean([
                    score for result in self.results 
                    for score in result.get("quality_scores", [])
                ]) if self.results else 0,
                "average_response_time": statistics.mean([r["response_time"] for r in self.results]) if self.results else 0
            }
        }
        
        with open("enhanced_accuracy_test_results.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: enhanced_accuracy_test_results.json")

async def main():
    """Main test function"""
    print("🎯 ENHANCED ACCURACY TESTING SUITE")
    print("Testing Maximum Accuracy Features")
    print("=" * 60)
    
    tester = EnhancedAccuracyTester(BASE_URL, API_TOKEN)
    
    # Test server connectivity
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{BASE_URL}/")
            if response.status_code == 200:
                print("✅ Server is running and accessible")
            else:
                print(f"⚠️  Server responded with status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure the enhanced server is running with: python main_enhanced.py")
        return
    
    # Run comprehensive accuracy tests
    await tester.test_enhanced_accuracy()
    
    print(f"\n🎉 Enhanced accuracy testing completed!")
    print("📊 Check enhanced_accuracy_test_results.json for detailed metrics")

if __name__ == "__main__":
    asyncio.run(main())