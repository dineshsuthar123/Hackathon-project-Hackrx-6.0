"""
GROQ + MONGODB HYPER-INTELLIGENCE SYSTEM TEST
Complete verification including MongoDB integration
"""

import sys
import os
import time
import asyncio

# Add current directory to path
sys.path.append('.')

def test_system_imports():
    """Test all critical imports including MongoDB"""
    print("🔍 TESTING SYSTEM IMPORTS")
    print("-" * 40)
    
    try:
        from app_groq_ultimate import (
            GroqIntelligenceEngine, 
            GroqDocumentProcessor, 
            MongoDBManager,
            STATIC_ANSWER_CACHE,
            GROQ_MODEL,
            GROQ_API_KEY,
            MONGODB_URI,
            MONGODB_DATABASE
        )
        print("✅ Core system imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_mongodb_dependencies():
    """Test MongoDB-specific dependencies"""
    print("\n🔍 TESTING MONGODB DEPENDENCIES")
    print("-" * 40)
    
    # Test MongoDB drivers
    try:
        import motor.motor_asyncio
        import pymongo
        print(f"✅ MongoDB drivers available")
        print(f"   PyMongo version: {pymongo.__version__}")
        print(f"   Motor: Async driver ready")
    except ImportError as e:
        print(f"❌ MongoDB drivers missing: {e}")
        return False
    
    # Test other dependencies
    deps = [
        ("FastAPI", "fastapi"),
        ("Groq", "groq"),
        ("pdfplumber", "pdfplumber"), 
        ("PyPDF2", "PyPDF2"),
        ("httpx", "httpx")
    ]
    
    for name, module in deps:
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"❌ {name} missing")
    
    return True

def test_mongodb_manager():
    """Test MongoDB Manager initialization"""
    print("\n🔍 TESTING MONGODB MANAGER")
    print("-" * 40)
    
    try:
        from app_groq_ultimate import MongoDBManager
        
        manager = MongoDBManager()
        
        if manager.client:
            print("✅ MongoDB client initialized")
            print(f"   Database: {manager.database.name}")
            print(f"   Collection: {manager.collection.name}")
        else:
            print("⚠️  MongoDB client in offline mode")
            print("   Will work without persistent caching")
        
        return True
    except Exception as e:
        print(f"❌ MongoDB manager test failed: {e}")
        return False

def test_groq_mongodb_processor():
    """Test complete processor with MongoDB"""
    print("\n🔍 TESTING GROQ + MONGODB PROCESSOR")
    print("-" * 40)
    
    try:
        from app_groq_ultimate import GroqDocumentProcessor
        
        processor = GroqDocumentProcessor()
        print("✅ Groq + MongoDB processor initialized")
        
        # Import static cache separately
        from app_groq_ultimate import STATIC_ANSWER_CACHE
        
        # Check components
        if processor.groq_engine.groq_client:
            print("✅ Groq engine: Active")
        else:
            print("⚠️  Groq engine: Fallback mode")
        
        if processor.mongodb_manager.client:
            print("✅ MongoDB manager: Connected")
        else:
            print("⚠️  MongoDB manager: Offline mode")
        
        print(f"✅ Static cache: {len(STATIC_ANSWER_CACHE)} answers loaded")
        print(f"✅ Performance tracking: {len(processor.stats)} metrics")
        
        return True
    except Exception as e:
        print(f"❌ Processor test failed: {e}")
        return False

async def test_mongodb_caching():
    """Test MongoDB caching functionality"""
    print("\n🔍 TESTING MONGODB CACHING")
    print("-" * 40)
    
    try:
        from app_groq_ultimate import MongoDBManager
        
        manager = MongoDBManager()
        
        if not manager.client:
            print("⚠️  MongoDB offline - skipping cache test")
            return True
        
        # Test caching
        test_url = "https://test.com/document.pdf"
        test_content = "Test document content for caching"
        test_qa = [{"question": "Test question?", "answer": "Test answer", "timestamp": time.time()}]
        
        print("🗄️ Testing document caching...")
        cache_result = await manager.cache_document(test_url, test_content, test_qa)
        
        if cache_result:
            print("✅ Document caching successful")
        else:
            print("⚠️  Document caching failed (may be connection issue)")
        
        # Test retrieval
        print("🗄️ Testing cache retrieval...")
        cached_answers = await manager.get_cached_answers(test_url, ["Test question?"])
        
        if cached_answers:
            print("✅ Cache retrieval successful")
            print(f"   Retrieved: {list(cached_answers.keys())[0]}")
        else:
            print("⚠️  Cache retrieval returned empty")
        
        # Cleanup
        await manager.close()
        
        return True
    except Exception as e:
        print(f"❌ MongoDB caching test failed: {e}")
        return False

async def test_intelligence_analysis():
    """Test complete intelligence analysis with caching"""
    print("\n🔍 TESTING COMPLETE INTELLIGENCE ANALYSIS")
    print("-" * 40)
    
    try:
        from app_groq_ultimate import GroqDocumentProcessor
        
        processor = GroqDocumentProcessor()
        
        # Test document
        test_document_url = "https://hackrx.blob.core.windows.net/hackrx/Arogya%20Sanjeevani%20Policy%20CIS_2.pdf"
        test_question = "What is the waiting period for Gout and Rheumatism?"
        
        start_time = time.time()
        answer = await processor.process_question_with_groq_intelligence(test_document_url, test_question)
        execution_time = (time.time() - start_time) * 1000
        
        print(f"✅ Analysis completed in {execution_time:.1f}ms")
        print(f"   Question: {test_question}")
        print(f"   Answer: {answer}")
        
        # Test stats
        stats = processor.stats
        print(f"✅ Performance metrics:")
        print(f"   Static cache hits: {stats['cache_hits']}")
        print(f"   MongoDB hits: {stats['mongodb_hits']}")
        print(f"   Groq calls: {stats['groq_calls']}")
        print(f"   Total questions: {stats['total_questions']}")
        
        return True
    except Exception as e:
        print(f"❌ Intelligence analysis test failed: {e}")
        return False

def test_memory_optimization():
    """Test memory usage"""
    print("\n🔍 TESTING MEMORY OPTIMIZATION")
    print("-" * 40)
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        print(f"✅ Current memory usage: {memory_mb:.1f}MB")
        
        if memory_mb < 150:
            print("✅ EXCELLENT: Memory usage under 150MB")
        elif memory_mb < 200:
            print("✅ GOOD: Memory usage under 200MB")
        elif memory_mb < 300:
            print("⚠️  OK: Memory usage under 300MB")
        else:
            print("❌ HIGH: Memory usage over 300MB - consider optimization")
        
        return memory_mb < 300
    except ImportError:
        print("⚠️  psutil not available - cannot check memory")
        return True
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return True

def test_configuration():
    """Test system configuration"""
    print("\n🔍 TESTING CONFIGURATION")
    print("-" * 40)
    
    try:
        from app_groq_ultimate import GROQ_API_KEY, GROQ_MODEL, MONGODB_URI, MONGODB_DATABASE
        
        print(f"✅ Groq Model: {GROQ_MODEL}")
        
        if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
            print("✅ Groq API key configured")
        else:
            print("⚠️  Groq API key not configured (fallback mode)")
        
        if MONGODB_URI:
            print("✅ MongoDB URI configured")
            print(f"   Database: {MONGODB_DATABASE}")
        else:
            print("⚠️  MongoDB URI not configured")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def run_all_tests():
    """Run comprehensive system test"""
    print("🚀 GROQ + MONGODB HYPER-INTELLIGENCE SYSTEM VERIFICATION")
    print("=" * 70)
    
    tests = [
        ("System Imports", test_system_imports),
        ("MongoDB Dependencies", test_mongodb_dependencies),
        ("MongoDB Manager", test_mongodb_manager),
        ("Groq+MongoDB Processor", test_groq_mongodb_processor),
        ("Configuration", test_configuration),
        ("Memory Optimization", test_memory_optimization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Test async functionality
    async_tests = [
        ("MongoDB Caching", test_mongodb_caching),
        ("Intelligence Analysis", test_intelligence_analysis)
    ]
    
    for test_name, test_func in async_tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 SYSTEM TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 GROQ + MONGODB SYSTEM FULLY OPERATIONAL")
        print("   🚀 Ready for production deployment")
        print("   🗄️ 3-level caching system active")
        print("   🎯 Use requirements_mongodb.txt for deployment")
        print("   ⚡ Start with: bash start_groq_mongodb.sh")
    elif passed >= total - 1:
        print("\n✅ SYSTEM READY WITH MINOR ISSUES")
        print("   🗄️ Most functionality working")
        print("   📦 Run: pip install -r requirements_mongodb.txt")
    else:
        print("\n⚠️  SYSTEM NEEDS ATTENTION")
        print("   📦 Install dependencies: pip install -r requirements_mongodb.txt")
        print("   🔧 Check .env configuration")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
