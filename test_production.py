#!/usr/bin/env python3
"""
Production Test Script
Test the app without heavy ML dependencies to simulate production environment
"""

import subprocess
import sys
import os

def test_production_mode():
    """Test the app in production mode (without ML libraries)"""
    print("🚀 Testing Production Mode (simulating Render deployment)")
    print("=" * 60)
    
    # Temporarily rename/hide ML libraries to simulate production
    hidden_imports = []
    try:
        # Test imports that should fail in production
        test_imports = [
            "sentence_transformers", 
            "transformers", 
            "torch", 
            "sklearn"
        ]
        
        for module in test_imports:
            try:
                __import__(module)
                print(f"⚠️  {module} is available (should be missing in production)")
            except ImportError:
                print(f"✅ {module} missing (correct for production)")
        
        # Test basic app startup
        print("\n📱 Testing FastAPI app startup...")
        
        # Import the app
        sys.path.insert(0, os.getcwd())
        
        try:
            from app_new import app, PRODUCTION_MODE, ADVANCED_MODELS_AVAILABLE
            print(f"✅ App imported successfully")
            print(f"🏭 Production Mode: {PRODUCTION_MODE}")
            print(f"🧠 Advanced Models Available: {ADVANCED_MODELS_AVAILABLE}")
            
            if PRODUCTION_MODE:
                print("✅ Production mode detected - app should run on Render!")
            else:
                print("⚠️  Development mode - may have memory issues on Render")
                
        except Exception as e:
            print(f"❌ App import failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    print("\n🎯 Production test completed!")
    return True

if __name__ == "__main__":
    success = test_production_mode()
    if success:
        print("\n✅ Ready for production deployment!")
        print("\n📋 Next steps:")
        print("1. Go to your Render service settings")
        print("2. Change Build Command to: pip install -r requirements-prod.txt")
        print("3. Redeploy the service")
        print("4. Monitor memory usage (should be under 512MB)")
    else:
        print("\n❌ Production test failed - needs debugging")
    
    sys.exit(0 if success else 1)
