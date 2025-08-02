#!/usr/bin/env python3
"""
Quick deployment test for the zero-dependency server
"""

import subprocess
import sys
import time

def test_server_startup():
    """Test if the server can start up"""
    print("🧪 Testing Zero-Dependency Server Startup")
    print("=" * 50)
    
    try:
        # Test Python compilation
        print("1️⃣ Testing Python compilation...")
        result = subprocess.run([sys.executable, "-m", "py_compile", "production_server_zero_deps.py"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Server file compiles successfully")
        else:
            print(f"❌ Compilation failed: {result.stderr}")
            return False
        
        # Test import
        print("2️⃣ Testing imports...")
        test_import = """
import sys
sys.path.insert(0, '.')
try:
    from production_server_zero_deps import app
    print("✅ App imported successfully")
    print(f"App type: {type(app)}")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
"""
        result = subprocess.run([sys.executable, "-c", test_import], 
                              capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.returncode != 0:
            print(f"❌ Import failed: {result.stderr}")
            return False
        
        print("3️⃣ Testing server startup (5 seconds)...")
        # Test actual server startup
        server_process = subprocess.Popen([
            sys.executable, "-c", 
            "import uvicorn; from production_server_zero_deps import app; uvicorn.run(app, host='127.0.0.1', port=8001)"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(5)  # Give server time to start
        
        if server_process.poll() is None:
            print("✅ Server started successfully")
            server_process.terminate()
            server_process.wait()
            return True
        else:
            stdout, stderr = server_process.communicate()
            print(f"❌ Server failed to start: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Zero-Dependency Server Deployment Test")
    print("=" * 50)
    
    if test_server_startup():
        print("\n🎉 SUCCESS: Server is ready for deployment!")
        print("✅ No compilation issues")
        print("✅ All imports working")
        print("✅ Server starts successfully")
        print("\n📋 Deployment Commands:")
        print("   Build: pip install -r requirements-ultra-light.txt")
        print("   Start: uvicorn production_server_zero_deps:app --host 0.0.0.0 --port $PORT")
    else:
        print("\n❌ FAILURE: Server has issues")
        print("❗ Check the error messages above")

if __name__ == "__main__":
    main()
