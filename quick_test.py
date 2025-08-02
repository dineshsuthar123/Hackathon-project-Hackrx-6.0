#!/usr/bin/env python3

# Quick test of the advanced app
print("Testing app import...")

try:
    import app
    print("✅ App imported successfully!")
    print("Advanced models:", getattr(app, 'ADVANCED_MODELS_AVAILABLE', 'Unknown'))
    
    # Test creating a processor
    processor = app.AdvancedDocumentProcessor()
    print("✅ AdvancedDocumentProcessor created successfully!")
    
    print("🎉 System ready for deployment!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
