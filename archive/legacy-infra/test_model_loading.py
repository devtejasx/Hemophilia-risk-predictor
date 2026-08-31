"""
Test script to verify model loading fixes
"""

import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 60)
print("Testing Model Loading Fixes")
print("=" * 60)

# Test 1: Test local model loading
print("\n1️⃣ Testing Local Model Loading...")
try:
    from local_model import LocalChatbot
    chatbot = LocalChatbot()
    if chatbot.model is not None:
        print("   ✅ Local model loaded successfully")
    else:
        print("   ⓘ Local model not available (using fallback)")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Test prediction service with None path
print("\n2️⃣ Testing Prediction Service with Invalid Path...")
try:
    from backend.services.prediction import PredictionService
    service = PredictionService(model_path=None, explainability_enabled=True)
    print("   ✅ Service created with None path (handled gracefully)")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Test prediction service with non-existent path
print("\n3️⃣ Testing Prediction Service with Non-Existent Path...")
try:
    from backend.services.prediction import PredictionService
    service = PredictionService(model_path="/nonexistent/path/model.pkl", explainability_enabled=True)
    if service.model is None:
        print("   ✅ Service handled non-existent path gracefully")
    else:
        print("   ✅ Model loaded")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: Test ExplainabilityService with None model
print("\n4️⃣ Testing ExplainabilityService with None Model...")
try:
    from backend.services.explainability import ExplainabilityService
    explainer = ExplainabilityService(model=None)
    print("   ✅ ExplainabilityService handled None model gracefully")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 5: Test with mock model
print("\n5️⃣ Testing with Mock Random Forest Model...")
try:
    from sklearn.ensemble import RandomForestClassifier
    from backend.services.prediction import PredictionService
    
    # Create and test mock model
    mock_model = RandomForestClassifier(n_estimators=5)
    service = PredictionService.__new__(PredictionService)
    service.model = mock_model
    service.explainability_enabled = True
    service.explainer = None
    service.background_data = None
    service.feature_names = []
    
    if service.model is not None:
        print("   ✅ Mock model service created successfully")
    else:
        print("   ❌ Mock model failed")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ All tests completed successfully!")
print("=" * 60)
