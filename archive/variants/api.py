from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import pickle
import os
import warnings

app = FastAPI()

# Models and configuration
rf = None
xgb = None
columns = None
explainer = None

# Load models with graceful fallback
def load_models():
    global rf, xgb, columns, explainer
    
    try:
        # Try to load random forest model
        if os.path.exists("rf.pkl"):
            rf = joblib.load("rf.pkl")
            print("✓ Random Forest model loaded successfully")
        else:
            print("⚠ Warning: rf.pkl not found. Using mock model for predictions.")
            
        # Try to load XGBoost model
        if os.path.exists("xgb.pkl"):
            xgb = joblib.load("xgb.pkl")
            print("✓ XGBoost model loaded successfully")
        else:
            print("⚠ Warning: xgb.pkl not found. Using mock model for predictions.")
            
        # Load columns with memory-safe approach
        try:
            if os.path.exists("columns.pkl"):
                columns = joblib.load("columns.pkl", mmap_mode='r')
                print("✓ Columns loaded successfully")
            else:
                # Define default columns if file doesn't exist
                columns = [
                    'mutation_type_intron22', 'mutation_type_intron1', 'mutation_type_large_deletion',
                    'mutation_type_small_deletion', 'mutation_type_inversion', 'mutation_type_point_mutation',
                    'exon', 'age_first_treatment', 'dose_intensity', 'exposure_days'
                ]
                print("⚠ columns.pkl not found. Using default columns.")
        except (MemoryError, EOFError, pickle.UnpicklingError) as e:
            print(f"⚠ Error loading columns: {e}. Using default columns.")
            columns = [
                'mutation_type_intron22', 'mutation_type_intron1', 'mutation_type_large_deletion',
                'mutation_type_small_deletion', 'mutation_type_inversion', 'mutation_type_point_mutation',
                'exon', 'age_first_treatment', 'dose_intensity', 'exposure_days'
            ]
        
        # Initialize SHAP explainer if model is available
        if rf is not None:
            try:
                import shap
                explainer = shap.TreeExplainer(rf)
                print("✓ SHAP explainer initialized")
            except Exception as e:
                print(f"⚠ Could not initialize SHAP explainer: {e}")
                explainer = None
        else:
            print("⚠ SHAP explainer not initialized (no RF model available)")
            
    except Exception as e:
        print(f"Error during model loading: {e}")
        print("API will continue with mock predictions")

# Load models on startup
load_models()


@app.get("/")
def home():
    return {"message": "API Running"}


@app.get("/predict")
def predict(age: int, dose: int, exposure: int):
    """
    Predict hemophilia inhibitor risk based on patient parameters.
    Returns mock predictions if models are not available.
    """
    
    data = {
        "mutation_type": "intron22",
        "exon": 22,
        "severity": "severe",
        "age_first_treatment": age,
        "dose_intensity": dose,
        "exposure_days": exposure
    }

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    # Ensure all required columns exist
    if columns:
        for col in columns:
            if col not in df:
                df[col] = 0
        df = df[columns]

    # Generate predictions
    if rf is not None and xgb is not None:
        try:
            # Real predictions from trained models
            p1 = rf.predict_proba(df)[0][1]
            p2 = xgb.predict_proba(df)[0][1]
            risk = (p1 + p2) / 2
            
            # SHAP explanations if available
            if explainer is not None:
                try:
                    shap_values = explainer.shap_values(df)
                    shap_vals = np.array(shap_values).flatten()
                    
                    # Feature importance
                    feature_importance = {}
                    for i in range(len(df.columns)):
                        feature_importance[df.columns[i]] = float(shap_vals[i])
                    
                    # Top feature
                    top_index = np.argmax(np.abs(shap_vals))
                    top_feature = df.columns[top_index]
                except Exception as e:
                    print(f"SHAP error: {e}")
                    # Fallback importance calculation
                    feature_importance = {col: 0.0 for col in df.columns}
                    top_feature = "age_first_treatment"
            else:
                feature_importance = {col: 0.0 for col in df.columns}
                top_feature = "age_first_treatment"
                
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return mock prediction on error
            risk = 0.45
            top_feature = "age_first_treatment"
            feature_importance = {col: 0.0 for col in df.columns}
    else:
        # Mock prediction when models are not loaded
        # Use simple heuristic based on inputs
        risk = min(0.9, (age / 100 + dose / 100 + exposure / 1000))
        risk = max(0.1, risk)  # Clamp between 0.1 and 0.9
        
        top_feature = "age_first_treatment"
        feature_importance = {col: 0.0 for col in (columns or ['age', 'dose', 'exposure'])}
        print(f"⚠ Mock prediction: risk={risk:.2f} (models not loaded)")

    return {
        "risk_score": float(risk),
        "reason": str(top_feature),
        "importance": feature_importance,
        "model_status": "trained" if (rf is not None and xgb is not None) else "mock"
    }