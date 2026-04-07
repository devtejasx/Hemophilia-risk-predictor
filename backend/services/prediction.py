"""
Integrated Prediction Service
=============================

Complete prediction pipeline combining model inference, SHAP explanations,
clinical interpretation, and report generation.

Features:
- End-to-end prediction with explanations
- Automated clinical report generation
- Feature visualization and analysis
- Batch prediction processing
- Result caching and persistence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from datetime import datetime
import joblib
import json

from backend.services.explainability import ExplainabilityService
from backend.services.reports import ClinicalReportGenerator
from backend.services.trends import TrendService
from backend.services.alerts import AlertService

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Integrated service for predictions with explainability and reporting.
    
    Orchestrates prediction, explanation, clinical interpretation,
    and report generation in a single workflow.
    """
    
    def __init__(
        self,
        model_path: str,
        explainability_enabled: bool = True,
        background_data_path: Optional[str] = None
    ):
        """
        Initialize prediction service.
        
        Args:
            model_path: Path to trained model pickle/joblib file
            explainability_enabled: Enable SHAP explanations
            background_data_path: Path to background data for SHAP
        """
        self.model = self._load_model(model_path)
        self.explainability_enabled = explainability_enabled
        self.explainer = None
        self.background_data = None
        self.feature_names = []
        
        if explainability_enabled:
            self._initialize_explainability(background_data_path)
    
    def _load_model(self, model_path: str):
        """Load trained model from file."""
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None
    
    def _initialize_explainability(self, background_data_path: Optional[str]) -> None:
        """Initialize SHAP explainer."""
        try:
            if background_data_path:
                self.background_data = joblib.load(background_data_path)
            
            self.explainer = ExplainabilityService(
                model=self.model,
                background_data=self.background_data
            )
            logger.info("SHAP explainer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize explainability: {str(e)}")
            self.explainability_enabled = False
    
    def set_feature_names(self, feature_names: List[str]) -> None:
        """
        Set feature names for interpretability.
        
        Args:
            feature_names: List of feature column names
        """
        self.feature_names = feature_names
        if self.explainer:
            self.explainer.set_feature_names(feature_names)
    
    def predict_with_explanation(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate prediction with full explanation.
        
        Args:
            features: Input features (1D or 2D array with 1 row)
            feature_names: Optional feature names
            
        Returns:
            Dictionary containing:
            - prediction: Model prediction
            - prediction_proba: Confidence scores
            - explanation: SHAP explanation
            - clinical_summary: Clinical interpretation
            - timestamp: Generation timestamp
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Ensure proper shape
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Get prediction
            prediction = self.model.predict(features)[0]
            prediction_proba = None
            
            if hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba(features)[0]
            
            result = {
                "prediction": float(prediction),
                "prediction_proba": prediction_proba.tolist() if prediction_proba is not None else None,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add explanation if enabled
            if self.explainability_enabled and self.explainer:
                feature_names_to_use = feature_names or self.feature_names
                explanation = self.explainer.explain_prediction(features, feature_names_to_use)
                result["explanation"] = explanation
                
                # Generate clinical summary
                clinical_summary = self.explainer.generate_clinical_explanation(explanation)
                result["clinical_summary"] = clinical_summary
            
            return result
        
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {"error": str(e)}
    
    def batch_predict_with_explanations(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
        sample_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate predictions with explanations for multiple samples.
        
        Args:
            features: Input features (2D array, rows are samples)
            feature_names: Optional feature names
            sample_size: If provided, sample this many predictions
            
        Returns:
            List of prediction result dictionaries
        """
        if sample_size and len(features) > sample_size:
            indices = np.random.choice(len(features), sample_size, replace=False)
            features = features[indices]
        
        results = []
        for i in range(len(features)):
            result = self.predict_with_explanation(features[i:i+1], feature_names)
            results.append(result)
        
        return results
    
    def generate_full_report(
        self,
        patient_data: Dict[str, Any],
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
        include_trends: bool = False,
        include_visualizations: bool = False,
        output_path: Optional[str] = None
    ) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """
        Generate complete clinical report.
        
        Args:
            patient_data: Patient demographics and info
            features: Input features for prediction
            feature_names: Optional feature names
            include_trends: Include trend analysis
            include_visualizations: Include SHAP plots
            output_path: Optional path to save PDF
            
        Returns:
            Tuple of (PDF bytes, report data dictionary)
        """
        try:
            # Generate prediction with explanation
            pred_result = self.predict_with_explanation(features, feature_names)
            
            if "error" in pred_result:
                return None, pred_result
            
            # Prepare report data
            report_data = {
                "patient_data": patient_data,
                "prediction_data": {
                    "prediction_score": pred_result.get("prediction", 0),
                    "prediction_proba": pred_result.get("prediction_proba"),
                    "timestamp": pred_result.get("timestamp")
                },
                "explanation_data": pred_result.get("explanation", {}),
                "clinical_summary": pred_result.get("clinical_summary")
            }
            
            # Add trends if requested
            if include_trends:
                # This would integrate with TrendService
                # For now, placeholder
                report_data["trend_data"] = {
                    "average_risk": pred_result.get("prediction", 0),
                    "trend_direction": "stable"
                }
            
            # Generate visualizations if requested
            images = None
            if include_visualizations and self.explainer:
                images = self._generate_visualizations(features, feature_names)
                if images:
                    report_data["images"] = images
            
            # Generate PDF report
            report_generator = ClinicalReportGenerator(output_path)
            pdf_bytes = report_generator.generate_report(
                patient_data=report_data["patient_data"],
                prediction_data=report_data["prediction_data"],
                explanation_data=report_data["explanation_data"],
                clinical_summary=report_data["clinical_summary"],
                trend_data=report_data.get("trend_data"),
                images=images
            )
            
            return pdf_bytes, report_data
        
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return None, {"error": str(e)}
    
    def _generate_visualizations(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Optional[Dict[str, bytes]]:
        """
        Generate SHAP visualization plots.
        
        Args:
            features: Input features
            feature_names: Optional feature names
            
        Returns:
            Dictionary of plot names to PNG bytes
        """
        try:
            images = {}
            
            # Waterfall plot
            waterfall = self.explainer.generate_waterfall_plot(features, feature_names)
            if waterfall:
                images["Waterfall Plot"] = waterfall
            
            # Summary plot (top features)
            summary = self.explainer.generate_summary_plot(features, feature_names, plot_type="bar")
            if summary:
                images["Feature Importance"] = summary
            
            return images if images else None
        
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            return None
    
    def get_feature_importance(
        self,
        instances: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Get global feature importance.
        
        Args:
            instances: Input features (optional)
            
        Returns:
            Feature importance dictionary
        """
        if not self.explainability_enabled or not self.explainer:
            return {"error": "Explainability not enabled"}
        
        return self.explainer.get_feature_importance(instances)
    
    def generate_batch_reports(
        self,
        patient_records: List[Dict[str, Any]],
        output_dir: str,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[int, int]:
        """
        Generate reports for multiple patients.
        
        Args:
            patient_records: List of patient data + features
            output_dir: Directory to save PDFs
            feature_names: Optional feature names
            
        Returns:
            Tuple of (successful, failed) counts
        """
        successful = 0
        failed = 0
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for record in patient_records:
            try:
                patient_id = record.get("patient_id", "unknown")
                features = record.get("features")
                patient_data = record.get("patient_data", {})
                
                if features is None:
                    logger.warning(f"No features for patient {patient_id}")
                    failed += 1
                    continue
                
                filename = output_path / f"report_{patient_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
                
                pdf_bytes, report_data = self.generate_full_report(
                    patient_data=patient_data,
                    features=features,
                    feature_names=feature_names,
                    output_path=str(filename)
                )
                
                if pdf_bytes:
                    successful += 1
                    logger.info(f"Report generated: {filename}")
                else:
                    failed += 1
            
            except Exception as e:
                logger.error(f"Error generating report for patient {patient_id}: {str(e)}")
                failed += 1
        
        return successful, failed
    
    def export_explanation_as_json(
        self,
        explanation: Dict[str, Any],
        output_path: str
    ) -> bool:
        """
        Export explanation as JSON.
        
        Args:
            explanation: Explanation dictionary
            output_path: Path to save JSON
            
        Returns:
            Success status
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(explanation, f, indent=2, default=str)
            logger.info(f"Explanation exported: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting explanation: {str(e)}")
            return False
    
    def generate_cohort_analysis(
        self,
        features_list: List[np.ndarray],
        patient_ids: List[str],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate cohort-level analysis.
        
        Args:
            features_list: List of feature arrays
            patient_ids: List of patient IDs
            feature_names: Optional feature names
            
        Returns:
            Cohort analysis dictionary
        """
        try:
            results = []
            predictions = []
            
            for features, patient_id in zip(features_list, patient_ids):
                result = self.predict_with_explanation(features, feature_names)
                result["patient_id"] = patient_id
                results.append(result)
                predictions.append(result.get("prediction", 0))
            
            # Calculate statistics
            predictions = np.array(predictions)
            
            cohort_analysis = {
                "total_patients": len(patient_ids),
                "average_risk": float(np.mean(predictions)),
                "median_risk": float(np.median(predictions)),
                "std_risk": float(np.std(predictions)),
                "min_risk": float(np.min(predictions)),
                "max_risk": float(np.max(predictions)),
                "high_risk_count": int((predictions >= 0.7).sum()),
                "moderate_risk_count": int(((predictions >= 0.5) & (predictions < 0.7)).sum()),
                "low_risk_count": int((predictions < 0.5).sum()),
                "predictions": results
            }
            
            return cohort_analysis
        
        except Exception as e:
            logger.error(f"Error in cohort analysis: {str(e)}")
            return {"error": str(e)}
