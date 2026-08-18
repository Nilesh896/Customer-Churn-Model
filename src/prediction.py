import os
import joblib
import pandas as pd

class ChurnPredictor:
    """
    Predictor class that loads the saved best pipeline and runs predictions on raw user inputs.
    """
    def __init__(self, pipeline_path="models/best_pipeline.pkl"):
        self.pipeline_path = pipeline_path
        self.pipeline = None
        self._load_pipeline()
        
    def _load_pipeline(self):
        """
        Loads the pickled sklearn Pipeline.
        """
        if not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(
                f"Model pipeline not found at {self.pipeline_path}. "
                f"Please run 'python main.py' to train and serialize the model pipeline first."
            )
        try:
            self.pipeline = joblib.load(self.pipeline_path)
            print(f"Successfully loaded model pipeline from {self.pipeline_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model pipeline from {self.pipeline_path}: {e}")
            
    def predict_single(self, input_dict):
        """
        Takes a dictionary of a single customer's raw inputs, converts it into a DataFrame,
        passes it through the loaded Pipeline, and returns predictions, probabilities, and risk levels.
        """
        if self.pipeline is None:
            self._load_pipeline()
            
        # Convert single input dict to DataFrame (matching expected features)
        df_input = pd.DataFrame([input_dict])
        
        # Calculate class prediction and probabilities
        try:
            pred_class = self.pipeline.predict(df_input)[0]
            pred_prob = self.pipeline.predict_proba(df_input)[0][1] # Probability of class 1 (Churn)
        except Exception as e:
            raise ValueError(f"Error occurred during preprocessing/prediction inside pipeline: {e}")
            
        # Define risk category based on thresholds:
        # Low < 30%, Medium 30%-69.99%, High >= 70%
        prob_percent = pred_prob * 100
        if prob_percent < 30.0:
            risk_level = "Low Risk"
        elif 30.0 <= prob_percent < 70.0:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
            
        prediction_label = "Likely to Churn" if pred_class == 1 else "Likely to Stay"
        
        return {
            'Prediction': prediction_label,
            'Probability': prob_percent,
            'Risk Level': risk_level
        }
