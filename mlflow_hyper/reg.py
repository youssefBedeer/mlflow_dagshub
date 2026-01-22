# Register the best model using the model URI
import mlflow
mlflow.register_model(
    model_uri="runs:/c11931a3314947e59e74e0c9f0916096/model",
    name="housing-price-predictor",
)