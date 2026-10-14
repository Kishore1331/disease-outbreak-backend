import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
from backend.config import settings
from backend.ml.model_loader import ModelLoader
from backend.ml.preprocessing import PreprocessingPipeline

logger = logging.getLogger("outbreakwatch.services.model")

class ModelService:
    """
    Singleton service managing the machine-learning model lifecycle.
    Loads the transferred model once on startup and reuses it for inference.
    """

    def __init__(self):
        self.loader = ModelLoader(
            model_dir=settings.MODEL_DIR,
            target_filename=settings.MODEL_FILENAME
        )
        self.pipeline = PreprocessingPipeline()
        self.initialize()

    def initialize(self):
        logger.info("Initializing ML Model Service...")
        success, message = self.loader.load()
        if success:
            logger.info(f"Model initialized successfully ({self.loader.model_type}).")
        else:
            logger.warning(f"Model service initialized without model artifact: {message}")

    @property
    def is_ready(self) -> bool:
        return self.loader.is_loaded

    def get_status(self) -> Dict[str, Any]:
        info = self.loader.get_info()
        return {
            "model_ready": self.is_ready,
            "status": info["status"],
            "artifact_path": info["artifact_path"],
            "model_type": info["model_type"],
            "error": info["error_message"],
            "feature_names": self.pipeline.FEATURE_NAMES,
            "supported_horizons": settings.SUPPORTED_HORIZON_DAYS,
        }

    def predict(self, feature_array: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
        """
        Runs inference on the loaded model.
        Returns:
            (predicted_cases, error_message)
        """
        if not self.is_ready:
            return None, (
                "Cannot perform ML inference: Trained model artifact is not loaded. "
                f"Status: {self.loader.status}."
            )

        try:
            model = self.loader.model_instance
            if self.loader.model_type == "scikit-learn/joblib":
                pred = model.predict(feature_array)
                val = float(pred[0])
                return max(0.0, val), None

            elif "pytorch" in (self.loader.model_type or ""):
                import torch
                with torch.no_grad():
                    inp = torch.tensor(feature_array, dtype=torch.float32)
                    if hasattr(model, "forward"):
                        out = model(inp)
                        val = float(out.item() if out.numel() == 1 else out[0].item())
                        return max(0.0, val), None
                    else:
                        return None, "PyTorch artifact is a state dict without model architecture wrapper."

            return None, f"Unsupported model execution type: {self.loader.model_type}"

        except Exception as e:
            logger.exception(f"Inference execution failed: {e}")
            return None, f"Inference execution error: {str(e)}"

model_service = ModelService()
