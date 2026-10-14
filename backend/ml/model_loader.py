import os
import logging
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger("outbreakwatch.ml.loader")

class ModelLoader:
    """
    Safely discovers, inspects, and loads transferred machine-learning model artifacts.
    Supports PyTorch (.pth, .pt) and serialized Scikit-learn/Joblib (.joblib, .pkl) models.
    Does NOT invent fake models or silently substitute random weights if missing.
    """

    def __init__(self, model_dir: Path, target_filename: Optional[str] = None):
        self.model_dir = Path(model_dir)
        self.target_filename = target_filename
        self.model_artifact_path: Optional[Path] = None
        self.model_type: Optional[str] = None
        self.model_instance: Any = None
        self.is_loaded: bool = False
        self.error_message: Optional[str] = None
        self.status: str = "uninitialized"

    def discover_model_artifact(self) -> Optional[Path]:
        """Search the model directory for recognized ML model artifacts."""
        if not self.model_dir.exists():
            try:
                self.model_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create model directory {self.model_dir}: {e}")
            return None

        # If a specific target filename was configured and exists
        if self.target_filename:
            target_path = self.model_dir / self.target_filename
            if target_path.exists():
                return target_path

        # Scan for supported formats
        supported_extensions = [".pth", ".pt", ".joblib", ".pkl", ".onnx"]
        for ext in supported_extensions:
            for file_path in self.model_dir.glob(f"*{ext}"):
                if file_path.is_file():
                    return file_path

        return None

    def load(self) -> Tuple[bool, str]:
        """
        Loads the model artifact.
        Returns:
            (success: bool, status_or_error: str)
        """
        artifact_path = self.discover_model_artifact()

        if artifact_path is None:
            self.model_artifact_path = None
            self.is_loaded = False
            self.status = "model_artifact_missing"
            self.error_message = (
                f"No trained machine-learning model artifact (.pth, .pt, .joblib, .pkl) "
                f"found in '{self.model_dir.resolve()}'. The original model was not transferred to this computer."
            )
            logger.warning(self.error_message)
            return False, self.error_message

        self.model_artifact_path = artifact_path
        ext = artifact_path.suffix.lower()

        try:
            if ext in [".joblib", ".pkl"]:
                import joblib
                logger.info(f"Loading joblib/pickle model from {artifact_path}...")
                self.model_instance = joblib.load(str(artifact_path))
                self.model_type = "scikit-learn/joblib"
                self.is_loaded = True
                self.status = "ready"
                self.error_message = None
                logger.info("Joblib model loaded successfully.")
                return True, "ready"

            elif ext in [".pth", ".pt"]:
                import torch
                logger.info(f"Loading PyTorch checkpoint from {artifact_path}...")
                checkpoint = torch.load(str(artifact_path), map_location=torch.device("cpu"), weights_only=False)
                
                # Check if checkpoint is a pure state_dict or full model
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self.model_instance = checkpoint["state_dict"]
                    self.model_type = "pytorch_state_dict"
                elif isinstance(checkpoint, dict):
                    self.model_instance = checkpoint
                    self.model_type = "pytorch_dict"
                else:
                    self.model_instance = checkpoint
                    if hasattr(self.model_instance, "eval"):
                        self.model_instance.eval()
                    self.model_type = "pytorch_module"

                self.is_loaded = True
                self.status = "ready"
                self.error_message = None
                logger.info(f"PyTorch artifact loaded successfully ({self.model_type}).")
                return True, "ready"

            else:
                self.status = "unsupported_format"
                self.error_message = f"Unsupported model format '{ext}' in {artifact_path}."
                self.is_loaded = False
                logger.error(self.error_message)
                return False, self.error_message

        except Exception as e:
            self.is_loaded = False
            self.status = "load_failed"
            self.error_message = f"Failed to deserialize model from {artifact_path}: {str(e)}"
            logger.exception(self.error_message)
            return False, self.error_message

    def get_info(self) -> dict:
        return {
            "is_loaded": self.is_loaded,
            "status": self.status,
            "artifact_path": str(self.model_artifact_path) if self.model_artifact_path else None,
            "model_type": self.model_type,
            "error_message": self.error_message,
        }
