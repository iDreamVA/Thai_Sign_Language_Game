import json
from pathlib import Path

import torch

from tsl_preprocess import landmarks_to_tensor, read_landmark_csv


class SignPredictor:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.model = torch.jit.load(str(model_dir / "model.torchscript.pt"), map_location="cpu").eval()

    @torch.inference_mode()
    def _predict(self, sequence):
        probabilities = self.model(torch.from_numpy(sequence).unsqueeze(0)).softmax(dim=1)[0]
        confidence, index = probabilities.max(dim=0)
        index, confidence = index.item(), confidence.item()
        labels = self.metadata["classes"]
        accepted = confidence >= self.metadata["confidence_threshold"]
        chosen = labels[index] if accepted else labels[self.metadata["null_index"]]
        return {
            "sign_id": chosen["sign_id"], "thai": chosen["thai"],
            "raw_sign_id": labels[index]["sign_id"], "raw_confidence": confidence,
            "rejected_low_confidence": not accepted,
        }

    def predict_csv(self, path):
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            return self._predict(read_landmark_csv(stream, self.metadata["sequence_frames"]))

    def predict_landmarks(self, coordinates, time_ms=None):
        return self._predict(landmarks_to_tensor(coordinates, self.metadata["sequence_frames"], time_ms))
