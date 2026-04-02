import torch
from transformers import TimesFm2_5ModelForPrediction
import os

# Minimal wrapper to bridge 2D tensor to the library's internal projection logic.
# This avoids the list-processing loops in the library that break standard ONNX tracing.
class TimesFmOnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, x):
        # 1. Global statistics & Normalization (uses library's _revin)
        mu, sigma = x.mean(dim=1, keepdim=True), x.std(dim=1, keepdim=True)
        norm = self.model.model._revin(x, mu, sigma)
        mask = torch.zeros(x.shape[0], x.shape[1] + self.config.horizon_length, dtype=torch.long, device=x.device)
        
        # 2. Use library's own decoder/projection logic (handles transformer + point/quantile heads)
        pf, qs, _ = self.model._decode_and_project(norm, mask)
        
        # 3. Combine point and quantile forecasts (replicates library's combination logic)
        full = pf[:, :self.config.horizon_length, :].clone()
        if self.config.use_continuous_quantile_head:
            m_idx = self.config.decode_index
            full = torch.cat([full[..., :1], qs[:, :, 1:] - qs[:, :, m_idx:m_idx+1] + full[:, :, m_idx:m_idx+1]], -1)

        # 4. Denormalize
        full = self.model.model._revin(full, mu, sigma, reverse=True)
        return full[:, :, self.config.decode_index], full

def main():
    model_id = "google/timesfm-2.5-200m-transformers"
    print(f"Loading {model_id}...")
    model = TimesFm2_5ModelForPrediction.from_pretrained(model_id).eval()
    
    output_path = "../timesfm-onnx-final/model.onnx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        TimesFmOnnxWrapper(model),
        (torch.randn(1, 512),),
        output_path,
        input_names=["past_values"],
        output_names=["mean_predictions", "full_predictions"],
        opset_version=18,
        do_constant_folding=True
    )
    print("Export complete!")

if __name__ == "__main__":
    main()
