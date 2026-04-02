import torch
import torch.nn as nn
from transformers import TimesFm2_5ModelForPrediction
import os

class TimesFmOnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, past_values, past_values_padding):
        # past_values: [batch, context_len]
        # past_values_padding: [batch, context_len + horizon_len]
        
        # 1. Global statistics
        mu_global = past_values.mean(dim=1, keepdim=True)
        sigma_global = past_values.std(dim=1, keepdim=True)
        
        # 2. RevIN normalization
        tolerance = self.model.model.tolerance
        safe_sigma = torch.where(sigma_global < tolerance, torch.ones_like(sigma_global), sigma_global)
        normalized_ts = (past_values - mu_global) / safe_sigma
        
        # 3. Decode and project
        pf_outputs, quantile_spreads, _ = self.model._decode_and_project(normalized_ts, past_values_padding)
        
        # 4. Flip invariance
        if self.config.force_flip_invariance:
            flipped_pf, flipped_qs, _ = self.model._decode_and_project(-normalized_ts, past_values_padding)
            
            # _flip_quantiles logic
            flipped_pf_q = torch.cat([flipped_pf[..., :1], torch.flip(flipped_pf[..., 1:], dims=(-1,))], dim=-1)
            flipped_qs_q = torch.cat([flipped_qs[..., :1], torch.flip(flipped_qs[..., 1:], dims=(-1,))], dim=-1)
            
            pf_outputs = (pf_outputs - flipped_pf_q) / 2
            quantile_spreads = (quantile_spreads - flipped_qs_q) / 2
            
        # 5. Combine point and quantiles
        horizon = self.config.horizon_length
        full_forecast = pf_outputs[:, :horizon, :].clone()

        median_index = self.config.decode_index
        if self.config.use_continuous_quantile_head:
            max_quantile_horizon = horizon
            
            qs_slice = quantile_spreads[:, :max_quantile_horizon, :]
            qs_median = quantile_spreads[:, :max_quantile_horizon, median_index:median_index+1]
            ff_median = full_forecast[:, :max_quantile_horizon, median_index:median_index+1]
            
            adjusted_quantiles = qs_slice - qs_median + ff_median
            full_forecast = torch.cat([full_forecast[:, :max_quantile_horizon, :1], adjusted_quantiles[:, :, 1:]], dim=-1)

        # 6. Denormalize
        full_predictions = full_forecast * sigma_global.unsqueeze(-1) + mu_global.unsqueeze(-1)
        
        decode_index = self.config.decode_index
        mean_predictions = full_predictions[:, :, decode_index]
        
        # 7. Truncate negative
        if self.config.infer_is_positive:
            input_min = torch.min(past_values, dim=1, keepdim=True).values
            should_clamp = (input_min >= 0)
            
            clamped_mean = torch.clamp(mean_predictions, min=0.0)
            clamped_full = torch.clamp(full_predictions, min=0.0)
            
            mean_predictions = torch.where(should_clamp, clamped_mean, mean_predictions)
            full_predictions = torch.where(should_clamp, clamped_full, full_predictions)
            
        return mean_predictions, full_predictions

def main():
    model_id = "google/timesfm-2.5-200m-transformers"
    print(f"Loading model {model_id}...")
    model = TimesFm2_5ModelForPrediction.from_pretrained(model_id)
    model.eval()
    
    wrapper = TimesFmOnnxWrapper(model)
    wrapper.eval()
    
    batch_size = 1
    context_len = 512 
    horizon_len = model.config.horizon_length
    
    past_values = torch.randn(batch_size, context_len)
    past_values_padding = torch.zeros(batch_size, context_len + horizon_len, dtype=torch.long)
    
    output_dir = "../timesfm-onnx"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model.onnx")
    
    print(f"Exporting to {output_path}...")
    # Use the legacy exporter if possible to avoid Dynamo issues for now, or ensure dynamo=False
    # Actually, let's try to see if we can use the old exporter by NOT using torch.export.
    # To force old exporter, we can try to use torch.onnx._export (internal but sometimes more stable)
    # or just use the standard torch.onnx.export and hope it doesn't default to Dynamo if we don't pass certain flags.
    
    # In Torch 2.x, torch.onnx.export might still use the old one by default if not told otherwise.
    # The "✅ Obtain model graph for ... with torch.export.export" message showed it WAS using the new one.
    
    # Let's try to use the legacy exporter explicitly if available, or just use torch.onnx.export with 
    # a simple model and see.
    
    torch.onnx.export(
        wrapper,
        (past_values, past_values_padding),
        output_path,
        input_names=["past_values", "past_values_padding"],
        output_names=["mean_predictions", "full_predictions"],
        opset_version=18,
        do_constant_folding=True,
    )
    print("Export complete!")

if __name__ == "__main__":
    main()
