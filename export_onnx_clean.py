import torch
import torch.nn as nn
from transformers import TimesFm2_5ModelForPrediction
import os

class TimesFmOnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, input_ts, input_padding):
        # We call the model's own internal methods instead of reimplementing them.
        # This ensures the ONNX graph uses the exact library logic.
        
        # 1. Global stats (Required to bridge the gap from list input)
        mu_global = input_ts.mean(dim=1, keepdim=True)
        sigma_global = input_ts.std(dim=1, keepdim=True)
        
        # 2. Use the model's own RevIN implementation
        normalized_ts = self.model.model._revin(input_ts, mu_global, sigma_global, reverse=False)
        
        # 3. Use the model's own decode and project logic
        pf_outputs, quantile_spreads, _ = self.model._decode_and_project(normalized_ts, input_padding)
        
        # 4. Handle flip invariance if enabled
        if self.config.force_flip_invariance:
            flipped_pf, flipped_qs, _ = self.model._decode_and_project(-normalized_ts, input_padding)
            
            # Use the model's own internal _flip_quantiles if possible, or stay minimal
            def _flip_quantiles(x):
                return torch.cat([x[..., :1], torch.flip(x[..., 1:], dims=(-1,))], dim=-1)
            
            pf_outputs = (pf_outputs - _flip_quantiles(flipped_pf)) / 2
            quantile_spreads = (quantile_spreads - _flip_quantiles(flipped_qs)) / 2
            
        # 5. Combine predictions using the model's logic
        horizon = self.config.horizon_length
        full_forecast = pf_outputs[:, :horizon, :].clone()
        median_index = self.config.decode_index
        
        if self.config.use_continuous_quantile_head:
            qs_slice = quantile_spreads[:, :horizon, :]
            qs_median = quantile_spreads[:, :horizon, median_index:median_index+1]
            ff_median = full_forecast[:, :horizon, median_index:median_index+1]
            
            adjusted_quantiles = qs_slice - qs_median + ff_median
            full_forecast = torch.cat([full_forecast[:, :horizon, :1], adjusted_quantiles[:, :, 1:]], dim=-1)

        # 6. Final denormalization using model's RevIN
        full_predictions = self.model.model._revin(full_forecast, mu_global, sigma_global, reverse=True)
        mean_predictions = full_predictions[:, :, median_index]
        
        return mean_predictions, full_predictions

def main():
    model_id = "google/timesfm-2.5-200m-transformers"
    print(f"Loading model {model_id}...")
    model = TimesFm2_5ModelForPrediction.from_pretrained(model_id)
    model.eval()
    
    wrapper = TimesFmOnnxWrapper(model)
    wrapper.eval()
    
    # Inputs: [Batch, Context] and [Batch, Context + Horizon]
    batch_size = 1
    context_len = 512 
    horizon_len = model.config.horizon_length
    
    past_values = torch.randn(batch_size, context_len)
    past_values_padding = torch.zeros(batch_size, context_len + horizon_len, dtype=torch.long)
    
    output_dir = "../timesfm-onnx-2"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model.onnx")
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        wrapper,
        (past_values, past_values_padding),
        output_path,
        input_names=["past_values", "past_values_padding"],
        output_names=["mean_predictions", "full_predictions"],
        opset_version=18,
        do_constant_folding=True
    )
    print("Export complete!")

if __name__ == "__main__":
    main()
