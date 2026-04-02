import torch
import os
from transformers import TimesFm2_5ModelForPrediction, TimesFm2_5Config

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class TimesFmOnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.config = model.config
        self.output_projection_point = model.output_projection_point
        self.output_projection_quantiles = model.output_projection_quantiles
        self.tolerance = self.model.tolerance

    def forward(self, past_values):
        # 1. VECTORIZED RUNNING STATS
        patch_len = self.config.patch_length
        batch_size, seq_len = past_values.shape[0], past_values.shape[1]
        
        patched_inputs = past_values.reshape(batch_size, -1, patch_len)
        patch_sum = patched_inputs.sum(dim=-1)
        patch_sq_sum = (patched_inputs ** 2).sum(dim=-1)

        cum_sum = patch_sum.cumsum(dim=1)
        cum_sq_sum = patch_sq_sum.cumsum(dim=1)
        
        num_patches = patched_inputs.shape[1]
        cum_count = torch.arange(1, num_patches + 1, device=past_values.device, dtype=past_values.dtype).unsqueeze(0) * patch_len

        context_mu = cum_sum / cum_count
        cum_var = (cum_sq_sum / cum_count) - (context_mu ** 2)
        context_sigma = torch.sqrt(torch.clamp(cum_var, min=0.0))
        
        # 2. TRANSFORMER LAYERS
        normed = self.model._revin(patched_inputs, context_mu, context_sigma, reverse=False)
        tok_in = torch.cat([normed, torch.ones_like(normed)], dim=-1)
        hidden = self.model.input_ff_layer(tok_in)
        
        pos_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        pos_emb = self.model.rotary_emb(hidden, pos_ids)
        
        for layer in self.model.layers:
            hidden = layer(hidden, position_embeddings=pos_emb, attention_mask=None, position_ids=pos_ids)
            
        # 3. DYNAMIC PATCH EXTRACTION
        # Using [:, -1:] ensures it remains symbolic in legacy tracer
        loc = context_mu[:, -1:]
        scale = torch.clamp(context_sigma[:, -1:], min=self.tolerance)
        hidden_last = hidden[:, -1:, :]

        # 4. PROJECTIONS
        num_q = len(self.config.quantiles) + 1
        pf = self.output_projection_point(hidden_last).view(batch_size, self.config.horizon_length, num_q)
        qs = self.output_projection_quantiles(hidden_last).view(batch_size, self.config.output_quantile_len, num_q)

        # 5. COMBINATION
        full = pf.clone()
        if self.config.use_continuous_quantile_head:
            m_idx = self.config.decode_index
            h = self.config.horizon_length
            qs_slice = qs[:, :h, :]
            full = torch.cat([pf[..., :1], qs_slice[:, :, 1:] - qs_slice[:, :, m_idx:m_idx+1] + pf[:, :, m_idx:m_idx+1]], dim=-1)

        # 6. DENORMALIZE
        full = full * scale.unsqueeze(-1) + loc.unsqueeze(-1)
        mean = full[:, :, self.config.decode_index]

        return mean, full

def main():
    model_id = "google/timesfm-2.5-200m-transformers"
    print(f"Loading {model_id}...")
    config = TimesFm2_5Config.from_pretrained(model_id)
    config._attn_implementation = "eager"
    model = TimesFm2_5ModelForPrediction.from_pretrained(model_id, config=config).eval()
    
    wrapper = TimesFmOnnxWrapper(model).eval()
    output_path = "../timesfm-onnx-dynamic/onnx/model.onnx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        wrapper,
        (torch.randn(1, 512),),
        output_path,
        input_names=["past_values"],
        output_names=["mean_predictions", "full_predictions"],
        dynamic_axes={
            "past_values": {0: "batch_size", 1: "seq_len"},
            "mean_predictions": {0: "batch_size"},
            "full_predictions": {0: "batch_size"}
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False # Legacy exporter is more stable for this complex combination
    )
    print("Export complete!")

if __name__ == "__main__":
    main()
