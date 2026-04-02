import torch
import os
from transformers import TimesFm2_5ModelForPrediction, TimesFm2_5Config

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class TimesFmOnnxWrapper(TimesFm2_5ModelForPrediction):
    """
    Inherits from the library. We only override the forward pass to bridge 
    the API and fix hardcoded indexing for dynamic context lengths.
    """
    def __init__(self, model):
        super().__init__(model.config)
        self.load_state_dict(model.state_dict())
        self.tolerance = model.model.tolerance

    def forward(self, past_values):
        # 1. VECTORIZED PREPROCESSING
        # Replacing library's for-loop with cumsum for dynamic running stats.
        b, seq_len = past_values.shape
        patch_len = self.config.patch_length
        
        patched = past_values.reshape(b, -1, patch_len)
        
        p_sum = patched.sum(dim=-1)
        p_sq_sum = (patched ** 2).sum(dim=-1)
        
        c_sum = p_sum.cumsum(dim=1)
        c_sq_sum = p_sq_sum.cumsum(dim=1)
        
        num_p = patched.shape[1]
        counts = torch.arange(1, num_p + 1, device=past_values.device, dtype=past_values.dtype).unsqueeze(0) * patch_len
        
        mu = c_sum / counts
        var = (c_sq_sum / counts) - (mu ** 2)
        sigma = torch.sqrt(torch.clamp(var, min=0.0))
        
        # 2. CORE TRANSFORMER
        normed = (patched - mu.unsqueeze(-1)) / torch.clamp(sigma.unsqueeze(-1), min=self.tolerance)
        tok_in = torch.cat([normed, torch.ones_like(normed)], dim=-1)
        hidden = self.model.input_ff_layer(tok_in)
        
        pos_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        pos_emb = self.model.rotary_emb(hidden, pos_ids)
        
        for layer in self.model.layers:
            hidden = layer(hidden, position_embeddings=pos_emb, attention_mask=None, position_ids=pos_ids)
            
        # 3. DYNAMIC PATCH EXTRACTION
        # Using [:, -1:] instead of [-1] to keep the patch dimension for broadcasting
        hidden_last = hidden[:, -1:, :]
        loc_last = mu[:, -1:].unsqueeze(-1)
        sigma_last = sigma[:, -1:].unsqueeze(-1)

        # 4. PROJECTIONS
        num_q = len(self.config.quantiles) + 1
        h = self.config.horizon_length
        
        pf = self.output_projection_point(hidden_last).reshape(b, h, num_q)
        qs = self.output_projection_quantiles(hidden_last).reshape(b, self.config.output_quantile_len, num_q)

        # Combine distribution
        full = pf.clone()
        if self.config.use_continuous_quantile_head:
            m_idx = self.config.decode_index
            qs_s = qs[:, :h, :]
            full = torch.cat([pf[..., :1], qs_s[:, :, 1:] - qs_s[:, :, m_idx:m_idx+1] + pf[:, :, m_idx:m_idx+1]], dim=-1)

        # 5. DENORMALIZE
        full_pred = full * sigma_last + loc_last
        mean_pred = full_pred[:, :, self.config.decode_index]

        return mean_pred, full_pred

def main():
    model_id = "google/timesfm-2.5-200m-transformers"
    print(f"Loading {model_id}...")
    
    # PATCH: The ONNX exporter (both legacy and Dynamo) currently cannot handle 
    # the `enable_gqa=True` flag in SDPA. Transformers blindly passes this if 
    # PyTorch >= 2.5 and q_heads != kv_heads. 
    # We patch the check to always return False, which forces a standard SDPA 
    # that exports perfectly while retaining all model logic.
    import transformers.integrations.sdpa_attention
    transformers.integrations.sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False

    model = TimesFm2_5ModelForPrediction.from_pretrained(model_id).eval()
    
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
        dynamo=False # Use legacy exporter to avoid SDPA SymbolicDim bugs
    )
    print("Export complete!")

if __name__ == "__main__":
    main()
