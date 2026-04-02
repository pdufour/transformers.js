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

    def forward(self, past_values, past_values_padding):
        # We process the 2D tensors directly to keep the batch dimension dynamic.
        model_outputs = self.model(
            past_values=past_values,
            past_values_padding=past_values_padding,
            return_dict=True,
        )
        
        hidden_states = model_outputs.last_hidden_state
        batch_size = hidden_states.shape[0]
        num_quantiles = len(self.config.quantiles) + 1
        
        # DYNAMIC INDEXING: Use dynamic slice
        hidden_last = hidden_states[:, -1:, :]
        
        point_forecast = self.output_projection_point(hidden_last).view(batch_size, self.config.horizon_length, num_quantiles)
        quantile_spreads = self.output_projection_quantiles(hidden_last).view(batch_size, self.config.output_quantile_len, num_quantiles)

        horizon = self.config.horizon_length
        full_forecast = point_forecast[:, :horizon, :].clone()
        median_index = self.config.decode_index

        if self.config.use_continuous_quantile_head:
            qs_slice = quantile_spreads[:, :horizon, :]
            qs_median = quantile_spreads[:, :horizon, median_index:median_index+1]
            ff_median = full_forecast[:, :horizon, median_index:median_index+1]
            adjusted_quantiles = qs_slice - qs_median + ff_median
            full_forecast = torch.cat([full_forecast[:, :horizon, :1], adjusted_quantiles[:, :, 1:]], dim=-1)

        mu = model_outputs.loc.unsqueeze(-1).unsqueeze(-1)
        sigma = model_outputs.scale.unsqueeze(-1).unsqueeze(-1)
        
        full_predictions = full_forecast * sigma + mu
        mean_predictions = full_predictions[:, :, median_index]

        return mean_predictions, full_predictions

def main():
    model_id = "google/timesfm-2.5-200m-transformers"
    print(f"Loading {model_id}...")
    
    config = TimesFm2_5Config.from_pretrained(model_id)
    config._attn_implementation = "eager"
    model = TimesFm2_5ModelForPrediction.from_pretrained(model_id, config=config).eval()
    
    wrapper = TimesFmOnnxWrapper(model).eval()
    
    def dynamic_inner_forward(self, inputs_embeds=None, past_values=None, past_values_padding=None, **kwargs):
        patch_len = self.config.patch_length
        batch_size = past_values.shape[0]
        
        patched_inputs = past_values.reshape(batch_size, -1, patch_len)
        
        # VECTORIZED RUNNING STATS
        patched_masks = past_values_padding[:, :past_values.shape[1]].reshape(batch_size, -1, patch_len)
        patched_masks_bool = patched_masks >= 0.5
        is_valid = (~patched_masks_bool).to(patched_inputs.dtype)
        
        patch_count = is_valid.sum(dim=-1)
        patch_sum = (patched_inputs * is_valid).sum(dim=-1)
        patch_sq_sum = ((patched_inputs ** 2) * is_valid).sum(dim=-1)

        cum_count = patch_count.cumsum(dim=1)
        cum_sum = patch_sum.cumsum(dim=1)
        cum_sq_sum = patch_sq_sum.cumsum(dim=1)
        
        cum_count_safe = torch.where(cum_count == 0, torch.ones_like(cum_count), cum_count)
        
        context_mu = cum_sum / cum_count_safe
        context_mu = torch.where(cum_count == 0, torch.zeros_like(context_mu), context_mu)

        cum_var = (cum_sq_sum / cum_count_safe) - (context_mu ** 2)
        cum_var = torch.where(cum_count == 0, torch.zeros_like(cum_var), cum_var)
        
        context_sigma = torch.sqrt(torch.clamp(cum_var, min=0.0))
        
        normed_inputs = self._revin(patched_inputs, context_mu, context_sigma, reverse=False, mask=patched_masks_bool)
        tokenizer_inputs = torch.cat([normed_inputs, patched_masks_bool.to(dtype=normed_inputs.dtype)], dim=-1)
        input_embeddings = self.input_ff_layer(tokenizer_inputs)
        position_ids = torch.arange(input_embeddings.shape[1], device=input_embeddings.device).unsqueeze(0) - patched_masks_bool[..., -1].to(torch.int32).sum(dim=-1, keepdim=True)
        position_embeddings = self.rotary_emb(input_embeddings, position_ids)
        hidden_states = input_embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings, attention_mask=None, position_ids=position_ids)
        
        from transformers.models.timesfm2_5.modeling_timesfm2_5 import TimesFm2_5Output
        num_patches = context_mu.shape[1]
        return TimesFm2_5Output(
            last_hidden_state=hidden_states,
            loc=context_mu[:, -1:].squeeze(1),
            scale=torch.clamp(context_sigma[:, -1:], min=self.tolerance).squeeze(1)
        )

    wrapper.model.forward = dynamic_inner_forward.__get__(wrapper.model, wrapper.model.__class__)

    output_dir = "../timesfm-onnx-dynamic"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model.onnx")
    
    dummy_input = torch.randn(1, 512)
    horizon = config.horizon_length
    dummy_padding = torch.zeros(1, 512 + horizon, dtype=torch.long)
    
    print(f"Exporting to {output_path} with pure dynamic batch and seq_len...")
    torch.onnx.export(
        wrapper,
        (dummy_input, dummy_padding),
        output_path,
        input_names=["past_values", "past_values_padding"],
        output_names=["mean_predictions", "full_predictions"],
        dynamic_axes={
            "past_values": {0: "batch_size", 1: "seq_len"},
            "past_values_padding": {0: "batch_size", 1: "seq_len_plus_horizon"},
            "mean_predictions": {0: "batch_size"},
            "full_predictions": {0: "batch_size"}
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False
    )
    print("Export complete!")

if __name__ == "__main__":
    main()
