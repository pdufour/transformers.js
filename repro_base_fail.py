import torch
from transformers import TimesFm2_5ModelForPrediction
import os
import traceback

def main():
    model_id = "google/timesfm-2.5-200m-transformers"
    print(f"Loading model {model_id}...")
    model = TimesFm2_5ModelForPrediction.from_pretrained(model_id)
    model.eval()

    # The prediction model's forward takes a Sequence[torch.Tensor]
    # This is exactly why it fails in a standard trace: it expects a list.
    dummy_input = [torch.randn(512)]
    
    print("\nAttempting standard torch.onnx.export on the base model...")
    try:
        # We try to export with the same settings Optimum would use
        torch.onnx.export(
            model,
            (dummy_input,),
            "test_base.onnx",
            input_names=["past_values"],
            opset_version=18
        )
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()
