import torch
from transformers import TimesFm2_5ModelForPrediction
import os

def main():
    model_id = "google/timesfm-2.5-200m-transformers"
    print(f"Loading model {model_id}...")
    model = TimesFm2_5ModelForPrediction.from_pretrained(model_id)
    model.eval()

    # The prediction model's forward takes a Sequence[torch.Tensor]
    # We provide a single tensor in a list as the dummy input
    dummy_input = [torch.randn(1, 512)]
    
    output_dir = "../timesfm-onnx-2"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model.onnx")

    print(f"Exporting base model to {output_path}...")
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["past_values"],
        output_names=["mean_predictions", "full_predictions"],
        opset_version=18,
        do_constant_folding=True
    )
    print("Export complete!")

if __name__ == "__main__":
    main()
