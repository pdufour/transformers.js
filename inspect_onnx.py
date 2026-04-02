import onnx

def main():
    path = "../timesfm-onnx-dynamic/onnx/model.onnx"
    print(f"Loading ONNX model from {path}...")
    model = onnx.load(path)
    
    print("\nInputs:")
    for input in model.graph.input:
        shape = []
        for dim in input.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(str(dim.dim_value))
            elif dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            else:
                shape.append("?")
        print(f"  {input.name}: [{', '.join(shape)}]")

    print("\nOutputs:")
    for output in model.graph.output:
        shape = []
        for dim in output.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(str(dim.dim_value))
            elif dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            else:
                shape.append("?")
        print(f"  {output.name}: [{', '.join(shape)}]")

if __name__ == "__main__":
    main()
