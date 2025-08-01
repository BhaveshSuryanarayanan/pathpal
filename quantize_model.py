import torch
from torch.quantization import quantize_dynamic
import torch.nn as nn

path_weights=r"C:\Users\khyat\Downloads\unet_weights.pth"
path_quantized_weights=r"C:\Users\khyat\Downloads\unet_weights_quantized.pth"

# Load the UNet model
model1_trial = UNET()  # Ensure UNET class is defined
model1_trial.load_state_dict(torch.load(path_weights), strict=False)
model1_trial.eval()  # Set model to evaluation mode

# Apply dynamic quantization
quantized_model = quantize_dynamic(
    model1_trial,  # Model to be quantized
    {torch.nn.Linear, torch.nn.Conv2d},  # Specify layers to quantize (Linear and Conv2d here)
    dtype=torch.qint8  # Use 8-bit integers
)

# Save the quantized model
torch.save(quantized_model.state_dict(), path_quantized_weights)

# Verify quantized model (Optional)
print("Quantized model saved at:", path_quantized_weights)

quantized_model = UNET()
quantized_model.load_state_dict(torch.load(path_quantized_weights))
quantized_model.eval()

# Dummy input for ONNX export (adjust size as per input)
dummy_input = torch.randn(1, 3, 256, 256)  # batch_size, channels, height, width

# Export the model to ONNX
onnx_path = r"C:\Users\khyat\Downloads\unet_quantized.onnx"
torch.onnx.export(
    quantized_model,              # Model
    dummy_input,                  # Dummy input tensor
    onnx_path,                    # Path to save the ONNX model
    export_params=True,           # Store trained parameters
    opset_version=12,             # ONNX version (adjust if needed)
    do_constant_folding=True,     # Optimize constants
    input_names=['input'],        # Input names
    output_names=['output'],      # Output names
    dynamic_axes={                # Dynamic axes for variable input sizes
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)