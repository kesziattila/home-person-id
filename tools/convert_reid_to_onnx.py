import torch
import argparse
import os
import sys

def convert_to_onnx(pth_path, onnx_path, arch='osnet_ain_x1_0'):
    """
    Convert a torchreid .pth model to ONNX format.
    """
    try:
        from torchreid import models
    except ImportError:
        print("Error: torchreid not installed. Please install with 'pip install torchreid'")
        return

    print(f"Building model: {arch}")
    # OSNet models usually use 1000 classes if trained on ImageNet, 
    # but for feature extraction it doesn't strictly matter as we strip the classifier.
    model = models.build_model(name=arch, num_classes=1000, pretrained=False)
    
    print(f"Loading weights from: {pth_path}")
    state_dict = torch.load(pth_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Remove classifier weights as we only want the feature extractor
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
    
    # Load state dict
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Weights loaded: {msg}")
    
    model.eval()
    
    # Create dummy input (Standard Re-ID size is 256x128)
    dummy = torch.randn(1, 3, 256, 128)
    
    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model, 
        dummy, 
        onnx_path, 
        # opset_version=11,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export successful!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert torchreid .pth model to ONNX")
    parser.add_argument("pth_path", help="Path to input .pth file")
    parser.add_argument("--output", "-o", help="Path to output .onnx file (default: input name with .onnx)")
    parser.add_argument("--arch", "-a", default="osnet_ain_x1_0", help="Architecture name (default: osnet_ain_x1_0)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pth_path):
        print(f"Error: File not found: {args.pth_path}")
        sys.exit(1)
        
    output_path = args.output
    if not output_path:
        output_path = os.path.splitext(args.pth_path)[0] + ".onnx"
        
    convert_to_onnx(args.pth_path, output_path, args.arch)
