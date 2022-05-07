import torch
import onnx
import argparse
from pathlib import Path
from onnxsim import simplify
import sys
sys.path.insert(0, '.')
from models import *


def convert(model, variant, num_classes, checkpoint, size):
    # create random input and initialize model
    inputs = torch.randn(1, 3, *size)    
    pt_model = eval(model)(variant, checkpoint, num_classes, max(*size))
    pt_model.eval()

    # create save file  
    checkpoint = Path(checkpoint)
    save_path = checkpoint.parent / f"{checkpoint.name.split('.')[0]}.onnx"

    # export to onnx
    torch.onnx.export(
        pt_model,
        inputs,
        save_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=13
    )

    # check onnx model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)

    # simplify onnx model
    onnx_model, check = simplify(onnx_model)
    onnx.save(onnx_model, save_path)

    assert check, "Simplified ONNX model could not be validated"
    print(f"Finished converting and saved model at `{save_path}`")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="VAN")
    parser.add_argument('--variant', type=str, default='S')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--checkpoint', type=str, default='/home/sithu/Documents/weights/backbones/van/van_small_811.pth.tar')
    parser.add_argument('--size', type=list, default=[224, 224])
    args = vars(parser.parse_args())
    convert(**args)