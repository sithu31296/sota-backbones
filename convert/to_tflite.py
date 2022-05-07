import os
import torch
import onnx
import argparse
import shutil
from pathlib import Path
from onnxsim import simplify
import sys
sys.path.insert(0, '.')
from models import *


def convert(model, variant, num_classes, checkpoint, size, precision):
    # create random input and initialize model
    inputs = torch.randn(1, 3, *size)    
    pt_model = eval(model)(variant, checkpoint, num_classes, max(*size))
    pt_model.eval()

    # create save file  
    checkpoint = Path(checkpoint)
    onnx_path = checkpoint.parent / f"{checkpoint.name.split('.')[0]}_openvino_tmp.onnx"
    openvino_path = checkpoint.parent / f'{checkpoint.name.split(".")[0]}_openvino_{precision}_tmp'
    openvino_xml_path = checkpoint.parent / f'{checkpoint.name.split(".")[0]}_openvino_{precision}_tmp' / f'{checkpoint.name.split(".")[0]}_openvino_tmp.xml'
    tf_path = checkpoint.parent / f'{checkpoint.name.split(".")[0]}_tmp_tf'
    tflite_path = checkpoint.parent / f'{checkpoint.name.split(".")[0]}.tflite'

    # export to onnx
    torch.onnx.export(
        pt_model,
        inputs,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=13
    )

    # check onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # simplify onnx model
    onnx_model, check = simplify(onnx_model)
    onnx.save(onnx_model, onnx_path)

    assert check, "Simplified ONNX model could not be validated"

    mo_command = f"""
    mo
    --input_model "{onnx_path}"
    --output_dir "{str(openvino_path)}"
    --input_shape "[1,3,{size[0]},{size[1]}]"
    --data_type {precision}
    """
    mo_command = " ".join(mo_command.split())
    os.system(mo_command)
    os.remove(onnx_path)

    optf_command = f"""
    openvino2tensorflow
    --model_path "{str(openvino_xml_path)}"
    --model_output_path "{str(tf_path)}"
    --output_saved_model
    """
    os.system(optf_command)
    shutil.rmtree(openvino_path)
    
    print(f"Finished converting and saved model at `{tflite_path}`")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="VAN")
    parser.add_argument('--variant', type=str, default='S')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--checkpoint', type=str, default='C:/Users/sithu/Documents/weights/backbones/van/van_small_811.pth.tar')
    parser.add_argument('--size', type=list, default=[224, 224])
    parser.add_argument('--precision', type=str, default="FP32", help="FP16 or FP32")
    args = vars(parser.parse_args())
    convert(**args)