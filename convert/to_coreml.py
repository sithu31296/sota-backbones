import torch
import argparse
import coremltools as ct
from pathlib import Path
import sys
sys.path.insert(0, '.')
from models import *


def convert(model, variant, num_classes, checkpoint, size):
    """
    Warning!!!! CoreML conversion will not work on Windows
    """
    # create random input and initialize model
    inputs = torch.randn(1, 3, *size)    
    pt_model = eval(model)(variant, checkpoint, num_classes, max(*size))
    pt_model.eval()

    # convert to torchscript model
    jit_model = torch.jit.trace(pt_model, inputs)

    # convert to coreml model
    core_ml_model = ct.convert(
        jit_model,
        inputs=[ct.ImageType('input', inputs.shape)],
        convert_to="neuralnetwork"
    )

    # create save file  
    checkpoint = Path(checkpoint)
    save_path = checkpoint.parent / f"{checkpoint.name.split('.')[0]}.mlmodel"

    # save model
    core_ml_model.save(save_path)
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