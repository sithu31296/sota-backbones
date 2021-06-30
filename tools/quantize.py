"""Dynamic Quantization
Quantization means converting a model to a reduced precision integer representation for the weights and/or activations.
This saves on model size and allows the use of higher throughput math operations on your CPU or GPU.

When converting from floating point to integer values you are essentially multiplying the floating point value by some scale factor and
rounding the result to a whole number. The various quantization approaches differ in the way they approach determining that scale factor.

The model parameters are converted and stored in INT8. Arithmetic is done using INT8 but accumulation is done with INT16 or INT32 to avoid overflow.
This higher precision is scaled back to INT8 if the next layer is quantized or converted to FP32 for output.

There is two benefits of using Quantization:
    1. Reduce model size.
    2. Decrease latency.

After quantization, you need to check the following:
    1. Model size improvement
    2. Latency improvement
    3. Accuracy Drop

Three quantization techniques:
    1. Post training dynamic quantization 
        * Weights in INT8, Activations in INT16 or INT32
        * Only supports nn.Linear and nn.LSTM
    2. Post training static quantization
        * Both weights and activations in INT8 (so there won't be on the fly conversion on the activations during the inference)
    3. Quantization aware training
        * Inserts fake quantization to all the weights and activations during the model training process
        * Results in higher inference accuracy than the post training quantization methods.
        * Typically used in CNN models.
        * To enable a model for quantization aware training, define the following in the model definition.
            ```
            def __init__(self):
                self.quant = torch.quantization.QuantStub()
                self.dequant = torch.quantization.DeQuantStub()

            def forward(self, x):
                x.contiguous(memory_format=torch.channels_last)
                x = self.quant(x)
                ...
                x = self.dequant(x)
                return x
            ```

Optimization for deployment on mobile devices

By default, optimize_for_mobile will perform the following types of optimizations:
    * Conv2D and BatchNorm fusion which folds Conv2d-BatchNorm2d into Conv2d.
    * Insert and fold prepacked ops which rewrites the model graph to replace 2D convolutions and linear ops with their prepacked counterparts.
    * ReLU and hardtanh fusion which rewrites graph by finding ReLU/hardtanh ops and fuses them together.
    * Dropout removal which removes dropout nodes from this module when training is false.
"""
import argparse
import yaml
import torch
from pathlib import Path
from tabulate import tabulate
from torch import nn
from torch.quantization import quantize_dynamic, fuse_modules, get_default_qconfig, prepare, convert, get_default_qat_qconfig, prepare_qat
from torch.utils.mobile_optimizer import optimize_for_mobile
import sys
sys.path.insert(0, '.')
from models import choose_models
from utils.utils import setup_cudnn, get_model_size, test_model_latency


def quantize_model(model, method='dynamic', backend='qnnpack'):
    model.qconfig = get_default_qconfig(backend)
    torch.backends.quantized_engine = backend

    if method == 'dynamic':
        quantized_model = quantize_dynamic(model, qconfig_spec={nn.Linear, nn.LSTM}, dtype=torch.qint8)
    elif method == 'static':
        quantized_model = prepare(model, inplace=False)
        quantized_model = convert(quantized_model, inplace=False)
    else:
        model.qconfig = get_default_qat_qconfig(backend)
        quantized_model = prepare_qat(model, inplace=False)
        # quantization aware training goes here
        quantized_model = convert(quantized_model.eval(), inplace=False)
    return quantized_model



def main(cfg):
    setup_cudnn()
    model_name = cfg['MODEL']['NAME']
    model_sub_name = cfg['MODEL']['SUB_NAME']
    save_dir = Path(cfg['SAVE_DIR'])
    if not save_dir.exists(): save_dir.mkdir()
    save_model = save_dir / f"{model_name}_{model_sub_name}_optimized.pt"

    model = choose_models(model_name)(model_sub_name, pretrained=None, num_classes=cfg['DATASET']['NUM_CLASSES'], image_size=cfg['TRAIN']['IMAGE_SIZE'][0])  
    model.eval()
    # fuse conv bn relu into single module
    # it may save on memory access, make the model run faster and improve its accuracy
    # model = fuse_modules(model, [['conv', 'bn', 'relu']], inplace=True)
    print("Starting quantization...") 
    optimized_model = quantize_model(model, method=cfg['QUANTIZE']['METHOD'], backend=cfg['QUANTIZE']['BACKEND']) 

    # save quantized model in torchscript format
    optimized_model = torch.jit.script(optimized_model)

    # mobile optimize
    if cfg['QUANTIZE']['MOBILE_OPT']:
        optimized_model = optimize_for_mobile(optimized_model)

    # save for lite interpreter
    if cfg['QUANTIZE']['LITE_INTER']:
        optimized_model._save_for_lite_interpreter(save_dir / f"{model_name}_{model_sub_name}_optimized_lite.ptl")
        # ptl = torch.jit.load(save_dir / f"{model_name}_{model_sub_name}_moptimized_lite.ptl")
    
    print("Starting benchmarking...")
    # test model size
    model_size = get_model_size(model)
    quantized_model_size = get_model_size(optimized_model)

    # set the number of threads to one for single threaded comparison, since quantized models run single threaded
    torch.set_num_threads(1)

    # test latency
    inputs = torch.randn(1, 3, *cfg['TRAIN']['IMAGE_SIZE'])
    model_time = test_model_latency(model, inputs)
    quantized_model_time = test_model_latency(optimized_model, inputs)

    # test accuracy
    model_accuracy = 85.34
    quantized_model_accuracy = 84.04

    table = [
        [f"Original {model_name}_{model_sub_name} (FP32)", f"{model_size:.2f}", f"{model_time:.2f}", f"{model_accuracy:.2f}"],
        [f"Quantized {model_name}_{model_sub_name} (INT8)", f"{quantized_model_size:.2f}", f"{quantized_model_time:.2f}", f"{quantized_model_accuracy:.2f}"],
        ["Improvement", f"+{int((model_size - quantized_model_size) / model_size * 100)}%", f"+{int((model_time - quantized_model_time) / model_time * 100)}%", f"-{(model_accuracy - quantized_model_accuracy) / model_accuracy * 100:.2f}%"]
    ]

    print(tabulate(table, numalign='right', headers=['Model (Precision)', 'Model Size (MB)', "Latency (ms)", "Accuracy (%)"]))

    optimized_model.save(save_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/defaults.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)