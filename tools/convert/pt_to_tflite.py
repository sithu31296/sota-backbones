import torch
import torch.onnx
import argparse
import onnx
import yaml
import os
import shutil
from pathlib import Path
from onnx_tf.backend import prepare
from onnxsim import simplify
import tensorflow as tf
import sys
sys.path.insert(0, '.')
from models import get_model


def main(cfg):
    save_dir = Path(cfg['SAVE_DIR'])
    if not save_dir.exists(): save_dir.mkdir()
    onnx_model_path = save_dir / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['VARIANT']}.onnx"
    tf_model_path = save_dir / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['VARIANT']}"
    if tf_model_path.exists(): shutil.rmtree(tf_model_path)
    tflite_model_path = save_dir / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['VARIANT']}.tflite"
    model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], cfg['MODEL_PATH'], cfg['DATASET']['NUM_CLASSES'], cfg['TRAIN']['IMAGE_SIZE'][0])
    model.eval()
    inputs = torch.randn(1, 3, *cfg['TRAIN']['IMAGE_SIZE'])

    torch.onnx.export(
        model,
        inputs,
        onnx_model_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=13
    )

    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # optimize with onnxsim
    onnx_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"

    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(str(tf_model_path))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()

    with open(str(tflite_model_path), 'wb') as f:
        f.write(tflite_model)

    shutil.rmtree(tf_model_path)
    os.remove(onnx_model_path)
    
    print(f"Finished converting and Saved model at {tflite_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/defaults.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)