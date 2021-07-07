import torch
import torch.onnx
import argparse
import onnx
import yaml
from pathlib import Path
import sys
sys.path.insert(0, '.')
from models import get_model


def main(cfg):
    save_dir = Path(cfg['SAVE_DIR'])
    if not save_dir.exists(): save_dir.mkdir()
    model_path = save_dir / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['VARIANT']}.onnx"
    model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], None, cfg['DATASET']['NUM_CLASSES'], cfg['TRAIN']['IMAGE_SIZE'][0])
    model.eval()
    inputs = torch.randn(1, 3, *cfg['TRAIN']['IMAGE_SIZE'])

    torch.onnx.export(
        model,
        inputs,
        model_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=13
    )

    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"Finished converting and Saved model at {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/defaults.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)