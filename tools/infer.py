import torch
import argparse
import yaml
from torch import Tensor
from pathlib import Path
from torchvision import io
from torchvision import transforms as T

import sys
sys.path.insert(0, '.')
from models import get_model
from datasets import ImageNet


class Model:
    def __init__(self, cfg) -> None:
        self.device = torch.device(cfg['DEVICE'])
        self.labels = ImageNet.CLASSES
        self.model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], cfg['MODEL']['PRETRAINED'], len(self.labels), cfg['TEST']['IMAGE_SIZE'][0])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.img_transforms = T.Compose([
            T.Resize(cfg['TEST']['IMAGE_SIZE']),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def preprocess(self, image: Tensor) -> Tensor:
        # scale to [0.0, 1.0]
        image = image.float()
        image /= 255
        # normalize
        image = self.img_transforms(image)
        # add batch dimension
        image = image.unsqueeze(0).to(self.device)
        return image

    def postprocess(self, prob: Tensor) -> str:
        cls_id = torch.argmax(prob)
        cls_name = self.labels[cls_id]
        return cls_name

    @torch.no_grad()
    def predict(self, img_fname: str) -> str:
        image = io.read_image(img_fname)
        image = self.preprocess(image)
        pred = self.model(image)
        cls_name = self.postprocess(pred)
        return cls_name

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/test.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    file_path = Path(cfg['TEST']['FILE'])
    model = Model(cfg)

    if file_path.is_file():
        cls_name = model.predict(str(file_path))
        print(f"File: {str(file_path)} >>>>> {cls_name.capitalize()}")
    else:
        files = file_path.glob('*jpg')
        for file in files:
            cls_name = model.predict(str(file))
            print(f"File: {str(file)} >>>>> {cls_name.capitalize()}")