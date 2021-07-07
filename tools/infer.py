import torch
import argparse
import yaml
from pathlib import Path
from torchvision import io
from torchvision import transforms

import sys
sys.path.insert(0, '.')
from models import get_model
from datasets.imagenet import CLASSES
from utils.utils import time_synchronized


class Model:
    def __init__(self, cfg) -> None:
        self.device = torch.device(cfg['TEST']['DEVICE'])
        self.model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], cfg['MODEL_PATH'], cfg['DATASET']['NUM_CLASSES'], cfg['TEST']['IMAGE_SIZE'][0])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.img_transforms = transforms.Compose(
            transforms.Resize(cfg['TEST']['IMAGE_SIZE']),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )

    @torch.no_grad()
    def predict(self, image) -> str:
        image = image.float()
        image /= 255
        image = self.img_transforms(image).unsqueeze(0).to(self.device)
        
        start = time_synchronized()
        pred = self.model(image)
        end = time_synchronized()
        print(f"Model Inference Time: {(end-start)*1000}ms")

        cls_id = torch.argmax(pred)

        return CLASSES[cls_id]

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/defaults.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    file_path = Path(cfg['TEST']['FILE'])
    results_path = Path(cfg['TEST']['OUTPUT'])
    if not results_path.exists(): results_path.mkdir()

    model = Model(cfg)

    if cfg['TEST']['MODE'] == 'image':
        if file_path.is_file():
            image = io.read_image(file_path)
            cls_name = model.predict(image)
            print(f"File: {str(file_path)} >>>>> {cls_name.capitalize()}")
        else:
            files = file_path.glob('*jpg')
            for file in files:
                image = io.read_image(file_path / file)
                cls_name = model.predict(image)
                print(f"File: {str(file)} >>>>> {cls_name.capitalize()}")
    else:
        raise NotImplementedError
