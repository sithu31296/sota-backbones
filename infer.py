import torch
import argparse
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
from models import *
from datasets import ImageNet


class ModelInference:
    def __init__(self, model: str, variant: str, checkpoint: str, size: int) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # dataset class labels (change to trained dataset labels) (can provide a list of labels here)
        self.labels = ImageNet.CLASSES
        # model initialization
        self.model = eval(model)(variant, checkpoint, len(self.labels), size)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.preprocess = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Resize((size, size)),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def __call__(self, img_path: str) -> str:
        # read image
        image = io.read_image(img_path)
        # preprocess
        image = self.preprocess(image).to(self.device)
        # model pass
        with torch.inference_mode():
            pred = self.model(image)
        # postprocess
        cls_name = self.labels[pred.argmax()]
        return cls_name

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='assests/dog.jpg')
    parser.add_argument('--model', type=str, default='VAN')
    parser.add_argument('--variant', type=str, default='S')
    parser.add_argument('--checkpoint', type=str, default='/home/sithu/Documents/weights/backbones/van/van_small_811.pth.tar')
    parser.add_argument('--size', type=int, default=224)
    args = vars(parser.parse_args())

    source = args.pop('source')
    file_path = Path(source)
    model = ModelInference(**args)

    if file_path.is_file():
        cls_name = model(str(file_path))
        print(f"{file_path} >>>>> {cls_name.capitalize()}")
    else:
        files = file_path.glob('*jpg')
        for file in files:
            cls_name = model(str(file))
            print(f"{file} >>>>> {cls_name.capitalize()}")