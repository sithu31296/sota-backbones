import torch
import time
import argparse
from torchvision import io
from torchvision import transforms as T
import sys
sys.path.insert(0, '.')
from models import *
from datasets import ImageNet


class ModelInference:
    def __init__(self, model: str, variant: str, checkpoint: str, size: list, device:str) -> None:
        self.size = size
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        print(f"using {self.device}")
        # dataset class labels (change to trained dataset labels) (can provide a list of labels here)
        self.labels = ImageNet.CLASSES
        # model initialization
        self.model = eval(model)(variant, checkpoint, len(self.labels), size)
        self.model = self.model.to(self.device)
        self.model.eval()      

        self.preprocess = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Resize(size),
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

    def test_latency(self) -> None:
        total = 0
        inputs = torch.randn(1, 3, *self.size).to(self.device)
        for _ in range(100):
            start = time.time()
            with torch.inference_mode():
                _ = self.model(inputs)
            end = time.time()
            total += (end - start) * 1000
        print(f"Latency: {total // 100}ms")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='assests/dog.jpg')
    parser.add_argument('--model', type=str, default='VAN')
    parser.add_argument('--variant', type=str, default='S')
    parser.add_argument('--checkpoint', type=str, default='/home/sithu/Documents/weights/backbones/van/van_small_811.pth.tar')
    parser.add_argument('--size', type=list, default=[224, 224])
    parser.add_argument('--device', type=str, default='cuda')
    args = vars(parser.parse_args())

    source = args.pop('source')
    model = ModelInference(**args)
    cls_name = model(source)
    print(f"{source} >>>>> {cls_name.capitalize()}")
    model.test_latency()
