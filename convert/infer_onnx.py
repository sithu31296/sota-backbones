import argparse
import onnxruntime
import time
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, '.')
from datasets import ImageNet


class Inference:
    def __init__(self, model: str) -> None:
        # onnx model session
        self.session = onnxruntime.InferenceSession(model)
        self.labels = ImageNet.CLASSES
        
        # preprocess parameters
        model_inputs = self.session.get_inputs()[0]
        self.input_name = model_inputs.name
        self.img_size = model_inputs.shape[-2:]
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.resize(self.img_size)
        image = np.array(image, dtype=np.float32).transpose(2, 0, 1)
        image /= 255
        image -= self.mean
        image /= self.std
        image = image[np.newaxis, ...]
        return image

    def postprocess(self, prob: np.ndarray) -> str:
        id = np.argmax(prob)
        return self.labels[id]

    def test_latency(self) -> None:
        total = 0
        inputs = np.random.randn(1, 3, *self.img_size).astype(np.float32)
        for _ in range(100):
            start = time.time()
            _ = self.session.run(None, {self.input_name: inputs})
            end = time.time()
            total += (end - start) * 1000
        print(f"Latency: {total // 100}ms")

    def predict(self, img_path: str) -> int:
        image = Image.open(img_path).convert('RGB')
        image = self.preprocess(image)
        pred = self.session.run(None, {self.input_name: image})
        cls_name = self.postprocess(pred)
        return cls_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/home/sithu/Documents/weights/backbones/van/van_small_811.onnx')
    parser.add_argument('--source', type=str, default='assests/dog.jpg')
    args = parser.parse_args()

    session = Inference(args.model)
    cls_name = session.predict(args.source)
    print(f"{args.source} >>>>> {cls_name.capitalize()}")
    print("Starting Latency test...")
    session.test_latency()
