import argparse
import time
import numpy as np
from PIL import Image
from pathlib import Path
from openvino.inference_engine import IECore
import sys
sys.path.insert(0, '.')
from datasets import ImageNet


class Inference:
    def __init__(self, model: str, use_device: str = 'CPU', cache: bool = False) -> None:
        self.labels = ImageNet.CLASSES

        model_files = Path(model).iterdir()
        for f in model_files:
            if f.suffix == '.xml':
                model = str(f)
            elif f.suffix == '.bin':
                weights = str(f)
        
        ie = IECore()

        self.show_available_devices(ie, use_device)

        if use_device == "GPU" and cache == True:
            self.cache_model(ie)

        model = ie.read_network(model=model, weights=weights)

        self.input_info = next(iter(model.input_info))
        self.output_info = next(iter(model.outputs))
        self.precision = model.input_info[self.input_info].precision
        self.img_size = model.input_info[self.input_info].tensor_desc.dims[-2:]

        self.engine = ie.load_network(network=model, device_name=use_device)

        if self.precision == "FP16":
            self.precision = np.float16
        else:
            self.precision = np.float32
        
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)

    def show_available_devices(self, ie, use_device) -> None:
        devices = ie.available_devices
        print("Available Devices:")
        for device in devices:
            device_name = ie.get_metric(device_name=device, metric_name="FULL_DEVICE_NAME")
            print(f"{device}: {device_name}")

        use_device_name = ie.get_metric(device_name=use_device, metric_name="FULL_DEVICE_NAME")
        print(f"Currently using {use_device}: {use_device_name}")

    def cache_model(self, ie) -> None:
        print("Using Model Caching")
        cache_path = Path("model") / "model_cache"
        cache_path.mkdir(exist_ok=True, parents=True)
        ie.set_config({"CACHE_DIR": str(cache_path)}, device_name="GPU")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.resize(self.img_size)
        image = np.array(image, dtype=self.precision).transpose(2, 0, 1)
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
        inputs = np.random.randn(1, 3, *self.img_size).astype(self.precision)
        for _ in range(100):
            start = time.perf_counter()
            _ = self.engine.infer(inputs={self.input_info: inputs})[self.output_info]
            end = time.perf_counter()
            total += (end - start) * 1000
        print(f"Latency: {total // 100}ms")

    def predict(self, img_path: str) -> int:
        image = Image.open(img_path).convert('RGB')
        image = self.preprocess(image)
        pred = self.engine.infer(inputs={self.input_info: image})[self.output_info]
        cls_name = self.postprocess(pred)
        return cls_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/home/sithu/Documents/weights/backbones/van/van_small_811_openvino_FP32')
    parser.add_argument('--source', type=str, default='assests/dog.jpg')
    parser.add_argument('--device', type=str, default='CPU', help="CPU or GPU")
    parser.add_argument('--cache', type=bool, default=False, help="use Model Caching in GPU")
    args = parser.parse_args()

    session = Inference(args.model, args.device, args.cache)
    cls_name = session.predict(args.source)
    print(f"{args.source} >>>>> {cls_name.capitalize()}")
    print("Starting Latency test...")
    session.test_latency()
