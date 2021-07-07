import argparse
import onnxruntime
import yaml
import time
import numpy as np
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, '.')
from datasets.imagenet import CLASSES


class ONNXInfer:
    def __init__(self, cfg) -> None:
        save_dir = Path(cfg['SAVE_DIR'])
        model_path = save_dir / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['VARIANT']}.onnx"

        # onnx model session
        self.session = onnxruntime.InferenceSession(str(model_path))

        # preprocess parameters
        self.size = cfg['TEST']['IMAGE_SIZE']
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)

    def preprocess(self, image) -> np.ndarray:
        # resize
        image = image.resize(self.size)

        # to numpy array and to channel first
        image = np.array(image, dtype=np.float32).transpose(2, 0, 1)

        # scale to [0.0, 1.0]
        image /= 255.0

        # normalize
        image -= self.mean
        image /= self.std

        # add batch dimension
        image = image[np.newaxis, ...]

        return image

    def postprocess(self, prob: np.ndarray) -> str:
        cls_id = np.argmax(prob)
        return CLASSES[cls_id]
        
    def predict(self, image) -> str:
        image = self.preprocess(image)

        start = time.time()
        pred = self.session.run(None, {
            self.session.get_inputs()[0].name: image
        })
        end = time.time()
        print(f"ONNX Model Inference Time: {(end-start)*1000:.2f}ms")

        cls_name = self.postprocess(pred)
        return cls_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/defaults.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    file_path = Path(cfg['TEST']['FILE'])
    model = ONNXInfer(cfg)

    if cfg['TEST']['MODE'] == 'image':
        if file_path.is_file():
            image = Image.open(file_path)
            cls_name = model.predict(image)
            print(f"File: {str(file_path)} >>>>> {cls_name.capitalize()}")
        else:
            files = file_path.glob('*jpg')
            for file in files:
                image = Image.open(file)
                cls_name = model.predict(image)
                print(f"File: {str(file)} >>>>> {cls_name.capitalize()}")
    else:
        raise NotImplementedError