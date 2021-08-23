import argparse
import yaml
import torch
import multiprocessing as mp
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '.')
from models import get_model
from datasets import ImageNet
from datasets.transforms import get_val_transforms
from utils.utils import setup_cudnn
from utils.metrics import accuracy


@torch.no_grad()
def evaluate(dataloader, model, device):
    print('Evaluating...')
    model.eval()
    top1_acc, top5_acc = 0.0, 0.0

    for img, lbl in tqdm(dataloader):
        img = img.to(device)
        lbl = lbl.to(device)

        pred = model(img)
        acc1, acc5 = accuracy(pred, lbl, topk=(1, 5))
        top1_acc += acc1 * img.shape[0]
        top5_acc += acc5 * img.shape[0]
        
    top1_acc /= len(dataloader.dataset)
    top5_acc /= len(dataloader.dataset)

    return 100*top1_acc, 100*top5_acc


def main(cfg):
    device = torch.device(cfg['DEVICE'])
    num_workers = mp.cpu_count()

    transform = get_val_transforms(cfg)
    dataset = ImageNet(cfg['DATASET']['ROOT'], 'val', transform)
    dataloader = DataLoader(dataset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=num_workers, pin_memory=True)

    model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], cfg['EVAL']['MODEL_PATH'], len(dataset.classes), cfg['EVAL']['IMAGE_SIZE'][0])
    model = model.to(device)

    top1_acc, top5_acc = evaluate(dataloader, model, device)

    table = [
        ['Top-1 Accuracy', top1_acc],
        ['Top-5 Accuracy', top5_acc],
    ]
    print(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/train.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(cfg)
