import argparse
import yaml
import torch
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '.')
from models import get_model
from datasets import get_dataset
from datasets.transforms import get_transforms
from utils.utils import fix_seeds, setup_cudnn
from utils.metrics import accuracy


@torch.no_grad()
def evaluate(dataloader, model, device, loss_fn = None):
    print('Evaluating...')
    model.eval()
    test_loss, top1_acc, top5_acc = 0.0, 0.0, 0.0

    for img, lbl in tqdm(dataloader):
        img = img.to(device)
        lbl = lbl.to(device)

        pred = model(img)
        
        if loss_fn:
            test_loss += loss_fn(pred, lbl).item()

        acc1, acc5 = accuracy(pred, lbl, topk=(1, 5))
        top1_acc += acc1 * img.shape[0]
        top5_acc += acc5 * img.shape[0]
        
    test_loss /= len(dataloader.dataset)
    top1_acc /= len(dataloader.dataset)
    top5_acc /= len(dataloader.dataset)

    return test_loss, 100*top1_acc, 100*top5_acc


def main(cfg):
    device = torch.device(cfg['DEVICE'])

    model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], cfg['MODEL_PATH'], cfg['DATASET']['NUM_CLASSES'], cfg['EVAL']['IMAGE_SIZE'][0])
    model = model.to(device)

    _, val_transform = get_transforms(cfg)
    _, val_dataset = get_dataset(cfg, val_transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=cfg['EVAL']['WORKERS'], pin_memory=True)

    _, top1_acc, top5_acc = evaluate(val_dataloader, model, device)

    table = [
        ['Top-1 Accuracy', top1_acc],
        ['Top-5 Accuracy', top5_acc],
    ]

    print(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/defaults.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    fix_seeds(cfg['TRAIN']['SEED'])
    setup_cudnn()
    main(cfg)
