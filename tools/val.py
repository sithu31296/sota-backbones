import argparse
import yaml
import torch
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchvision import transforms as T

import sys
sys.path.insert(0, '.')
from models import choose_models
from datasets.imagenet import ImageNet
from utils.utils import fix_seeds
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
        top1_acc += acc1
        top5_acc += acc5
        
    test_loss /= len(dataloader.dataset)
    top1_acc /= len(dataloader.dataset)
    top5_acc /= len(dataloader.dataset)

    return test_loss, 100*top1_acc, 100*top5_acc


def main(cfg):
    fix_seeds(cfg['TRAIN']['SEED'])
    device = torch.device(cfg['DEVICE'])

    model = choose_models(cfg['MODEL']['NAME'])(cfg['MODEL']['SUB_NAME'], pretrained=cfg['EVAL']['MODEL_PATH'], num_classes=cfg['DATASET']['NUM_CLASSES'], image_size=cfg['EVAL']['IMAGE_SIZE'][0])   
    model = model.to(device)

    val_transform = T.Compose(
        T.Resize(tuple(map(lambda x: int(x / 0.9), cfg['EVAL']['IMAGE_SIZE']))),
        T.CenterCrop(cfg['EVAL']['IMAGE_SIZE']),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )

    val_dataset = ImageNet(cfg['DATASET']['ROOT'], split='val', transform=val_transform)
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

    main(cfg)
