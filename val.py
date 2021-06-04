import argparse
import yaml
import torch
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchvision import transforms

from models import choose_models
from datasets.imagenet import ImageNet
from utils.utils import setup_cuda
from utils.metrics import accuracy


@torch.no_grad()
def evaluate(dataloader, model, device, loss_fn = None):
    model.eval()
    test_loss, top1_acc, top5_acc = 0.0, 0.0, 0.0

    for iter, (img, lbl) in enumerate(dataloader):
        img = img.to(device)
        lbl = lbl.to(device)

        pred = model(img)
        
        if loss_fn:
            test_loss += loss_fn(pred, lbl).item()

        acc1, acc5 = accuracy(pred, lbl, topk=(1, 5))
        top1_acc += acc1
        top5_acc += acc5
        
    test_loss /= iter + 1
    top1_acc /= iter + 1
    top5_acc /= iter + 1

    return test_loss, 100*top1_acc, 100*top5_acc


def main(cfg):
    setup_cuda()
    device = torch.device(cfg['DEVICE'])

    model = choose_models[cfg['MODEL']['NAME']](cfg['MODEL']['SUB_NAME'], pretrained=cfg['EVAL']['MODEL_PATH'], num_classes=cfg['DATASET']['NUM_CLASSES'])   
    model = model.to(device)

    val_transform = transforms.Compose(
        transforms.CenterCrop(cfg['EVAL']['IMAGE_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
