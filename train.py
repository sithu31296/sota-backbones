import argparse
import torch
import yaml
import time
import os
from tqdm import tqdm
from tabulate import tabulate
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from datasets.imagenet import ImageNet
from models import choose_models
from utils.utils import fix_seeds, setup_cuda
from val import evaluate


def main(cfg):
    start = time.time()

    fix_seeds(cfg['TRAIN']['SEED'])
    setup_cuda()

    save_dir = Path(cfg['TRAIN']['SAVE_DIR'])
    if not save_dir.exists(): save_dir.mkdir()

    device = torch.device(cfg['DEVICE'])
    model_name = cfg['MODEL']['NAME']
    model_sub_name = cfg['MODEL']['SUB_NAME']

    model = choose_models[model_name](model_sub_name, num_classes=cfg['DATASET']['NUM_CLASSES'])    
    model = model.to(device)

    train_transform = transforms.Compose(
        transforms.RandomSizedCrop(cfg['TRAIN']['IMAGE_SIZE']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )

    val_transform = transforms.Compose(
        transforms.CenterCrop(cfg['EVAL']['IMAGE_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )

    train_dataset = ImageNet(cfg['DATASET']['ROOT'], split='train', transform=train_transform)
    val_dataset = ImageNet(cfg['DATASET']['ROOT'], split='val', transform=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=True, num_workers=cfg['TRAIN']['WORKERS'], drop_last=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=cfg['EVAL']['WORKERS'], pin_memory=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['TRAIN']['LR'])

    assert cfg['TRAIN']['STEP_LR']['STEP_SIZE'] < cfg['TRAIN']['EPOCHS'], "Step LR scheduler's step size must be less than number of epochs"
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg['TRAIN']['STEP_LR']['STEP_SIZE'], cfg['TRAIN']['STEP_LR']['GAMMA'])

    best_top1_acc, best_top5_acc = 0.0, 0.0
    epochs = cfg['TRAIN']['EPOCHS']
    iters_per_epoch = int(len(train_dataset)) / cfg['TRAIN']['BATCH_SIZE']
    model_save_path = save_dir / f"{model_name}{model_sub_name}.pth"

    writer = SummaryWriter(save_dir / 'logs')

    for epoch in range(1, epochs+1):
        model.train()

        train_loss = 0.0

        pbar = tqdm(enumerate(train_dataloader), total=iters_per_epoch, desc=f"Epoch: [{epoch}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {cfg['TRAIN']['LR']:.8f} Loss: {0:.8f}")
        
        for iter, (img, lbl) in pbar:
            img = img.to(device)
            lbl = lbl.to(device)

            # Compute prediction and loss
            pred = model(img)
            loss = loss_fn(pred, lbl).item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = scheduler.get_last_lr()
            train_loss += loss

            pbar.set_description(f"Epoch: [{epoch}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {loss:.8f}")

        train_loss /= iter + 1
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/lr', lr, epoch)

        scheduler.step()
        torch.cuda.empty_cache()

        if (epoch % cfg['TRAIN']['EVAL_INTERVAL'] == 0) and (epoch >= cfg['TRAIN']['EVAL_INTERVAL']):
            print('Evaluating...')

            test_loss, top1_acc, top5_acc = evaluate(val_dataloader, model, device, loss_fn)

            print(f"Top-1 Accuracy: {top1_acc:>0.1f} Top-5 Accuracy: {top5_acc:>0.1f} Avg Loss: {test_loss:>8f}")

            writer.add_scalar('val/loss', test_loss, epoch)
            writer.add_scalar('val/Top1_Acc', top1_acc, epoch)
            writer.add_scalar('val/Top5_Acc', top5_acc, epoch)

            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                best_top5_acc = top5_acc
                torch.save(model.state_dict(), model_save_path)

                print(f"Best Top-1 Accuracy: {best_top1_acc:>0.1f} Best Top-5 Accuracy: {best_top5_acc:>0.5f}")
        
    writer.close()
    pbar.close()

    end = time.gmtime(time.time() - start)
    total_time = time.strftime("%H:%M:%S", end)

    table = [
        ['Top-1 Accuracy', best_top1_acc],
        ['Top-5 Accuracy', best_top5_acc],
        ['Total Training Time', total_time]
    ]

    print(tabulate(table, numalign='right'))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Experiment configuration file name')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)