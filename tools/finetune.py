import argparse
import torch
import yaml
import time
import multiprocessing as mp
from pprint import pprint
from tqdm import tqdm
from tabulate import tabulate
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, '.')
from datasets import get_sampler
from datasets.transforms import get_train_transforms, get_val_transforms
from models import get_model
from utils.utils import fix_seeds, setup_cudnn, setup_ddp, cleanup_ddp
from utils.schedulers import get_scheduler
from utils.losses import LabelSmoothCrossEntropy, CrossEntropyLoss
from utils.optimizers import get_optimizer
from val import evaluate


def main(cfg, gpu, save_dir):
    start = time.time()
    best_top1_acc, best_top5_acc = 0.0, 0.0
    num_workers = mp.cpu_count()
    device = torch.device(cfg['DEVICE'])
    train_cfg = cfg['TRAIN']
    eval_cfg = cfg['EVAL']
    optim_cfg = cfg['OPTIMIZER']
    epochs = train_cfg['EPOCHS']
    lr = optim_cfg['LR']

    # augmentations
    train_transforms = get_train_transforms(train_cfg['IMAGE_SIZE'])
    val_transforms = get_val_transforms(eval_cfg['IMAGE_SIZE'])

    # dataset
    train_dataset = CIFAR10(cfg['DATASET']['ROOT'], True, train_transforms)
    val_dataset = CIFAR10(cfg['DATASET']['ROOT'], False, val_transforms)

    # dataset sampler
    train_sampler, val_sampler = get_sampler(train_cfg['DDP'], train_dataset, val_dataset)
    
    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=True, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=True, sampler=val_sampler)

    # training model
    model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], None, len(train_dataset.classes), train_cfg['IMAGE_SIZE'][0])
    ## some models pretrained weights have extra keys, so check them in pretrained loading in model construction
    pretrained_dict = torch.load(cfg['MODEL']['PRETRAINED'], map_location='cpu')
    pretrained_dict.popitem()
    pretrained_dict.popitem()
    model.load_state_dict(pretrained_dict, strict=False)

    if cfg['MODEL']['FREEZE']:
        for n, p in model.named_parameters():
            if 'head' not in n:
                p.requires_grad_ = False
                
    model = model.to(device)

    if train_cfg['DDP']: model = DDP(model, device_ids=[gpu])

    # loss function, optimizer, scheduler, AMP scaler, tensorboard writer
    loss_fn = LabelSmoothCrossEntropy()
    optimizer = get_optimizer(model, optim_cfg['NAME'], optim_cfg['LR'], optim_cfg['DECAY'])
    scheduler = get_scheduler(cfg['SCHEDULER'], optimizer)
    scaler = GradScaler(enabled=train_cfg['AMP'])
    writer = SummaryWriter(save_dir / 'logs')
    iters_per_epoch = len(train_dataset) // train_cfg['BATCH_SIZE']

    for epoch in range(epochs):
        model.train()
        
        if train_cfg['DDP']: train_sampler.set_epoch(epoch)
        train_loss = 0.0
        pbar = tqdm(enumerate(train_dataloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")
        
        for iter, (img, lbl) in pbar:
            img = img.to(device)
            lbl = lbl.to(device)

            optimizer.zero_grad()

            with autocast(enabled=train_cfg['AMP']):
                pred = model(img)
                loss = loss_fn(pred, lbl)

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr = scheduler.get_last_lr()[0]
            train_loss += loss.item() * img.shape[0]

            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {loss.item():.8f}")

        train_loss /= iter + 1
        writer.add_scalar('train/loss', train_loss, epoch)
        scheduler.step()
        torch.cuda.empty_cache()

        if (epoch+1) % cfg['TRAIN']['EVAL_INTERVAL'] == 0 or (epoch+1) == epochs:
            # evaluate the model
            top1_acc, top5_acc = evaluate(val_dataloader, model, device) 

            print(f"Top-1 Accuracy: {top1_acc:>0.1f} Top-5 Accuracy: {top5_acc:>0.1f}")
            writer.add_scalar('val/Top1_Acc', top1_acc, epoch)
            writer.add_scalar('val/Top5_Acc', top5_acc, epoch)

            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                best_top5_acc = top5_acc
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['VARIANT']}.pth")
            print(f"Best Top-1 Accuracy: {best_top1_acc:>0.1f} Best Top-5 Accuracy: {best_top5_acc:>0.5f}")
        
    writer.close()
    pbar.close()
        
    end = time.gmtime(time.time() - start)
    total_time = time.strftime("%H:%M:%S", end)

    table = [
        ['Best Top-1 Accuracy', best_top1_acc],
        ['Best Top-5 Accuracy', best_top5_acc],
        ['Total Training Time', total_time]
    ]

    print(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Experiment configuration file name')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    pprint(cfg)
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    fix_seeds(123)
    setup_cudnn()
    gpu = setup_ddp()
    main(cfg, gpu, save_dir)
    cleanup_ddp()