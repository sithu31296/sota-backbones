import argparse
import torch
import yaml
import time
from tqdm import tqdm
from tabulate import tabulate
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, '.')
from datasets import get_dataset, get_sampler
from datasets.transforms import get_transforms
from models import get_model
from utils.utils import fix_seeds, time_sync, setup_cudnn, setup_ddp
from utils.schedulers import get_scheduler
from utils.losses import get_loss
from utils.optimizers import get_optimizer
from val import evaluate


def main(cfg):
    start = time_sync()
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)

    device = torch.device(cfg['DEVICE'])
    kd_enable = cfg['KD']['ENABLE']
    ddp_enable = cfg['TRAIN']['DDP']['ENABLE']
    epochs = cfg['TRAIN']['EPOCHS']
    best_top1_acc, best_top5_acc = 0.0, 0.0
    gpu = setup_ddp()

    # augmentations
    train_transform, val_transform = get_transforms(cfg)

    # dataset
    train_dataset, val_dataset = get_dataset(cfg, train_transform, val_transform)

    # dataset sampler
    train_sampler, val_sampler = get_sampler(cfg, train_dataset, val_dataset)
    
    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], num_workers=cfg['TRAIN']['WORKERS'], drop_last=True, pin_memory=True, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=cfg['EVAL']['WORKERS'], pin_memory=True, sampler=val_sampler)
    
    # training model
    model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], None, len(train_dataset.CLASSES), cfg['TRAIN']['IMAGE_SIZE'][0])
    model = model.to(device)

    if ddp_enable:
        model = DDP(model, device_ids=[gpu])

    # knowledge distillation teacher model
    if kd_enable:
        teacher_model = get_model(cfg['KD']['TEACHER']['NAME'], cfg['KD']['TEACHER']['VARIANT'], cfg['KD']['TEACHER']['PRETRAINED'], cfg['DATASET']['NUM_CLASSES'], cfg['TRAIN']['IMAGE_SIZE'][0])
        teacher_model = teacher_model.to(device)
        teacher_model.eval()

    # loss function, optimizer, scheduler, AMP scaler, tensorboard writer
    loss_fn = get_loss(cfg)
    optimizer = get_optimizer(model, cfg['TRAIN']['OPTIMIZER']['NAME'], cfg['TRAIN']['OPTIMIZER']['LR'], cfg['TRAIN']['OPTIMIZER']['WEIGHT_DECAY'])
    scheduler = get_scheduler(cfg, optimizer)
    scaler = GradScaler(enabled=cfg['TRAIN']['AMP'])
    writer = SummaryWriter(save_dir / 'logs')
    iters_per_epoch = int(len(train_dataset)) / cfg['TRAIN']['BATCH_SIZE']

    for epoch in range(1, epochs+1):
        model.train()
        
        if ddp_enable: train_sampler.set_epoch(epoch)
        train_loss = 0.0
        pbar = tqdm(enumerate(train_dataloader), total=iters_per_epoch, desc=f"Epoch: [{epoch}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {cfg['TRAIN']['LR']:.8f} Loss: {0:.8f}")
        
        for iter, (img, lbl) in pbar:
            img = img.to(device)
            lbl = lbl.to(device)

            optimizer.zero_grad()

            if kd_enable:
                with torch.no_grad():
                    pred_teacher = teacher_model(img)

            with autocast(enabled=cfg['TRAIN']['AMP']):
                pred = model(img)
                loss = loss_fn(pred, pred_teacher, lbl) if kd_enable else loss_fn(pred, lbl)

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr = scheduler.get_last_lr()[0]
            train_loss += loss.item() * img.shape[0]

            pbar.set_description(f"Epoch: [{epoch}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {loss.item():.8f}")

        train_loss /= len(train_dataset)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/lr', lr, epoch)
        writer.flush()

        scheduler.step()
        torch.cuda.empty_cache()

        if (epoch % cfg['TRAIN']['EVAL_INTERVAL'] == 0) and (epoch >= cfg['TRAIN']['EVAL_INTERVAL']):
            # evaluate the model
            test_loss, top1_acc, top5_acc = evaluate(val_dataloader, model, device, F.cross_entropy()) if kd_enable else evaluate(val_dataloader, model, device, loss_fn)

            print(f"Top-1 Accuracy: {top1_acc:>0.1f} Top-5 Accuracy: {top5_acc:>0.1f} Avg Loss: {test_loss:>8f}")
            writer.add_scalar('val/loss', test_loss, epoch)
            writer.add_scalar('val/Top1_Acc', top1_acc, epoch)
            writer.add_scalar('val/Top5_Acc', top5_acc, epoch)
            writer.flush()

            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                best_top5_acc = top5_acc
                torch.save(model.module.state_dict() if ddp_enable else model.state_dict(), save_dir / f"{cfg['MODEL']['NAME']}{cfg['MODEL']['SUB_NAME']}.pth")
            print(f"Best Top-1 Accuracy: {best_top1_acc:>0.1f} Best Top-5 Accuracy: {best_top5_acc:>0.5f}")
        
    writer.close()
    pbar.close()

    # results table
    table = [[f"{cfg['MODEL']['NAME']}-{cfg['MODEL']['SUB_NAME']}", best_top1_acc, best_top5_acc]]

    # evaluating teacher model
    if kd_enable:
        _, teacher_top1_acc, teacher_top5_acc = evaluate(val_dataloader, teacher_model, device)
        table.append([f"{cfg['KD']['TEACHER']['NAME']}-{cfg['KD']['TEACHER']['SUB_NAME']}", teacher_top1_acc, teacher_top5_acc])
        
    end = time.gmtime(time_sync() - start)
    total_time = time.strftime("%H:%M:%S", end)

    print(tabulate(table, headers=['Top-1 Accuracy', 'Top-5 Accuracy'], numalign='right'))
    print(f"Total Training Time: {total_time}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Experiment configuration file name')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    fix_seeds(123)
    setup_cudnn()
    main(cfg)