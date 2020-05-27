import numpy as np
import os
from datasets.characeter_dataset import CharacterDateset
from darknet53 import DarkNet
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from utils import torch_utils
import math
from net import resnet34
from torch.utils.tensorboard import SummaryWriter
from absl import logging,flags,app
mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed

  
flags.DEFINE_boolean('enable_pretrained',True,'Enable pretrained model')
flags.DEFINE_boolean('enable_converted',False,'Enable convert Image')
flags.DEFINE_string('ckpt_dir','./weights/best_model_bk.pt','pretrained model dir')
flags.DEFINE_integer('epoch',300,'number of epoch to train')
flags.DEFINE_float('lr',0.001,'learning rate')
flags.DEFINE_integer('batch',128,'batch size')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.xavier_normal_(m.bias.data)


def train(model, loss_fn, optimizer, lr_schedule, train_loader, val_loader,writer, device,num_epochs=1):
    nb = len(train_loader)
    best_loss = 1
    best_val_acc = 0
    best_val_loss = 10
    n_iter = 0
    for epoch in range(num_epochs):
        # print(f"Starting epoch {epoch + 1}/{num_epochs}")
        model.train()
        pbar = tqdm(enumerate(train_loader), total=nb)
        num_correct = 0
        num_samples = 0

        for i, (img, lab) in pbar:
            n_iter = n_iter + 1
            img_train = img.to(device)
            lab_train = lab.to(device)

            scores = model(img_train)

            loss = loss_fn(scores, lab_train)

            _, preds = scores.data.max(dim=1)

            num_correct += (preds == lab_train).sum()
            print(f"num_correct: {num_correct}")
            num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_correct = 0
            num_samples = 0
            writer.add_scalar('Loss/train',loss.item(),n_iter)
            writer.add_scalar('Accuracy/train',acc,n_iter)
            print(('\n' + '%10s' * 4) % ('Epoch', 'loss', 'acc', 'lr'))
            # s = ('%10s' + '%10.3g' * 3) % (f'{epoch}/{num_epochs}', loss, acc, lr_schedule.get_lr()[0])
            s = ('%10s' + '%10.3g' * 3) % (f'{epoch}/{num_epochs}', loss, acc, 0.001)
            pbar.set_description(s)
            # ------------------end batch-------------------------------------------
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 
                        'best_loss': loss.item(),
                        'optimizer': optimizer.state_dict()},
                        f'./weights/model.pt')
        
        
        val_acc, val_loss = check_val(model, loss_fn, val_loader,device)
        # update scheduler
        lr_schedule.step(val_acc)
        writer.add_scalar('Loss/test',val_loss,n_iter)
        writer.add_scalar('Accuracy/test',val_acc,n_iter)
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            print(f"best val accuracy:{best_val_acc}")
            print("--------------------")
            print(f"save model")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 
                            'best_loss': val_loss,
                            'optimizer': optimizer.state_dict()},
                           f'./weights/best_model.pt')
            print("--------------------")



def check_val(model, loss_fn, loader,device):
    print(f"checking top1")
    val_correct = 0
    val_samples = 0
    loss = 0

    model.eval()
    for t, (x, y) in enumerate(loader):
        x_val = x.to(device)
        y_val = y.to(device)

        scores = model(x_val)
        loss += loss_fn(scores, y_val).item()
        t = t + 1
        _, preds = scores.data.cpu().max(dim=1)
        val_correct += (preds == y).sum()
        val_samples += preds.size(0)
    val_acc = float(val_correct) / val_samples
    val_loss = loss / t
    print('val_loss:%.4f, Got %d / %d correct (%.4f%%)' % (val_loss, val_correct, val_samples, 100 * val_acc))
    print("-------------------------------------")
    return val_acc, val_loss

FLAGS = flags.FLAGS  

def main(unused_argv):
    start_epoch = 0
    
    writer = SummaryWriter()
    
    train_datasets = CharacterDateset('./data/train', type='train',converted=FLAGS.enable_converted)
    train_dataloader = DataLoader(train_datasets, batch_size=FLAGS.batch, shuffle=True, num_workers=4)
    val_datasets = CharacterDateset('./data/train', type='val')
    val_dataloader = DataLoader(val_datasets, batch_size=FLAGS.batch, shuffle=True, num_workers=4)

    # device = torch_utils.select_device('cpu', apex=mixed_precision, batch_size=batch_size)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cpu':
        mixed_precision = False

    net = resnet34(False).to(device)
    # net = DarkNet(num_classes=100).to(device)
    model_dict = torch.load(FLAGS.ckpt_dir,map_location=device) 
      
    optimizer = optim.Adam(params=net.parameters(), lr=FLAGS.lr, weight_decay=1e-10)

    if FLAGS.enable_pretrained:
        print(f"using pretrained model: {FLAGS.ckpt_dir}")
        net.load_state_dict(model_dict['model_state_dict']) 
        if model_dict['optimizer'] is not None:
            optimizer.load_state_dict(model_dict['optimizer'])

    
    lf = lambda x: (((1 + math.cos(
        x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine https://arxiv.org/pdf/1812.01187.pdf
    # lr_schedule = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=0.3, verbose= True, patience=4,min_lr=0.00001)
    loss_fn = nn.CrossEntropyLoss()

    train(net, loss_fn, optimizer, lr_schedule, train_dataloader, val_dataloader,writer,device,num_epochs=FLAGS.epoch) 


if __name__ == "__main__":
    app.run(main)
    
