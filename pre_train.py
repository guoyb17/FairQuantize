import argparse, os
import shutil
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from dataloaders import fitzpatric17k_dataloader_score_v2, fitzpatric17k_dataloader_score_v3, \
                        celeba_dataloader_score_v2, \
                        isic2019_dataloader_score_v2, isic2019_dataloader_score_v3


parser = argparse.ArgumentParser(description='Pre-train')

parser.add_argument('-n', '--num_classes', type=int, default=114,
                    help="number of classes; used for fitzpatrick17k")
parser.add_argument('-f', '--fair_attr', type=str, default="Male",
                    help="fairness attribute; now support: Male, Young; used for celeba")
parser.add_argument('-y', '--y_attr', type=str, default="Big_Nose",
                    help="y attribute; now support: Attractive, Big_Nose, Bags_Under_Eyes, Mouth_Slightly_Open, Big_Nose_And_Bags_Under_Eyes, Attractive_And_Mouth_Slightly_Open; used for celeba")
parser.add_argument('-d', '--dataset', type=str, default="fitzpatrick17k",
                    help="the dataset to use; now support: fitzpatrick17k, celeba, isic2019")
parser.add_argument('-j', '--workers', default=15, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of epochs')
# parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
parser.add_argument('--backbone', type=str, default="resnet18",
                    help="backbone model; now support: resnet18, resnet34, resnet50, resnet101, resnet152, vgg11, vgg11_bn, mobilenet_v2, mobilenet_v3_large, shufflenet_v2_x1_0, efficientnet_b0, vit_b_16")
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--optimizer', type=str, default="sgd",
                    help='Optimizer. Now support: sgd, adam, momentum, rms')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 0.0001)',
                    dest='weight_decay')
parser.add_argument('--log_dir', type=str, required=True,
                    help='directory to log the checkpoint and training log to')
parser.add_argument('--csv_file_name', type=str, default=None,
                    help="CSV file position")
parser.add_argument('--image_dir', type=str, default=None,
                    help="Image files directory")
parser.add_argument('--ckpt_limit', default=0, type=int, metavar='N',
                    help='max number of checkpoints to save; default now is 0; must be non-negative, or it would be considered as 0')
parser.add_argument('--step_size', default=20, type=int, metavar='N',
                    help='after epochs of step size, learning rate decay')
parser.add_argument('--gamma', default=0.57, type=float, metavar='N', # 0.57^4 is about 0.1
                    help='learning rate decay by gamma*')
parser.add_argument('--log_file', type=str, required=True,
                    help="Accuracy and loss on training set and validation set during training; format: epoch, train_acc, train_loss, val_acc, val_loss")
parser.add_argument("--pre_trained", type=int, default=0,
                    help="whether to use pre-trained model; 0: False (default), >= 1: True (only 1 uses pre-trained model); -1: use full VGG11 pre-trained model")
parser.add_argument('-m', '--model_file', type=str, default=None,
                    help="Partly trained model file path")
parser.add_argument('--pre_load', type=int, default=0,
                    help="If 1, pre-load datasets into GPU memory")

def save_checkpoint(state, is_best_acc, is_best_loss, filename, to_delete=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    if is_best_acc:
        print("[INFO] New best accuracy:", filename)
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename),'best_acc.pth.tar'))
    if is_best_loss:
        print("[INFO] New best loss:", filename)
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename),'best_loss.pth.tar'))
    if to_delete is not None:
        os.remove(to_delete)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.ckpt_limit < 0:
        ckpt_limit = 0
    else:
        ckpt_limit = args.ckpt_limit
    # check gpu
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    if args.dataset == "fitzpatrick17k":
        if args.csv_file_name is None:
            csv_file_name = "fitzpatrick17k/fitzpatrick17k.csv"
        else:
            csv_file_name = args.csv_file_name
        if args.image_dir is None:
            image_dir = "fitzpatrick17k/dataset_images"
        else:
            image_dir = args.image_dir
        # score_size = 799
        num_classes = args.num_classes
        if num_classes == 3:
            ctype = "high"
        elif num_classes == 9:
            ctype = "mid"
        elif num_classes == 114:
            ctype = "low"
        else:
            raise NotImplementedError
        f_attr = "skin_color_binary"
        if args.pre_load == 1:
            trainloader, valloader, _, _ = fitzpatric17k_dataloader_score_v3(args.batch_size, args.workers, image_dir, csv_file_name, ctype, device, True, True, False)
        else:
            trainloader, valloader, _, _ = fitzpatric17k_dataloader_score_v2(args.batch_size, args.workers, image_dir, csv_file_name, ctype)
    elif args.dataset == "celeba":
        if args.csv_file_name is None:
            csv_file_name = "img_align_celeba/list_attr_celeba_modify.txt"
        else:
            csv_file_name = args.csv_file_name
        if args.image_dir is None:
            image_dir = "img_align_celeba/img_align_celeba"
        else:
            image_dir = args.image_dir
        # score_size = 10130
        num_classes = 2
        ctype = "y_attr"
        f_attr = "fair_attr"
        if args.pre_load == 1:
            raise NotImplementedError
        else:
            trainloader, valloader, _, _ = celeba_dataloader_score_v2(args.batch_size, args.workers, image_dir, csv_file_name, args.fair_attr, args.y_attr)
    elif args.dataset == "isic2019":
        if args.csv_file_name is None:
            csv_file_name = "ISIC_2019_train/ISIC_2019_Training_Metadata.csv"
        else:
            csv_file_name = args.csv_file_name
        if args.image_dir is None:
            image_dir = "ISIC_2019_train/ISIC_2019_Training_Input"
        else:
            image_dir = args.image_dir
        # score_size = 1248
        num_classes = 8
        ctype = "y_attr"
        f_attr = "fair_attr"
        if args.pre_load == 1:
            trainloader, valloader, _, _ = isic2019_dataloader_score_v3(args.batch_size, args.workers, image_dir, csv_file_name, device, True, True, False)
        else:
            trainloader, valloader, _, _ = isic2019_dataloader_score_v2(args.batch_size, args.workers, image_dir, csv_file_name, use_val=True)
    else:
        raise NotImplementedError

    # define the backbone model
    if args.model_file is not None and args.pre_trained == -1:
        if args.backbone == "resnet18":
            backbone = models.resnet18(num_classes=num_classes)
        elif args.backbone == "resnet34":
            backbone = models.resnet34(num_classes=num_classes)
        elif args.backbone == "resnet50":
            backbone = models.resnet50(num_classes=num_classes)
        elif args.backbone == "resnet101":
            backbone = models.resnet101(num_classes=num_classes)
        elif args.backbone == "resnet152":
            backbone = models.resnet152(num_classes=num_classes)
        elif args.backbone == "vgg11":
            backbone = models.vgg11(num_classes=num_classes)
        elif args.backbone == "vgg11_bn":
            backbone = models.vgg11_bn(num_classes=num_classes)
        elif args.backbone == "mobilenet_v2":
            backbone = models.mobilenet_v2(num_classes=num_classes)
        elif args.backbone == "mobilenet_v3_large":
            backbone = models.mobilenet_v3_large(num_classes=num_classes)
        elif args.backbone == "shufflenet_v2_x1_0":
            backbone = models.shufflenet_v2_x1_0(num_classes=num_classes)
        elif args.backbone == "efficientnet_b0":
            backbone = models.efficientnet_b0(num_classes=num_classes)
        elif args.backbone == "vit_b_16":
            backbone = models.vit_b_16(num_classes=num_classes, image_size=(224 // 2)) # TODO: change image_size
        else:
            raise NotImplementedError
        loaded_ckpt = torch.load(args.model_file, map_location=device)
        try:
            backbone.load_state_dict(loaded_ckpt['state_dict'])
        except:
            try:
                backbone.load_state_dict(loaded_ckpt['model_dict']) # For Dewen's FairPrune models
            except:
                try:
                    backbone.load_state_dict(loaded_ckpt)
                except:
                    raise NotImplementedError
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.avgpool = nn.AvgPool2d((1, 1))
        backbone.classifier[0] = nn.Linear(512 * 3 * 3, 4096)
        backbone.classifier[-1] = nn.Linear(4096, num_classes)
    elif args.pre_trained == 1:
        if args.backbone == "resnet18":
            backbone = models.resnet18(pretrained=True)
        elif args.backbone == "resnet34":
            backbone = models.resnet34(pretrained=True)
        elif args.backbone == "resnet50":
            backbone = models.resnet50(pretrained=True)
        elif args.backbone == "resnet101":
            backbone = models.resnet101(pretrained=True)
        elif args.backbone == "resnet152":
            backbone = models.resnet152(pretrained=True)
        else:
            raise NotImplementedError
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    else:
        if args.backbone == "resnet18":
            backbone = models.resnet18(num_classes=num_classes)
        elif args.backbone == "resnet34":
            backbone = models.resnet34(num_classes=num_classes)
        elif args.backbone == "resnet50":
            backbone = models.resnet50(num_classes=num_classes)
        elif args.backbone == "resnet101":
            backbone = models.resnet101(num_classes=num_classes)
        elif args.backbone == "resnet152":
            backbone = models.resnet152(num_classes=num_classes)
        elif args.backbone == "vgg11":
            backbone = models.vgg11(pretrained=True) # TODO: change to False
            backbone.avgpool = nn.AvgPool2d((1, 1))
            backbone.classifier[0] = nn.Linear(512 * 3 * 3, 4096)
            backbone.classifier[-1] = nn.Linear(4096, num_classes)
        elif args.backbone == "vgg11_bn":
            backbone = models.vgg11_bn(num_classes=num_classes)
        elif args.backbone == "mobilenet_v2":
            backbone = models.mobilenet_v2(num_classes=num_classes)
        elif args.backbone == "mobilenet_v3_large":
            backbone = models.mobilenet_v3_large(num_classes=num_classes)
        elif args.backbone == "shufflenet_v2_x1_0":
            backbone = models.shufflenet_v2_x1_0(num_classes=num_classes)
        elif args.backbone == "efficientnet_b0":
            backbone = models.efficientnet_b0(num_classes=num_classes)
        elif args.backbone == "vit_b_16":
            backbone = models.vit_b_16(num_classes=num_classes, image_size=(224 // 2)) # TODO: change image_size
        else:
            raise NotImplementedError
    net = backbone.to(device)
    if args.model_file is not None and args.pre_trained >= 0:
        loaded_ckpt = torch.load(args.model_file, map_location=device)
        try:
            net.load_state_dict(loaded_ckpt['state_dict'])
        except:
            try:
                net.load_state_dict(loaded_ckpt['model_dict']) # For Dewen's FairPrune models
            except:
                try:
                    net.load_state_dict(loaded_ckpt)
                except:
                    raise NotImplementedError

    # define loss funtion & optimizer
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "momentum":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    elif args.optimizer == "rms":
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    f = open(args.log_file, "w")
    # train
    best_val_acc = 0.0
    best_val_loss = 100.0
    for epoch in range(args.epochs): # range(args.start_epoch, args.epochs)
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(tqdm(trainloader)):
            # prepare dataset
            # length = len(trainloader)
            inputs, labels = data["image"].float().to(device), torch.from_numpy(np.asarray(data[ctype])).long().to(device)
            optimizer.zero_grad()
            
            # forward & backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print ac & loss in each batch
            sum_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            # print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%%' 
            #     % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
        train_loss = sum_loss / total
        train_accuracy = 100. * correct / total
        print('[epoch:%d] Loss: %.03f | Acc: %.3f%%' % (epoch + 1, train_loss, train_accuracy))
        f.write('%03d, %.03f, %.03f, ' % (epoch + 1, train_accuracy, train_loss))
            
        scheduler.step()
        # get the ac with testdataset in each epoch
        # print('Waiting Test...')
        with torch.no_grad():
            correct = 0
            total = 0
            val_loss = 0.0
            net.eval()
            for i, data in enumerate(tqdm(valloader)):
                inputs, labels = data["image"].float().to(device), torch.from_numpy(np.asarray(data[ctype])).long().to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            val_loss /= total
            val_acc = 100. * correct / total
            print('Val\'s ac is: %.3f%%, loss is: %.3f' % (val_acc, val_loss))
            f.write('%.03f, %.03f\n' % (val_acc, val_loss))

        save_checkpoint({
                'state_dict': net.state_dict(),
                # 'optimizer': optimizer.state_dict(),
            }, val_acc > best_val_acc, val_loss < best_val_loss, filename=os.path.join(args.log_dir, '{}.pth.tar'.format(epoch)), # val_acc > best_val_acc to replace False
            to_delete=None if epoch < ckpt_limit else os.path.join(args.log_dir, '{}.pth.tar'.format(epoch - ckpt_limit)))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    print('Train has finished, total epoch is %d' % args.epochs)
    f.close()
