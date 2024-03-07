import os
from time import sleep
from tqdm import tqdm
import numpy as np
import torch
import logging.config
import shutil
import pandas as pd
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.layouts import column
import torchvision.models
import torch.nn as nn

from fairness_metrics import compute_fairness_metrics
from dataloaders import Fitzpatrick_17k_Augmentations, Fitzpatrick17k, Fitzpatrick17kV2, \
                        CelebA_Augmentations, CelebA, \
                        ISIC2019_Augmentations, ISIC2019, ISIC2019V2


def setup_logging(log_file='log.txt'):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


class ResultsLog(object):

    def __init__(self, path='results.csv', plot_path=None):
        self.path = path
        self.plot_path = plot_path or (self.path + '.html')
        self.figures = []
        self.results = None

    def add(self, **kwargs):
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if self.results is None:
            self.results = df
        else:
            self.results = self.results.append(df, ignore_index=True)

    def save(self, title='Training Results'):
        if len(self.figures) > 0:
            if os.path.isfile(self.plot_path):
                os.remove(self.plot_path)
            output_file(self.plot_path, title=title)
            plot = column(*self.figures)
            save(plot)
            self.figures = []
        self.results.to_csv(self.path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.path
        if os.path.isfile(path):
            self.results.read_csv(path)

    def show(self):
        if len(self.figures) > 0:
            plot = column(*self.figures)
            show(plot)

    #def plot(self, *kargs, **kwargs):
    #    line = Line(data=self.results, *kargs, **kwargs)
    #    self.figures.append(line)

    def image(self, *kargs, **kwargs):
        fig = figure()
        fig.image(*kargs, **kwargs)
        self.figures.append(fig)


def save_checkpoint(state, is_best, path='.', filename='checkpoint.pth.tar', save_all=False):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))
    if save_all:
        shutil.copyfile(filename, os.path.join(
            path, 'checkpoint_epoch_%s.pth.tar' % state['epoch']))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

__optimizers = {
    'SGD': torch.optim.SGD,
    'ASGD': torch.optim.ASGD,
    'Adam': torch.optim.Adam,
    'Adamax': torch.optim.Adamax,
    'Adagrad': torch.optim.Adagrad,
    'Adadelta': torch.optim.Adadelta,
    'Rprop': torch.optim.Rprop,
    'RMSprop': torch.optim.RMSprop
}


def adjust_optimizer(optimizer, epoch, config):
    """Reconfigures the optimizer according to epoch and config dict"""
    def modify_optimizer(optimizer, setting):
        if 'optimizer' in setting:
            optimizer = __optimizers[setting['optimizer']](
                optimizer.param_groups)
            logging.debug('OPTIMIZER - setting method = %s' %
                          setting['optimizer'])
        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    logging.debug('OPTIMIZER - setting %s = %s' %
                                  (key, setting[key]))
                    param_group[key] = setting[key]
        return optimizer

    if callable(config):
        optimizer = modify_optimizer(optimizer, config(epoch))
    else:
        for e in range(epoch + 1):  # run over all epochs - sticky setting
            if e in config:
                optimizer = modify_optimizer(optimizer, config[e])

    return optimizer


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

    # kernel_img = model.features[0][0].kernel.data.clone()
    # kernel_img.add_(-kernel_img.min())
    # kernel_img.mul_(255 / kernel_img.max())
    # save_image(kernel_img, 'kernel%s.jpg' % epoch)

def model_backbone_v2(num_classes, backbone):
    if backbone == None:
        return None
    elif backbone == "mobilenet_v2":
        model = torchvision.models.mobilenet_v2()
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif backbone == "mobilenet_v3_small":
        model = torchvision.models.mobilenet_v3_small()
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone == "mobilenet_v3_large":
        model = torchvision.models.mobilenet_v3_large()
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone == "shufflenet_v2_x0_5":
        model = torchvision.models.shufflenet_v2_x0_5()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "shufflenet_v2_x1_0":
        model = torchvision.models.shufflenet_v2_x1_0()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "shufflenet_v2_x1_5":
        model = torchvision.models.shufflenet_v2_x1_5()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "shufflenet_v2_x2_0":
        model = torchvision.models.shufflenet_v2_x2_0()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "resnet18":
        model = torchvision.models.resnet18()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "resnet34":
        model = torchvision.models.resnet34()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "resnet50":
        model = torchvision.models.resnet50()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "resnet101":
        model = torchvision.models.resnet101()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "resnet152":
        model = torchvision.models.resnet152()
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "vgg11":
        model = torchvision.models.vgg11()
        model.avgpool = nn.AvgPool2d((1, 1))
        model.classifier[0] = nn.Linear(512 * 3 * 3, 4096)
        model.classifier[-1] = nn.Linear(4096, num_classes)
    elif backbone == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0()
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone == "vit_b_16":
        model = torchvision.models.vit_b_16(num_classes=num_classes, image_size=(224 // 2)) # TODO: change image_size
    else:
        raise NotImplementedError
    return model

def genScoreDataset(args, train_df, image_dir, device="cpu", enable_preload=True):
    if args.dataset == "fitzpatrick17k":
        light_score_df = train_df
        dark_score_df = train_df
        light_score_df = light_score_df.drop(light_score_df[(light_score_df["fitzpatrick"] < 1) | (light_score_df["fitzpatrick"] > 3)].index)
        dark_score_df = dark_score_df.drop(dark_score_df[(dark_score_df["fitzpatrick"] < 4) | (dark_score_df["fitzpatrick"] > 6)].index)
        score_size = min(light_score_df.shape[0], dark_score_df.shape[0])
        light_score_df = light_score_df.sample(n=score_size)
        dark_score_df = dark_score_df.sample(n=score_size)
        image_size = 256 // 2
        crop_size = 224 // 2
        test_transform = Fitzpatrick_17k_Augmentations(is_training=False, image_size=image_size, input_size=crop_size).transforms
        if args.pre_load == 1 and enable_preload:
            light_score_dataset = Fitzpatrick17kV2(df=light_score_df, root_dir=image_dir, transform=test_transform)
            dark_score_dataset = Fitzpatrick17kV2(df=dark_score_df, root_dir=image_dir, transform=test_transform)
            light_score_dataset.to(device)
            dark_score_dataset.to(device)
        else:
            light_score_dataset = Fitzpatrick17k(df=light_score_df, root_dir=image_dir, transform=test_transform)
            dark_score_dataset = Fitzpatrick17k(df=dark_score_df, root_dir=image_dir, transform=test_transform)
    elif args.dataset == "celeba":
        light_score_df = train_df
        dark_score_df = train_df
        light_score_df = light_score_df.drop(light_score_df[light_score_df[args.fair_attr] != 0].index)
        dark_score_df = dark_score_df.drop(dark_score_df[dark_score_df[args.fair_attr] != 1].index)
        score_size = min(light_score_df.shape[0], dark_score_df.shape[0])
        light_score_df = light_score_df.sample(n=score_size)
        dark_score_df = dark_score_df.sample(n=score_size)
        image_size = 256 // 2
        crop_size = 224 // 2
        test_transform = CelebA_Augmentations(is_training=False, image_size=image_size, input_size=crop_size).transforms
        if args.pre_load == 1 and enable_preload:
            raise NotImplementedError
        else:
            light_score_dataset = CelebA(df=light_score_df, fair_attr=args.fair_attr, y_attr=args.y_attr, root_dir=image_dir, transform=test_transform)
            dark_score_dataset = CelebA(df=dark_score_df, fair_attr=args.fair_attr, y_attr=args.y_attr, root_dir=image_dir, transform=test_transform)
    elif args.dataset == "isic2019":
        light_score_df = train_df
        dark_score_df = train_df
        light_score_df = light_score_df.drop(light_score_df[light_score_df["sex_id"] != 0].index)
        dark_score_df = dark_score_df.drop(dark_score_df[dark_score_df["sex_id"] != 1].index)
        score_size = min(light_score_df.shape[0], dark_score_df.shape[0])
        light_score_df = light_score_df.sample(n=score_size)
        dark_score_df = dark_score_df.sample(n=score_size)
        image_size = 256 // 2
        crop_size = 224 // 2
        test_transform = ISIC2019_Augmentations(is_training=False, image_size=image_size, input_size=crop_size).transforms
        if args.pre_load == 1 and enable_preload:
            light_score_dataset = ISIC2019V2(df=light_score_df, root_dir=image_dir, transform=test_transform)
            dark_score_dataset = ISIC2019V2(df=dark_score_df, root_dir=image_dir, transform=test_transform)
            light_score_dataset.to(device)
            dark_score_dataset.to(device)
        else:
            light_score_dataset = ISIC2019(df=light_score_df, root_dir=image_dir, transform=test_transform)
            dark_score_dataset = ISIC2019(df=dark_score_df, root_dir=image_dir, transform=test_transform)
    return light_score_dataset, dark_score_dataset

def validate(net, valloader, criterion, device, ctype, f_attr):
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        val_loss = 0.0
        label_list = []
        y_pred_list = []
        skin_color_list = []
        for _, data in enumerate(tqdm(valloader)):
            inputs, labels = data["image"].float().to(device), torch.from_numpy(np.asarray(data[ctype])).long().to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * labels.size(0)
            label_list.append(labels.detach().cpu().numpy())
            y_pred_list.append(predicted.detach().cpu().numpy())
            skin_color_list.append(data[f_attr].numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum()
        label_list = np.concatenate(label_list)
        y_pred_list = np.concatenate(y_pred_list)
        skin_color_list = np.concatenate(skin_color_list)
        # return_results = {
        #     'skin_color/overall_acc': metrics.accuracy_score(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1]),
        #     'skin_color/light_acc': metrics.accuracy_score(label_list[skin_color_list==0], y_pred_list[skin_color_list==0]),
        #     'skin_color/dark_acc': metrics.accuracy_score(label_list[skin_color_list==1], y_pred_list[skin_color_list==1]),
        #     'skin_color/overall_precision': metrics.precision_score(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1], average='macro', zero_division=0),
        #     'skin_color/light_precision': metrics.precision_score(label_list[skin_color_list==0], y_pred_list[skin_color_list==0], average='macro', zero_division=0),
        #     'skin_color/dark_precision': metrics.precision_score(label_list[skin_color_list==1], y_pred_list[skin_color_list==1], average='macro', zero_division=0),
        #     'skin_color/overall_recall': metrics.recall_score(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1], average='macro', zero_division=0),
        #     'skin_color/light_recall': metrics.recall_score(label_list[skin_color_list==0], y_pred_list[skin_color_list==0], average='macro', zero_division=0),
        #     'skin_color/dark_recall': metrics.recall_score(label_list[skin_color_list==1], y_pred_list[skin_color_list==1], average='macro', zero_division=0),
        #     'skin_color/overall_f1_score': metrics.f1_score(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1], average='macro', zero_division=0),
        #     'skin_color/light_f1_score': metrics.f1_score(label_list[skin_color_list==0], y_pred_list[skin_color_list==0], average='macro', zero_division=0),
        #     'skin_color/dark_f1_score': metrics.f1_score(label_list[skin_color_list==1], y_pred_list[skin_color_list==1], average='macro', zero_division=0),
        # }
        # get fairness metric
        # fairness_metrics = compute_fairness_metrics(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1], skin_color_list[skin_color_list!=-1])
        # for k, v in return_results.items():
        #     print(f'{k}:{v:.4f}')
        # for k, v in fairness_metrics.items():
        #     print(f'{k}:{v:.4f}')
        val_loss /= total
        val_acc = 100. * correct / total
        fairness_metrics = compute_fairness_metrics(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1], skin_color_list[skin_color_list!=-1])
        val_eopp0 = abs(fairness_metrics['fairness/EOpp0'])
        val_eopp0_abs = fairness_metrics['fairness/EOpp0_abs']
        val_eopp1_abs = fairness_metrics['fairness/EOpp1_abs']
        val_eodds_abs = fairness_metrics['fairness/EOdds_abs']
        print('Val\'s ac is: %.02f%%, loss is: %.04f, eopp0 is: %.04f, eopp0_abs is: %.04f, eopp1_abs is: %.04f, eodds_abs is: %.04f' % (val_acc, val_loss, val_eopp0, val_eopp0_abs, val_eopp1_abs, val_eodds_abs))
    return val_acc, val_loss, val_eopp0, val_eopp0_abs, val_eopp1_abs, val_eodds_abs
