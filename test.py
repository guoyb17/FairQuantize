import argparse
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp

from utils import model_backbone_v2 # , genScoreDataset
from dataloaders import fitzpatric17k_dataloader_score_v2, fitzpatric17k_pruning_examples, \
                        isic2019_dataloader_score_v2, isic2019_pruning_examples, \
                        celeba_dataloader_score_v2, celeba_pruning_examples
from fairness_metrics import compute_fairness_metrics

parser = argparse.ArgumentParser(description='Fairness on Validation Set and Test Set (Single Model)')
parser.add_argument('-n', '--num_classes', type=int, default=114,
                    help="number of classes; used for fitzpatrick17k")
parser.add_argument('-f', '--fair_attr', type=str, default="Male",
                    help="fairness attribute; now support: Male, Young; used for celeba")
parser.add_argument('-y', '--y_attr', type=str, default="Big_Nose",
                    help="y attribute; now support: Attractive, Big_Nose, Bags_Under_Eyes, Mouth_Slightly_Open, Big_Nose_And_Bags_Under_Eyes, Attractive_And_Mouth_Slightly_Open; used for celeba")
parser.add_argument('-d', '--dataset', type=str, default="fitzpatrick17k",
                    help="the dataset to use; now support: fitzpatrick17k, celeba, isic2019")
parser.add_argument('-m', '--model_file', type=str, required=True,
                    help="Model file path")
parser.add_argument('--csv_file_name', type=str, default=None,
                    help="CSV file position")
parser.add_argument('--image_dir', type=str, default=None,
                    help="Image files directory")
parser.add_argument('-o', '--output_file', type=str, required=True,
                    help="Output file path")
parser.add_argument('--pruned', type=int, default=0,
                    help="Whether it is a FairPrune model; default: 0 (no); 1: pre-set to zero; 2: masking layers")
parser.add_argument('--backbone', type=str, default="resnet18",
                    help="backbone model; now support: resnet18, resnet34, resnet50, resnet101, resnet152, vgg11, vgg11_bn, mobilenet_v2, mobilenet_v3_large, shufflenet_v2_x1_0, efficientnet_b0, vit_b_16")
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "fitzpatrick17k":
        if args.csv_file_name is None:
            csv_file_name = "fitzpatrick17k/fitzpatrick17k.csv"
        else:
            csv_file_name = args.csv_file_name
        if args.image_dir is None:
            image_dir = "fitzpatrick17k/dataset_images"
        else:
            image_dir = args.image_dir
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
        score_size = 799
        trainloader, _, testloader, train_df = fitzpatric17k_dataloader_score_v2(args.batch_size, args.workers, image_dir, csv_file_name, ctype)
        # if args.shape_type == 1:
        #     trainloader, _, testloader, train_df = fitzpatric17k_dataloader_score_v3(args.batch_size, args.workers, image_dir, csv_file_name, ctype, device)
        # else:
        #     trainloader, _, testloader, train_df = fitzpatric17k_dataloader_score_v2(args.batch_size, args.workers, image_dir, csv_file_name, ctype)
    elif args.dataset == "celeba":
        if args.csv_file_name is None:
            csv_file_name = "img_align_celeba/list_attr_celeba_modify.txt"
        else:
            csv_file_name = args.csv_file_name
        if args.image_dir is None:
            image_dir = "img_align_celeba/img_align_celeba"
        else:
            image_dir = args.image_dir
        num_classes = 2
        ctype = "y_attr"
        f_attr = "fair_attr"
        score_size = 10130
        trainloader, _, testloader, train_df = celeba_dataloader_score_v2(args.batch_size, args.workers, image_dir, csv_file_name, args.fair_attr, args.y_attr)
    elif args.dataset == "isic2019":
        if args.csv_file_name is None:
            csv_file_name = "ISIC_2019_train/ISIC_2019_Training_Metadata.csv"
        else:
            csv_file_name = args.csv_file_name
        if args.image_dir is None:
            image_dir = "ISIC_2019_train/ISIC_2019_Training_Input"
        else:
            image_dir = args.image_dir
        num_classes = 8
        ctype = "y_attr"
        f_attr = "fair_attr"
        score_size = 1248
        trainloader, _, testloader, train_df = isic2019_dataloader_score_v2(args.batch_size, args.workers, image_dir, csv_file_name, use_test=True)
    else:
        raise NotImplementedError
    f = open(args.output_file, "w")
    # f.write("dataset,overall_accuracy,light_accuracy,dark_accuracy,diff_accuracy,overall_precision,light_precision,dark_precision,diff_precision,overall_recall,light_recall,dark_recall,diff_recall,overall_F1_score,light_F1_score,dark_F1_score,diff_F1_score,(abs_of_)EOpp0,EOpp1_abs,EOdds_abs\n")
    f.write("model,overall_accuracy,light_accuracy,dark_accuracy,diff_accuracy,overall_precision,light_precision,dark_precision,diff_precision,overall_recall,light_recall,dark_recall,diff_recall,overall_F1_score,light_F1_score,dark_F1_score,diff_F1_score,EOpp0,EOpp1,EOdds,abs_of_EOpp0,EOpp0_abs,EOpp1_abs,EOdds_abs\n")

    # define the backbone model
    backbone = model_backbone_v2(num_classes, args.backbone)
    # backbone = models.vgg11(num_classes=num_classes)
    net = backbone.to(device)

    loaded_ckpt = torch.load(args.model_file, map_location=device)
    print("[NOTE] Testing", args.model_file)
    if args.pruned == 3:
        if args.dataset == "fitzpatrick17k":
            example_inputs, example_labels = fitzpatric17k_pruning_examples(minimum=True)
        elif args.dataset == "isic2019":
            example_inputs, example_labels = isic2019_pruning_examples(minimum=True)
        elif args.dataset == "celeba":
            if args.fair_attr != "Male" or args.y_attr != "Big_Nose": # FIXME: hard code
                raise NotImplementedError
            example_inputs, example_labels = celeba_pruning_examples(minimum=True)
        example_inputs = torch.stack(example_inputs, dim=0).to(device)
        example_labels = torch.tensor(example_labels).to(device)
        DG = tp.DependencyGraph().build_dependency(net, example_inputs)
        DG.load_pruning_history(loaded_ckpt['pruning'])
        net.load_state_dict(loaded_ckpt['model'])
    elif args.pruned == 2:
        idx = 0
        for name, module in net.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                try:
                    loaded_ckpt[name + ".weight"] = loaded_ckpt[name + ".weight"] * loaded_ckpt[name + ".weight_mask"]
                    loaded_ckpt.pop(name + ".weight_mask")
                except:
                    pass
                try:
                    loaded_ckpt[name + ".bias"] = loaded_ckpt[name + ".bias"] * loaded_ckpt[name + ".bias_mask"]
                    loaded_ckpt.pop(name + ".bias_mask")
                except:
                    pass # TODO: how to deal with bias?
                idx += 1
        for key in list(loaded_ckpt.keys()):
            if key.endswith("_mask"):
                loaded_ckpt.pop(key) # TODO: check if it's ok to ignore batch normalization
        net.load_state_dict(loaded_ckpt)
    elif args.pruned == 1:
        # FairPrune loading
        net.load_state_dict(loaded_ckpt['model_dict'])
    else:
        try:
            net.load_state_dict(loaded_ckpt['state_dict'])
        except:
            try:
                net.load_state_dict(loaded_ckpt['model_dict'])
            except:
                try:
                    net.load_state_dict(loaded_ckpt)
                except:
                    raise NotImplementedError

    # define loss funtion & optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    # Test
    net.eval()
    label_list = []
    y_pred_list = []
    skin_color_list = []
    for i, data in enumerate(tqdm(testloader)):
        inputs, labels = data["image"].float().to(device), torch.from_numpy(np.asarray(data[ctype])).long().to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        label_list.append(labels.detach().cpu().numpy())
        y_pred_list.append(predicted.detach().cpu().numpy())
        skin_color_list.append(data[f_attr].numpy())
        # print("[DEBUG]", outputs.data)
        # print("[DEBUG]", labels)
        # print("[DEBUG]", predicted)
        # break
    label_list = np.concatenate(label_list)
    y_pred_list = np.concatenate(y_pred_list)
    skin_color_list = np.concatenate(skin_color_list)
    return_results = {
        'skin_color/overall_acc': metrics.accuracy_score(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1]),
        'skin_color/light_acc': metrics.accuracy_score(label_list[skin_color_list==0], y_pred_list[skin_color_list==0]),
        'skin_color/dark_acc': metrics.accuracy_score(label_list[skin_color_list==1], y_pred_list[skin_color_list==1]),
        'skin_color/overall_precision': metrics.precision_score(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1], average='macro', zero_division=0),
        'skin_color/light_precision': metrics.precision_score(label_list[skin_color_list==0], y_pred_list[skin_color_list==0], average='macro', zero_division=0),
        'skin_color/dark_precision': metrics.precision_score(label_list[skin_color_list==1], y_pred_list[skin_color_list==1], average='macro', zero_division=0),
        'skin_color/overall_recall': metrics.recall_score(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1], average='macro', zero_division=0),
        'skin_color/light_recall': metrics.recall_score(label_list[skin_color_list==0], y_pred_list[skin_color_list==0], average='macro', zero_division=0),
        'skin_color/dark_recall': metrics.recall_score(label_list[skin_color_list==1], y_pred_list[skin_color_list==1], average='macro', zero_division=0),
        'skin_color/overall_f1_score': metrics.f1_score(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1], average='macro', zero_division=0),
        'skin_color/light_f1_score': metrics.f1_score(label_list[skin_color_list==0], y_pred_list[skin_color_list==0], average='macro', zero_division=0),
        'skin_color/dark_f1_score': metrics.f1_score(label_list[skin_color_list==1], y_pred_list[skin_color_list==1], average='macro', zero_division=0),
    }
    # get fairness metric
    fairness_metrics = compute_fairness_metrics(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1], skin_color_list[skin_color_list!=-1])
    for k, v in return_results.items():
        print(f'{k}:{v:.4f}')
    for k, v in fairness_metrics.items():
        print(f'{k}:{v:.4f}')
    f.write(args.model_file + ",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n"
    % (return_results['skin_color/overall_acc'], return_results['skin_color/light_acc'], return_results['skin_color/dark_acc'], return_results['skin_color/dark_acc'] - return_results['skin_color/light_acc'],
    return_results['skin_color/overall_precision'], return_results['skin_color/light_precision'], return_results['skin_color/dark_precision'], return_results['skin_color/dark_precision'] - return_results['skin_color/light_precision'],
    return_results['skin_color/overall_recall'], return_results['skin_color/light_recall'], return_results['skin_color/dark_recall'], return_results['skin_color/dark_recall'] - return_results['skin_color/light_recall'],
    return_results['skin_color/overall_f1_score'], return_results['skin_color/light_f1_score'], return_results['skin_color/dark_f1_score'], return_results['skin_color/dark_f1_score'] - return_results['skin_color/light_f1_score'],
    fairness_metrics['fairness/EOpp0'], fairness_metrics['fairness/EOpp1'], fairness_metrics['fairness/EOdds'],
    abs(fairness_metrics['fairness/EOpp0']), fairness_metrics['fairness/EOpp0_abs'], fairness_metrics['fairness/EOpp1_abs'], fairness_metrics['fairness/EOdds_abs']
    ))

    f.close()
