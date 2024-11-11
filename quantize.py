import argparse, os # , sys
from tqdm import tqdm
from tqdm.contrib import tzip
# from time import sleep

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Hessian by backPack
from backpack import backpack, extend
from backpack.extensions import DiagGGNExact, DiagHessian

from utils import model_backbone_v2, genScoreDataset, validate
from dataloaders import fitzpatric17k_dataloader_score_v2, fitzpatric17k_dataloader_score_v3, \
                        celeba_dataloader_score_v2, \
                        isic2019_dataloader_score_v2, isic2019_dataloader_score_v3


parser = argparse.ArgumentParser(description='Score with weight-wise quantization')

parser.add_argument('-n', '--num_classes', type=int, default=114,
                    help="number of classes; used for fitzpatrick17k")
parser.add_argument('-f', '--fair_attr', type=str, default="Male",
                    help="fairness attribute; now support: Male, Young; used for celeba")
parser.add_argument('-y', '--y_attr', type=str, default="Big_Nose",
                    help="y attribute; now support: Attractive, Big_Nose, Bags_Under_Eyes, Mouth_Slightly_Open, Big_Nose_And_Bags_Under_Eyes, Attractive_And_Mouth_Slightly_Open; used for celeba")
parser.add_argument('-d', '--dataset', type=str, default="fitzpatrick17k",
                    help="the dataset to use; now support: fitzpatrick17k, celeba, isic2019")
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of epochs for re-training in each cycle')
parser.add_argument('--model_file', type=str, required=True,
                    help="pre-trained model file path")
parser.add_argument('--backbone', type=str, default="resnet18",
                    help="backbone model; now support: resnet18, vgg11, vgg11_bn")
parser.add_argument('-b', '--batch_size', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.005)',
                    dest='weight_decay')
parser.add_argument('--log_dir', type=str, required=True,
                    help='directory to log the checkpoint and training log to')
parser.add_argument('--optimizer', type=str, default="sgd",
                    help='Optimizer. Now support: sgd, adam, momentum, rms')
parser.add_argument('--csv_file_name', type=str, default=None,
                    help="CSV file position")
parser.add_argument('--image_dir', type=str, default=None,
                    help="Image files directory")
parser.add_argument('--pre_load', type=int, default=0,
                    help="If 1, pre-load datasets into GPU memory")
parser.add_argument('--light_weight', required=True, type=int,
                    help='weight for hessian list on fair_attr=0 data; original: 9')
parser.add_argument('--dark_weight', required=True, type=int,
                    help='weight for hessian list on fair_attr=1 data; original: -5')
parser.add_argument('--abs_hessian', type=int, default=0,
                    help='default: 0; if 1, use abs(hessian) instead of hessian; if 2, weighted hessian takes abs value')
parser.add_argument('--diag_hessian', type=int, default=0,
                    help='if 1, use DiagHessian instead of DiagGGNExact')
parser.add_argument('--alpha', default=0.0, type=float,
                    help='alpha parameter; if you do not understand, do not change')
parser.add_argument('--step_size', default=4, type=int, metavar='N',
                    help='after epochs of step size, learning rate decay')
parser.add_argument('--gamma', default=0.57, type=float, metavar='N', # 0.57^4 is about 0.1
                    help='learning rate decay by gamma*')
parser.add_argument('--debug', type=int, default=0,
                    help='if 1, skip scoring')

if __name__ == "__main__":
    args = parser.parse_args()
    # check gpu
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    os.makedirs(args.log_dir, exist_ok=True)

    # score_size = 0
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
            _, valloader, _, train_df = fitzpatric17k_dataloader_score_v3(args.batch_size, args.workers, image_dir, csv_file_name, ctype, device, False, True, False)
        else:
            _, valloader, _, train_df = fitzpatric17k_dataloader_score_v2(args.batch_size, args.workers, image_dir, csv_file_name, ctype)
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
            _, valloader, _, train_df = celeba_dataloader_score_v2(args.batch_size, args.workers, image_dir, csv_file_name, args.fair_attr, args.y_attr)
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
            _, valloader, _, train_df = isic2019_dataloader_score_v3(args.batch_size, args.workers, image_dir, csv_file_name, device, False, True, False)
        else:
            _, valloader, _, train_df = isic2019_dataloader_score_v2(args.batch_size, args.workers, image_dir, csv_file_name, use_val=True)
    else:
        raise NotImplementedError

    # define the backbone model
    net = model_backbone_v2(num_classes, args.backbone).to(device)

    # define loss funtion & optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # quantize_rate = [0.1 for _ in range(10)]
    quantize_rate = [0.05 for _ in range(20)]
    # quantize_rate = [0.2 for _ in range(5)]
    # quantize_rate = [0.01 for _ in range(100)]
    n_quantize = len(quantize_rate)
    # get the related layers/weights which can be quantized
    quantize_layer_list = []
    weight_num = 0
    # bias_num = 0
    param_num = 0
    to_quantize_weight = 0
    to_quantize_bias = 0
    # print('Initial: get the related layers/weights which can be quantized')

    # load pre-trained model
    quantized_weights = []
    quantized_weight_index = []
    unquantized_weight_index = []
    # quantized_bias = []
    # quantized_bias_index = []
    loaded_ckpt = torch.load(args.model_file, map_location=device)
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
    for name, param in net.named_parameters():
        if 'weight' in name:
            quantize_layer_list.append(name)
            weight_num += param.data.numel()
            quantized_weight_index.append(torch.zeros(param.shape, device=device))
            unquantized_weight_index.append(torch.ones(param.shape, device=device))
            quantized_weights.append(torch.zeros(param.shape, device=device))
            # bias_num += module.bias.data.numel()
            # quantized_bias_index.append(torch.zeros(module.bias.shape, device=device))
            # quantized_bias.append(torch.zeros(module.bias.shape, device=device))
        # use elif to freeze batchnorm and pooling layers
        # elif isinstance(module, (nn.BatchNorm2d, nn.AdaptiveAvgPool2d)):
        #     for param in module.parameters():
        #         param.requires_grad = False

    to_quantize_weight += weight_num
    # to_quantize_bias += weight_num
    # to_quantize = to_quantize_weight + to_quantize_bias
    to_quantize = to_quantize_weight
    left_to_quantize = to_quantize
    total_rate = 0.0
    # param_num = weight_num + bias_num
    param_num = weight_num

    quantize_target = 0.0
    for iter in range(n_quantize):  # args.n_quantize
        quantize_target += quantize_rate[iter]
        print('########## Ovarall Quantization n: %d' % iter)
        print('Quantization target: %.3f' % quantize_target)

        #### step_1:  score, get the score of each weight
        print('step_1: score')
        net.eval()
        snet = model_backbone_v2(num_classes, args.backbone).to(device)
        snet.load_state_dict(net.state_dict())
        snet.eval()
        snet = extend(snet, use_converter=True)
        scriterion = extend(criterion)

        weight_hessian_list_light = None
        weight_hessian_list_dark = None
        # bias_hessian_list_light = None
        # bias_hessian_list_dark = None
        weight_change = []
        for name, param in snet.named_parameters():
            if 'weight' in name:
                weight_change.append(2 ** torch.round(torch.log2(torch.abs(param))) * torch.sign(param) - param)

        # Generate lightloader and darkloader
        light_score_dataset, dark_score_dataset = genScoreDataset(args, train_df, image_dir, device, False)
        lightloader = torch.utils.data.DataLoader(light_score_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        darkloader = torch.utils.data.DataLoader(dark_score_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

        print("Scoring on light images...")
        with backpack(DiagHessian() if args.diag_hessian == 1 else DiagGGNExact()):  # DiagHessian is not supported for ResNet models
            for i, data in enumerate(tqdm(lightloader)):
                inputs, labels = data["image"].float().to(device), torch.from_numpy(np.asarray(data[ctype])).long().to(device)
                snet.zero_grad()
                outputs = snet(inputs)
                _, predicted = torch.max(outputs.data, 1)
                loss = scriterion(outputs, labels)
                loss.backward()
                local_weight_hessian_list = []
                # local_bias_hessian_list = []
                for name, param in snet.named_parameters():
                    if 'weight' in name:
                        local_weight_hessian_list.append(param.diag_h if args.diag_hessian == 1 else param.diag_ggn_exact)

                # Update the hessian list for weights
                if weight_hessian_list_light is None:
                    weight_hessian_list_light = local_weight_hessian_list
                else:
                    for i in range(len(weight_hessian_list_light)):
                        weight_hessian_list_light[i] = weight_hessian_list_light[i] + local_weight_hessian_list[i]
                # Update the hessian list for biases
                # if bias_hessian_list_light is None:
                #     bias_hessian_list_light = local_bias_hessian_list
                # else:
                #     for i in range(len(bias_hessian_list_light)):
                #         bias_hessian_list_light[i] = bias_hessian_list_light[i] + local_bias_hessian_list[i]
                if args.debug == 1:
                    break
            # for hessian in weight_hessian_list_light:
            #     hessian /= (i + 1)

        print("Scoring on dark images...")
        with backpack(DiagHessian() if args.diag_hessian == 1 else DiagGGNExact()):  # DiagHessian is not supported for ResNet models
            for i, data in enumerate(tqdm(darkloader)):
                inputs, labels = data["image"].float().to(device), torch.from_numpy(np.asarray(data[ctype])).long().to(device)
                snet.zero_grad()
                outputs = snet(inputs)
                _, predicted = torch.max(outputs.data, 1)
                loss = scriterion(outputs, labels)
                loss.backward()
                local_weight_hessian_list = []
                # local_bias_hessian_list = []
                for name, param in snet.named_parameters():
                    if 'weight' in name:
                        local_weight_hessian_list.append(param.diag_h if args.diag_hessian == 1 else param.diag_ggn_exact)

                # Similar to the light images
                if weight_hessian_list_dark is None:
                    weight_hessian_list_dark = local_weight_hessian_list
                else:
                    for i in range(len(weight_hessian_list_dark)):
                        weight_hessian_list_dark[i] = weight_hessian_list_dark[i] + local_weight_hessian_list[i]
                # if bias_hessian_list_dark is None:
                #     bias_hessian_list_dark = local_bias_hessian_list
                # else:
                #     for i in range(len(bias_hessian_list_dark)):
                #         bias_hessian_list_dark[i] = bias_hessian_list_dark[i] + local_bias_hessian_list[i]
                if args.debug == 1:
                    break
            # for hessian in weight_hessian_list_dark:
            #     hessian /= (i + 1)

        #### step_2:  quantization
        print('step_2: quantization')

        # tmp_unquant_param_idx = {}
        local_quantize_weight_idx_dict = {}
        # current_idx_bias_start = 0
        for i in range(len(quantized_weight_index)):
            if args.abs_hessian == 1:
                tmp_score = torch.abs(weight_hessian_list_light[i]) * args.light_weight + torch.abs(weight_hessian_list_dark[i]) * args.dark_weight
            elif args.abs_hessian == 2:
                tmp_score = torch.abs(weight_hessian_list_light[i] * args.light_weight + weight_hessian_list_dark[i] * args.dark_weight)
            else:
                tmp_score = weight_hessian_list_light[i] * args.light_weight + weight_hessian_list_dark[i] * args.dark_weight
            tmp_tensor = unquantized_weight_index[i] * tmp_score * weight_change[i] ** 2 + quantized_weight_index[i] * (-1.0e38) # sys.float_info.max is float (i.e., float64), and will cause overflow in torch which uese float32
            # tmp_unquant_param_idx[id(quantize_layer_list[i])] = tmp_tensor
            local_scores = torch.flatten(tmp_tensor)
            local_score_rank = nn.Unflatten(0, tmp_tensor.shape)(torch.argsort(local_scores))
            kth = int(round(quantize_target * local_scores.numel()))
            if kth < 1:
                kth = 1
                raise NotImplementedError
            local_quantize_weight_idx_dict[quantize_layer_list[i]] = torch.where(local_score_rank <= kth, torch.tensor([1.]).to(device), torch.tensor([0.]).to(device))

        current_idx_weight = 0
        # current_idx_bias = current_idx_bias_start
        # quantized_cnt = 0 # debug
        # unquantized_cnt = 0 # debug
        idx = 0
        for name, param in snet.named_parameters():
            if 'weight' in name:
                # power of 2
                tmp_quantized_weight = torch.round(torch.log2(torch.abs(param)))
                # tmp_valid_quantized_weight = 2 ** tmp_quantized_weight * local_quantize_weight_idx_dict[name] * torch.sign(module.weight)
                quantized_weights[idx] = 2 ** tmp_quantized_weight * torch.sign(param)
                # tmp_quantized_bias = torch.round(torch.log2(torch.abs(module.bias)))
                # tmp_valid_quantized_bias = 2 ** tmp_quantized_bias * local_quantize_bias_idx * torch.sign(module.bias)

                # quantized_weights[idx] = quantized_weights[idx] * quantized_weight_index[idx] + tmp_valid_quantized_weight * unquantized_weight_index[idx] # update quantization results; only results matching updated quantized_weight_index[idx] is legal
                # quantized_weights[idx] = quantized_weights[idx] + tmp_valid_quantized_weight
                # print("[DEBUG]", name, torch.sum(quantized_weight_index[idx]), torch.sum(local_quantize_weight_idx_dict[name]))
                quantized_weight_index[idx] = ((quantized_weight_index[idx] + local_quantize_weight_idx_dict[name] * 1) > 0) * 1
                unquantized_weight_index[idx] = torch.ones(quantized_weight_index[idx].shape, device=device) - quantized_weight_index[idx]

                # quantized_bias[idx] = quantized_bias[idx] * quantized_bias_index[idx] + tmp_valid_quantized_bias * (torch.ones(quantized_bias_index[idx].shape, device=device) - quantized_bias_index[idx])
                # quantized_bias_index[idx] = ((quantized_bias_index[idx] + local_quantize_bias_idx * 1) > 0) * 1

                # if iter == 0:
                #     unquantized_weight_index[idx] = local_unquantized_idx * 1
                # else:
                #     unquantized_weight_index[idx] = ((unquantized_weight_index[idx] - local_unquantized_idx*1)>0)*1

                # print("[DEBUG]", name, torch.sum(quantized_weight_index[idx]), torch.sum(unquantized_weight_index[idx]))
                # quantized_cnt += torch.sum(quantized_weight_index[idx]) # debug
                # unquantized_cnt += torch.sum(unquantized_weight_index[idx]) # debug

                idx += 1
        # print("[DEBUG]", "TOTAL", quantized_cnt, unquantized_cnt)
        # tmp_weights = quantized_weights[1].reshape(quantized_weights[1].numel())
        # print("[DEBUG] iter = " + str(iter) + ":", tmp_weights[0], tmp_weights[100], tmp_weights[200], tmp_weights[300], tmp_weights[400], tmp_weights[500])

        print('step_3: re-training')
        # for epoch in range(args.start_epoch, args.epochs):
        # quanti_sum = 0
        # for j in range(len(quantize_layer_list)):
        #     debug = quantized_weight_index[j].numel() == torch.sum(quantized_weight_index[j]) + torch.sum(unquantized_weight_index[j])
        # print(j, torch.sum(quantized_weight_index[j]), torch.sum(unquantized_weight_index[j]), debug)
        # quanti_sum = quanti_sum + torch.sum(quantized_weight_index[j])
        # print(quanti_sum)
        # print("[DEBUG] Before quantization")
        # validate(net, valloader, criterion, device, ctype, f_attr)
        # validate(net, testloader, criterion, device, ctype, f_attr)
        tmp_state_dict = net.state_dict()
        for j in range(len(quantize_layer_list)):
            # print(j, torch.sum(quantized_weight_index[j]), torch.sum(unquantized_weight_index[j]))
            tmp_state_dict[quantize_layer_list[j]] = quantized_weights[j] * quantized_weight_index[j] + tmp_state_dict[quantize_layer_list[j]] * unquantized_weight_index[j]
            # tmp_state_dict[quantize_layer_list[j] + '.bias'] = quantized_bias[j] * quantized_bias_index[j] + tmp_state_dict[quantize_layer_list[j] + '.bias'] * (torch.ones(quantized_bias_index[j].shape, device=device) - quantized_bias_index[j])
        net.load_state_dict(tmp_state_dict)
        # print("[DEBUG] After quantization")
        # validate(net, valloader, criterion, device, ctype, f_attr)
        torch.save({
            "state_dict": net.state_dict(),
            "quantize_layer_list": quantize_layer_list,
            "quantized_weights": quantized_weights,
            "quantized_weight_index": quantized_weight_index,
            "unquantized_weight_index": unquantized_weight_index
        }, os.path.join(args.log_dir, 'quantized.{}.pth.tar'.format(iter)))

        if args.optimizer == "adam":
            optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "momentum":
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
        elif args.optimizer == "rms":
            optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        best_val_acc = 0.0
        best_val_loss = 100.0
        best_val_eopp0 = 100.0
        best_val_eopp0_abs = 100.0
        best_val_eopp1_abs = 100.0
        best_val_eodds_abs = 100.0
        for epoch in range(args.epochs):  # range(args.start_epoch, n_epoch_debug)
            print('Retrain Epoch: %d' % (epoch + 1))
            sum_loss = 0.0
            correct = 0
            total = 0

            light_score_dataset, dark_score_dataset = genScoreDataset(args, train_df, image_dir, device, False)
            lightloader = torch.utils.data.DataLoader(light_score_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            darkloader = torch.utils.data.DataLoader(dark_score_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            # for i, data in enumerate(tqdm(trainloader)):
            for i, (light_data, dark_data) in enumerate(tzip(lightloader, darkloader)):
                net.train()
                optimizer.zero_grad()
                # prepare dataset
                # inputs, labels = data["image"].float().to(device), torch.from_numpy(np.asarray(data[ctype])).long().to(device)
                light_inputs, light_labels = light_data["image"].float().to(device), torch.from_numpy(np.asarray(light_data[ctype])).long().to(device)
                dark_inputs, dark_labels = dark_data["image"].float().to(device), torch.from_numpy(np.asarray(dark_data[ctype])).long().to(device)

                # forward & backward
                # outputs = net(inputs)
                light_outputs = net(light_inputs)
                dark_outputs = net(dark_inputs)

                # loss = criterion(outputs, labels)
                loss = (1.0 - args.alpha) * criterion(torch.cat([light_outputs, dark_outputs]), torch.cat([light_labels, dark_labels])) + args.alpha * torch.abs(criterion(light_outputs, light_labels) - criterion(dark_outputs, dark_labels))
                loss.backward()
                optimizer.step()

                # print("[DEBUG] Before re-quantization")
                # validate(net, valloader, criterion, device, ctype, f_attr)
                # Re-quantization
                tmp_state_dict = net.state_dict()
                for j in range(len(quantize_layer_list)):
                    # print(j, torch.sum(quantized_weight_index[j]), torch.sum(unquantized_weight_index[j]))
                    tmp_state_dict[quantize_layer_list[j]] = quantized_weights[j] * quantized_weight_index[j] + tmp_state_dict[quantize_layer_list[j]] * unquantized_weight_index[j]
                    # new_quantized_results = 2 ** torch.round(torch.log2(torch.abs(tmp_state_dict[quantize_layer_list[j]]))) * torch.sign(tmp_state_dict[quantize_layer_list[j]])
                    # tmp_state_dict[quantize_layer_list[j]] = new_quantized_results * quantized_weight_index[j] + tmp_state_dict[quantize_layer_list[j]] * unquantized_weight_index[j]
                    # tmp_state_dict[quantize_layer_list[j] + '.bias'] = quantized_bias[j] * quantized_bias_index[j] + tmp_state_dict[quantize_layer_list[j] + '.bias'] * (torch.ones(quantized_bias_index[j].shape, device=device) - quantized_bias_index[j])
                net.load_state_dict(tmp_state_dict)
                # print("[DEBUG] After re-quantization")
                # validate(net, valloader, criterion, device, ctype, f_attr)

                # print ac & loss in each batch
                # sum_loss += loss.item() * labels.size(0)
                # _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += predicted.eq(labels.data).cpu().sum()
                # print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                #     % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
            # print('[epoch:%d] Loss: %.03f | Acc: %.3f%%' % (epoch + 1, sum_loss / total, 100. * correct / total))
            # break # DEBUG
            scheduler.step()

            # Validate
            val_acc, val_loss, val_eopp0, val_eopp0_abs, val_eopp1_abs, val_eodds_abs = validate(net, valloader, criterion, device, ctype, f_attr)

            if val_acc > best_val_acc:
                print("New best val_acc: %.02f%% -> %.02f%%" % (best_val_acc, val_acc))
                best_val_acc = val_acc
                torch.save({
                    "state_dict": net.state_dict(),
                    "quantize_layer_list": quantize_layer_list,
                    "quantized_weights": quantized_weights,
                    "quantized_weight_index": quantized_weight_index,
                    "unquantized_weight_index": unquantized_weight_index
                }, os.path.join(args.log_dir, 'best_acc.{}.pth.tar'.format(iter)))
            if val_loss < best_val_loss:
                print("New best val_loss: %.04f -> %.04f" % (best_val_loss, val_loss))
                best_val_loss = val_loss
                torch.save({
                    "state_dict": net.state_dict(),
                    "quantize_layer_list": quantize_layer_list,
                    "quantized_weights": quantized_weights,
                    "quantized_weight_index": quantized_weight_index,
                    "unquantized_weight_index": unquantized_weight_index
                }, os.path.join(args.log_dir, 'best_loss.{}.pth.tar'.format(iter)))
            if val_eopp0 < best_val_eopp0:
                print("New best val_eopp0: %.04f -> %.04f" % (best_val_eopp0, val_eopp0))
                best_val_eopp0 = val_eopp0
                torch.save({
                    "state_dict": net.state_dict(),
                    "quantize_layer_list": quantize_layer_list,
                    "quantized_weights": quantized_weights,
                    "quantized_weight_index": quantized_weight_index,
                    "unquantized_weight_index": unquantized_weight_index
                }, os.path.join(args.log_dir, 'best_eopp0.{}.pth.tar'.format(iter)))
            if val_eopp0_abs < best_val_eopp0_abs:
                print("New best val_eopp0_abs: %.04f -> %.04f" % (best_val_eopp0_abs, val_eopp0_abs))
                best_val_eopp0_abs = val_eopp0_abs
                torch.save({
                    "state_dict": net.state_dict(),
                    "quantize_layer_list": quantize_layer_list,
                    "quantized_weights": quantized_weights,
                    "quantized_weight_index": quantized_weight_index,
                    "unquantized_weight_index": unquantized_weight_index
                }, os.path.join(args.log_dir, 'best_eopp0_abs.{}.pth.tar'.format(iter)))
            if val_eopp1_abs < best_val_eopp1_abs:
                print("New best val_eopp1_abs: %.04f -> %.04f" % (best_val_eopp1_abs, val_eopp1_abs))
                best_val_eopp1_abs = val_eopp1_abs
                torch.save({
                    "state_dict": net.state_dict(),
                    "quantize_layer_list": quantize_layer_list,
                    "quantized_weights": quantized_weights,
                    "quantized_weight_index": quantized_weight_index,
                    "unquantized_weight_index": unquantized_weight_index
                }, os.path.join(args.log_dir, 'best_eopp1_abs.{}.pth.tar'.format(iter)))
            if val_eodds_abs < best_val_eodds_abs:
                print("New best val_eodds_abs: %.04f -> %.04f" % (best_val_eodds_abs, val_eodds_abs))
                best_val_eodds_abs = val_eodds_abs
                torch.save({
                    "state_dict": net.state_dict(),
                    "quantize_layer_list": quantize_layer_list,
                    "quantized_weights": quantized_weights,
                    "quantized_weight_index": quantized_weight_index,
                    "unquantized_weight_index": unquantized_weight_index
                }, os.path.join(args.log_dir, 'best_eodds_abs.{}.pth.tar'.format(iter)))
