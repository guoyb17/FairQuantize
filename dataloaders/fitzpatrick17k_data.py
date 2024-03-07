import os, random
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import skimage
from skimage import io
import torch
from torch import stack
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataloaders.data_utils import AutoAugment, Cutout_v0


def read_fitzpatrick17k_dataset_metainfo(csv_file_name='fitzpatrick17k_no_histo/fitzpatrick17k_no_histo.csv', dev_mode=None):
    if dev_mode == "dev":
        df = pd.read_csv(csv_file_name).sample(1000)
    else:
        df = pd.read_csv(csv_file_name)
    df['fitzpatrick'].value_counts().sort_index() # Removed print
    # print("Rows: {}".format(df.shape[0]))
    df["low"] = df['label'].astype('category').cat.codes
    df["mid"] = df['nine_partition_label'].astype('category').cat.codes
    df["high"] = df['three_partition_label'].astype('category').cat.codes
    df["hasher"] = df["md5hash"]
    df["image_path"] = df["url"]
    # print(len(df['label'].astype('category').cat.categories))
    # print(len(df['nine_partition_label'].astype('category').cat.categories))
    # print(len(df['three_partition_label'].astype('category').cat.categories))
    return df

def fitzpatrick17k_holdout(df, holdout_sets, ctype):
    """
    df: dataframe of the whole set
    holdout_sets: how to split the data for training and testing
    """
    total_classes = len(df.label.unique())
    for holdout_set in holdout_sets:
        print("holdout policy: {:}".format(holdout_set))
        if holdout_set == "expert_select":
            df2 = df
            train = df2[df2.qc.isnull()]
            test = df2[df2.qc=="1 Diagnostic"]
        elif holdout_set == "random_holdout":
            train, test, y_train, y_test = train_test_split(
                                                df,
                                                eval("df." + ctype),
                                                test_size=0.2,
                                                random_state=4242,
                                                stratify=eval("df." + ctype))
            print(train['fitzpatrick'].value_counts().sort_index())
            print(test['fitzpatrick'].value_counts().sort_index())
        elif holdout_set == "dermaamin":
            combo = set(df[df.image_path.str.contains("dermaamin")==True].label.unique()) & set(df[df.image_path.str.contains("dermaamin")==False].label.unique())
            df2 = df[df.label.isin(combo)]
            df2["low"] = df2['label'].astype('category').cat.codes
            train = df2[df2.image_path.str.contains("dermaamin") == False]
            test = df2[df2.image_path.str.contains("dermaamin") == True]
            print("# common labels: {:}".format(len(combo)))
        elif holdout_set == "br":
            combo = set(df[df.image_path.str.contains("dermaamin")==True].label.unique()) & set(df[df.image_path.str.contains("dermaamin")==False].label.unique())
            df2 = df[df.label.isin(combo)]
            df2["low"] = df2['label'].astype('category').cat.codes
            train = df2[df2.image_path.str.contains("dermaamin") == True]
            test = df2[df2.image_path.str.contains("dermaamin") == False]
            print("# common labels: {:}".format(len(combo)))
            print(train.label.nunique())
            print(test.label.nunique())
        elif holdout_set == "a12":
            train = df[(df.fitzpatrick==1)|(df.fitzpatrick==2)]
            test = df[(df.fitzpatrick!=1)&(df.fitzpatrick!=2)]
            combo = set(train.label.unique()) & set(test.label.unique())
            print("# common labels: {:}/{:}({:}+{:})".format(len(combo), total_classes, len(train.label.unique()), len(test.label.unique())))
            # print(combo)
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train['label'].astype('category').cat.codes
            test["low"] = test['label'].astype('category').cat.codes
        elif holdout_set == "a34":
            train = df[(df.fitzpatrick==3)|(df.fitzpatrick==4)]
            test = df[(df.fitzpatrick!=3)&(df.fitzpatrick!=4)]
            combo = set(train.label.unique()) & set(test.label.unique())
            print("# common labels: {:}/{:}({:}+{:})".format(len(combo), total_classes, len(train.label.unique()), len(test.label.unique())))
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train['label'].astype('category').cat.codes
            test["low"] = test['label'].astype('category').cat.codes
        elif holdout_set == "a56":
            train = df[(df.fitzpatrick==5)|(df.fitzpatrick==6)]
            test = df[(df.fitzpatrick!=5)&(df.fitzpatrick!=6)]
            combo = set(train.label.unique()) & set(test.label.unique())
            print("# common labels: {:}/{:}({:}+{:})".format(len(combo), total_classes, len(train.label.unique()), len(test.label.unique())))
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train['label'].astype('category').cat.codes
            test["low"] = test['label'].astype('category').cat.codes
        elif holdout_set == "a123":
            train = df[(df.fitzpatrick==4)|(df.fitzpatrick==5)|(df.fitzpatrick==6)]
            test = df[(df.fitzpatrick!=4)&(df.fitzpatrick!=5)&(df.fitzpatrick!=6)]
            combo = set(train.label.unique()) & set(test.label.unique())
            print("# common labels: {:}/{:}({:}+{:})".format(len(combo), total_classes, len(train.label.unique()), len(test.label.unique())))
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train['label'].astype('category').cat.codes
            test["low"] = test['label'].astype('category').cat.codes
        elif holdout_set == "a456":
            train = df[(df.fitzpatrick==1)|(df.fitzpatrick==2)|(df.fitzpatrick==3)]
            test = df[(df.fitzpatrick!=1)&(df.fitzpatrick!=2)&(df.fitzpatrick!=3)]
            combo = set(train.label.unique()) & set(test.label.unique())
            print("# common labels: {:}/{:}({:}+{:})".format(len(combo), total_classes, len(train.label.unique()), len(test.label.unique())))
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["low"] = train['label'].astype('category').cat.codes
            test["low"] = test['label'].astype('category').cat.codes

    return train, test

def fitzpatrick17k_holdout_score(df, holdout_sets, ctype):
    for holdout_set in holdout_sets:
        print("holdout policy: {:}".format(holdout_set))
        try:
            train_and_score = read_fitzpatrick17k_dataset_metainfo(csv_file_name="fitzpatrick17k/train_score.csv")
            val = read_fitzpatrick17k_dataset_metainfo(csv_file_name="fitzpatrick17k/all_val.csv")
            test = read_fitzpatrick17k_dataset_metainfo(csv_file_name="fitzpatrick17k/all_test.csv")
        except:
            train_and_score, val_and_test, _, _ = train_test_split(
                                            df,
                                            eval("df." + ctype),
                                            test_size=0.2,
                                            random_state=4242,
                                            stratify=eval("df." + ctype))
            # train, score, _, _ = train_test_split(
            #                                 train_and_score,
            #                                 eval("train_and_score." + ctype),
            #                                 test_size=0.125,
            #                                 random_state=4242,
            #                                 stratify=eval("train_and_score." + ctype))
            val, test, _, _ = train_test_split(
                                            val_and_test,
                                            eval("val_and_test." + ctype),
                                            test_size=0.5,
                                            random_state=4242,
                                            stratify=eval("val_and_test." + ctype))
            # train.to_csv("fitzpatrick17k/all_train.csv")
            # score.to_csv("fitzpatrick17k/all_score.csv")
            val.to_csv("fitzpatrick17k/all_val.csv")
            test.to_csv("fitzpatrick17k/all_test.csv")
            train_and_score.to_csv("fitzpatrick17k/train_score.csv")
            # print("[DEBUG]", "Train_and_score", train_and_score.shape[0]) # 13223
            # train = read_fitzpatrick17k_dataset_metainfo(csv_file_name="fitzpatrick17k/all_train.csv")
            # score = read_fitzpatrick17k_dataset_metainfo(csv_file_name="fitzpatrick17k/all_score.csv")

        score_size = 799 # 12776 / 0.8 * 0.05 = 798.5 (8857 + 3919 = 12776; there are some images with the fitzpatrick value of -1)
        
        try:
            light_data = read_fitzpatrick17k_dataset_metainfo(csv_file_name="fitzpatrick17k/light_score.csv")
            dark_data = read_fitzpatrick17k_dataset_metainfo(csv_file_name="fitzpatrick17k/dark_score.csv")
        except:
            light_data = train_and_score
            dark_data = train_and_score
            light_data = light_data.drop(light_data[(light_data["fitzpatrick"] < 1) | (light_data["fitzpatrick"] > 3)].index)
            # print("[DEBUG]", "Light", light_data.shape[0]) # 8857
            light_data = light_data.sample(n=score_size)
            light_data.to_csv("fitzpatrick17k/light_score.csv")
            dark_data = dark_data.drop(dark_data[(dark_data["fitzpatrick"] < 4) | (dark_data["fitzpatrick"] > 6)].index)
            # print("[DEBUG]", "Dark", dark_data.shape[0]) # 3919
            dark_data = dark_data.sample(n=score_size)
            dark_data.to_csv("fitzpatrick17k/dark_score.csv")
            score = pd.concat([light_data, dark_data])
            score.to_csv("fitzpatrick17k/all_score.csv")

        # train[ctype].value_counts().sort_index() # 70%
        # score[ctype].value_counts().sort_index() # ~10%
        train_and_score[ctype].value_counts().sort_index() # 80% (13223)
        val[ctype].value_counts().sort_index() # 10%
        test[ctype].value_counts().sort_index() # 10%
        light_data[ctype].value_counts().sort_index() # ~5% (799)
        dark_data[ctype].value_counts().sort_index() # ~5% (799)

    return train_and_score, light_data, dark_data, val, test

def fitzpatrick17k_holdout_score_v2(df, ctype):
    try:
        train_and_score = read_fitzpatrick17k_dataset_metainfo(csv_file_name="fitzpatrick17k/train_score.csv")
        val = read_fitzpatrick17k_dataset_metainfo(csv_file_name="fitzpatrick17k/all_val.csv")
        test = read_fitzpatrick17k_dataset_metainfo(csv_file_name="fitzpatrick17k/all_test.csv")
    except:
        train_and_score, val_and_test, _, _ = train_test_split(
                                        df,
                                        eval("df." + ctype),
                                        test_size=0.4,
                                        random_state=4242,
                                        stratify=eval("df." + ctype))
        # train, score, _, _ = train_test_split(
        #                                 train_and_score,
        #                                 eval("train_and_score." + ctype),
        #                                 test_size=0.125,
        #                                 random_state=4242,
        #                                 stratify=eval("train_and_score." + ctype))
        val, test, _, _ = train_test_split(
                                        val_and_test,
                                        eval("val_and_test." + ctype),
                                        test_size=0.5,
                                        random_state=4242,
                                        stratify=eval("val_and_test." + ctype))
        # train.to_csv("fitzpatrick17k/all_train.csv")
        # score.to_csv("fitzpatrick17k/all_score.csv")
        val.to_csv("fitzpatrick17k/all_val.csv")
        test.to_csv("fitzpatrick17k/all_test.csv")
        train_and_score.to_csv("fitzpatrick17k/train_score.csv")
        # print("[DEBUG]", "Train_and_score", train_and_score.shape[0]) # 13223
        # train = read_fitzpatrick17k_dataset_metainfo(csv_file_name="fitzpatrick17k/all_train.csv")
        # score = read_fitzpatrick17k_dataset_metainfo(csv_file_name="fitzpatrick17k/all_score.csv")

    # train[ctype].value_counts().sort_index() # 70%
    # score[ctype].value_counts().sort_index() # ~10%
    train_and_score[ctype].value_counts().sort_index() # 60%
    val[ctype].value_counts().sort_index() # 20%
    test[ctype].value_counts().sort_index() # 20%

    return train_and_score, val, test

def fitzpatrick17k_holdout_test_deploy(csv_file_name, ctype):
    test = read_fitzpatrick17k_dataset_metainfo(csv_file_name=csv_file_name)
    test[ctype].value_counts().sort_index() # 10%
    return test

class Fitzpatrick_17k_Augmentations():
    def __init__(self, is_training, image_size=256, input_size=224):
        mdlParams = dict()
        
        mdlParams['setMean'] = np.array([0.0, 0.0, 0.0])   
        mdlParams['setStd'] = np.array([1.0, 1.0, 1.0])
        self.image_size = image_size
        mdlParams['input_size'] = [input_size, input_size, 3]
        self.input_size = (np.int32(mdlParams['input_size'][0]),np.int32(mdlParams['input_size'][1]))   

        # training augmentations
        if is_training:
            self.same_sized_crop = True
            self.only_downsmaple = False

            # For Fitzpatrik17k dataset
            #mdlParams['full_color_distort'] = True
            mdlParams['autoaugment'] = True     
            mdlParams['flip_lr_ud'] = True
            mdlParams['full_rot'] = 180
            mdlParams['scale'] = (0.8,1.2)
            mdlParams['shear'] = 10
            mdlParams['cutout'] = 16

            # For PAD-UFES-20 dataset
            # mdlParams['autoaugment'] = False     
            # mdlParams['flip_lr_ud'] = True
            # mdlParams['full_rot'] = 180
            # mdlParams['scale'] = (0.9,1.1)
            # mdlParams['shear'] = 10
            # mdlParams['cutout'] = 10
            
            # For celebA dataset
            # mdlParams['autoaugment'] = False     
            # mdlParams['flip_lr_ud'] = True
            # mdlParams['full_rot'] = 180
            # mdlParams['scale'] = (0.95,1.05)
            # mdlParams['shear'] = 5
            # mdlParams['cutout'] = 10

            transforms = self.get_train_augmentations(mdlParams)
        else:
            # test augmentations
            transforms = self.get_test_augmentations(mdlParams)
        self.transforms = transforms
    
    def get_test_augmentations(self, mdlParams):
        all_transforms = [
                transforms.ToPILImage(),
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(np.float32(mdlParams['setMean']),np.float32(mdlParams['setStd']))]
        composed = transforms.Compose(all_transforms)
        return composed

    def get_train_augmentations(self, mdlParams):
        all_transforms = [transforms.ToPILImage()]
        # Normal train proc
        if self.same_sized_crop:
            all_transforms.append(transforms.Resize(self.image_size))
            all_transforms.append(transforms.RandomCrop(self.input_size))
        elif self.only_downsmaple:
            all_transforms.append(transforms.Resize(self.input_size))
        else:
            all_transforms.append(transforms.RandomResizedCrop(self.input_size[0],scale=(mdlParams.get('scale_min',0.08),1.0)))
        if mdlParams.get('flip_lr_ud',False):
            all_transforms.append(transforms.RandomHorizontalFlip())
            all_transforms.append(transforms.RandomVerticalFlip())
        # Full rot
        if mdlParams.get('full_rot',0) > 0:
            if mdlParams.get('scale',False):
                all_transforms.append(transforms.RandomChoice([transforms.RandomAffine(mdlParams['full_rot'], scale=mdlParams['scale'], shear=mdlParams.get('shear',0), interpolation=InterpolationMode.NEAREST),
                                                            transforms.RandomAffine(mdlParams['full_rot'],scale=mdlParams['scale'],shear=mdlParams.get('shear',0), interpolation=InterpolationMode.BICUBIC),
                                                            transforms.RandomAffine(mdlParams['full_rot'],scale=mdlParams['scale'],shear=mdlParams.get('shear',0), interpolation=InterpolationMode.BILINEAR)])) 
            else:
                all_transforms.append(transforms.RandomChoice([transforms.RandomRotation(mdlParams['full_rot'], interpolation=InterpolationMode.NEAREST),
                                                            transforms.RandomRotation(mdlParams['full_rot'], interpolation=InterpolationMode.BICUBIC),
                                                            transforms.RandomRotation(mdlParams['full_rot'], interpolation=InterpolationMode.BILINEAR)]))    
        # Color distortion
        if mdlParams.get('full_color_distort') is not None:
            all_transforms.append(transforms.ColorJitter(brightness=mdlParams.get('brightness_aug',32. / 255.),saturation=mdlParams.get('saturation_aug',0.5), contrast = mdlParams.get('contrast_aug',0.5), hue = mdlParams.get('hue_aug',0.2)))
        else:
            all_transforms.append(transforms.ColorJitter(brightness=32. / 255.,saturation=0.5))   
        # Autoaugment
        if mdlParams.get('autoaugment',False):
            all_transforms.append(AutoAugment())             
        # Cutout
        if mdlParams.get('cutout',0) > 0:
            all_transforms.append(Cutout_v0(n_holes=1,length=mdlParams['cutout']))                             
        # Normalize
        all_transforms.append(transforms.ToTensor())
        all_transforms.append(transforms.Normalize(np.float32(mdlParams['setMean']),np.float32(mdlParams['setStd'])))            
        # All transforms
        composed = transforms.Compose(all_transforms)         

        return composed

class Fitzpatrick17k(Dataset):

    def __init__(self, df=None, root_dir=None, transform=None):
        """
        Args:
            train: True for training, False for testing
            transform (callable, optional): Optional transform to be applied
                on a sample.
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        # if train:
        #     predefined_csv_file = 'e:/data/fitzpatrick17k_no_histo/fitzpatrick17k_no_histo.csv'
        # else:
        #     predefined_root_file = ''
        # predefined_root_dir = 'e:/data/fitzpatrick17k_no_histo/dataset_images'
        # csv_file = predefined_csv_file if csv_file is None else csv_file
        # root_dir = predefined_root_dir if root_dir is None else root_dir
        # self.df = pd.read_csv(csv_file)
        assert df is not None
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(idx)
        # print(self.df.loc[self.df.index[idx], 'hasher']+'.jpg')
        img_name = os.path.join(self.root_dir,
                                self.df.loc[self.df.index[idx], 'hasher']+'.jpg')
        # tmp_cnt = 0
        # while tmp_cnt < 10:
        #     if tmp_cnt > 0:
        #         print("[Warning] Cycling, tmp_cnt =", tmp_cnt)
        #     try:
        #         image = io.imread(img_name)
        #     except:
        #         tmp_cnt += 1
        #         # sleep(0.1)
        # if tmp_cnt >= 10:
        #     raise NotImplementedError
        image = io.imread(img_name)
        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)

        hasher = self.df.loc[self.df.index[idx], 'hasher']
        high = self.df.loc[self.df.index[idx], 'high']
        mid = self.df.loc[self.df.index[idx], 'mid']
        low = self.df.loc[self.df.index[idx], 'low']
        fitzpatrick = self.df.loc[self.df.index[idx], 'fitzpatrick']
        if 1 <= fitzpatrick <= 3:
            skin_color_binary = 0
        elif 4 <= fitzpatrick <= 6:
            skin_color_binary = 1
        else:
            skin_color_binary = -1
        if self.transform:
            image = self.transform(image)
        sample = {
                    'image': image,
                    'high': high,
                    'mid': mid,
                    'low': low,
                    'hasher': hasher,
                    'fitzpatrick': fitzpatrick,
                    'skin_color_binary': skin_color_binary,
                }
        # sample = (image, int(val))
        return sample

class Fitzpatrick17kV2(Dataset):

    def __init__(self, df, root_dir=None, transform=None):
        """
        Args:
            train: True for training, False for testing
            transform (callable, optional): Optional transform to be applied
                on a sample.
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

        self.images = []
        self.info = []
        for idx in tqdm(range(len(self.df))):
            img_name = os.path.join(self.root_dir,
                                    self.df.loc[self.df.index[idx], 'hasher']+'.jpg')
            image = io.imread(img_name)
            if(len(image.shape) < 3):
                image = skimage.color.gray2rgb(image)

            hasher = self.df.loc[self.df.index[idx], 'hasher']
            high = self.df.loc[self.df.index[idx], 'high']
            mid = self.df.loc[self.df.index[idx], 'mid']
            low = self.df.loc[self.df.index[idx], 'low']
            fitzpatrick = self.df.loc[self.df.index[idx], 'fitzpatrick']
            if 1 <= fitzpatrick <= 3:
                skin_color_binary = 0
            elif 4 <= fitzpatrick <= 6:
                skin_color_binary = 1
            else:
                skin_color_binary = -1
            if self.transform:
                image = self.transform(image)
            self.images.append(image)
            self.info.append({
                        'high': high,
                        'mid': mid,
                        'low': low,
                        'hasher': hasher,
                        'fitzpatrick': fitzpatrick,
                        'skin_color_binary': skin_color_binary,
                    })
        self.images = stack(self.images)

    def to(self, device):
        self.images = self.images.to(device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.info[idx]
        sample['image'] = self.images[idx]
        return sample

def get_weighted_sampler(df, label_level = 'mid'):
    class_sample_count = np.array(df[label_level].value_counts().sort_index())
    class_weight = 1. / class_sample_count
    samples_weight = np.array([class_weight[t] for t in df[label_level]])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)
    return sampler

def fitzpatric17k_dataloader(dataset, batch_size, train, workers, length=None, num_classes=9):
    if num_classes == 3:
        ctype = "high"
    elif num_classes == 9:
        ctype = "mid"
    elif num_classes == 114:
        ctype = "low"
    else:
        raise NotImplementedError
    # predefined_root_dir = '/data/users/dewenzeng/data/fitzpatrick17k/dataset_images' # specify the image dir
    # df = read_fitzpatrick17k_dataset_metainfo(csv_file_name='/data/users/dewenzeng/data/fitzpatrick17k/fitzpatrick17k_no_histo.csv')
    # predefined_root_dir = 'e:/data/fitzpatrick17k_no_histo/dataset_images' # specify the image dir, in my local
    # df = read_fitzpatrick17k_dataset_metainfo(csv_file_name='e:/data/fitzpatrick17k_no_histo/fitzpatrick17k_no_histo.csv')
    predefined_root_dir = 'fitzpatrick17k/dataset_images' # specify the image dir
    df = read_fitzpatrick17k_dataset_metainfo(csv_file_name='fitzpatrick17k/fitzpatrick17k.csv')
    train_df, test_df = fitzpatrick17k_holdout(df, ["random_holdout"], ctype)
    # print(f'train_df:{train_df}, test_df:{test_df}')
    image_size = 256 // 2
    crop_size = 224 // 2
    train_transform = Fitzpatrick_17k_Augmentations(is_training=True, image_size=image_size, input_size=crop_size).transforms
    test_transform = Fitzpatrick_17k_Augmentations(is_training=False, image_size=image_size, input_size=crop_size).transforms
    train_dataset = Fitzpatrick17k(df=train_df, root_dir=predefined_root_dir, transform=train_transform, ctype=ctype)
    test_dataset = Fitzpatrick17k(df=test_df, root_dir=predefined_root_dir, transform=test_transform, ctype=ctype)
    # prune_df_color1, _ = fitzpatrick17k_holdout(train_df, ["a456"], ctype)
    # prune_df_color2, _ = fitzpatrick17k_holdout(train_df, ["a123"], ctype)
    # prune_dataset_all = Fitzpatrick17k(df=train_df, root_dir=predefined_root_dir, transform=test_transform)
    # prune_dataset_color1 = Fitzpatrick17k(df=prune_df_color1, root_dir=predefined_root_dir, transform=test_transform)
    # prune_dataset_color2 = Fitzpatrick17k(df=prune_df_color2, root_dir=predefined_root_dir, transform=test_transform)
    # now we are also interested in training with only part of the data
    # train_dataset_color1 = Fitzpatrick17k(df=prune_df_color1, root_dir=predefined_root_dir, transform=train_transform)
    # train_dataset_color2 = Fitzpatrick17k(df=prune_df_color2, root_dir=predefined_root_dir, transform=train_transform)
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    sampler = get_weighted_sampler(train_df, ctype)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, **kwargs)
    # sampler = get_weighted_sampler(prune_df_color1)
    # train_dataloader_color1 = torch.utils.data.DataLoader(train_dataset_color1, batch_size=batch_size, sampler=sampler, **kwargs)
    # sampler = get_weighted_sampler(prune_df_color2)
    # train_dataloader_color2 = torch.utils.data.DataLoader(train_dataset_color2, batch_size=batch_size, sampler=sampler, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    # use a subset of dataset for debugging
    # prune_dataset_all = torch.utils.data.Subset(prune_dataset_all, torch.randperm(len(prune_dataset_all))[:2000])
    # prune_dataset_color1 = torch.utils.data.Subset(prune_dataset_color1, torch.randperm(len(prune_dataset_color1))[:2000])
    # prune_dataset_color2 = torch.utils.data.Subset(prune_dataset_color2, torch.randperm(len(prune_dataset_color2))[:2000])
    # prune_dataloader_all = torch.utils.data.DataLoader(prune_dataset_all, batch_size=prune_batch_size, shuffle=False, **kwargs)
    # prune_dataloader_color1 = torch.utils.data.DataLoader(prune_dataset_color1, batch_size=prune_batch_size, shuffle=False, **kwargs)
    # prune_dataloader_color2 = torch.utils.data.DataLoader(prune_dataset_color2, batch_size=prune_batch_size, shuffle=False, **kwargs)

    # return train_dataloader, test_dataloader, prune_dataloader_all, prune_dataloader_color1, prune_dataloader_color2, train_dataloader_color1, train_dataloader_color2
    return train_dataloader, test_dataloader

def fitzpatric17k_dataloader_score(batch_size, workers, predefined_root_dir='fitzpatrick17k/dataset_images', csv_file_name='fitzpatrick17k/fitzpatrick17k.csv', ctype='mid'):
    df = read_fitzpatrick17k_dataset_metainfo(csv_file_name=csv_file_name)
    train_df, light_score_df, dark_score_df, val_df, test_df = fitzpatrick17k_holdout_score(df, ["random_holdout"], ctype)
    image_size = 256 // 2
    crop_size = 224 // 2
    train_transform = Fitzpatrick_17k_Augmentations(is_training=True, image_size=image_size, input_size=crop_size).transforms
    test_transform = Fitzpatrick_17k_Augmentations(is_training=False, image_size=image_size, input_size=crop_size).transforms
    train_dataset = Fitzpatrick17k(df=train_df, root_dir=predefined_root_dir, transform=train_transform)
    light_score_dataset = Fitzpatrick17k(df=light_score_df, root_dir=predefined_root_dir, transform=test_transform)
    dark_score_dataset = Fitzpatrick17k(df=dark_score_df, root_dir=predefined_root_dir, transform=test_transform)
    val_dataset = Fitzpatrick17k(df=val_df, root_dir=predefined_root_dir, transform=test_transform)
    test_dataset = Fitzpatrick17k(df=test_df, root_dir=predefined_root_dir, transform=test_transform)
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,**kwargs)
    light_score_dataloader = torch.utils.data.DataLoader(light_score_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    dark_score_dataloader = torch.utils.data.DataLoader(dark_score_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_dataloader, light_score_dataloader, dark_score_dataloader, val_dataloader, test_dataloader

def fitzpatric17k_dataloader_score_v2(batch_size, workers, predefined_root_dir='fitzpatrick17k/dataset_images', csv_file_name='fitzpatrick17k/fitzpatrick17k.csv', ctype='low'):
    df = read_fitzpatrick17k_dataset_metainfo(csv_file_name=csv_file_name)
    train_df, val_df, test_df = fitzpatrick17k_holdout_score_v2(df, ctype)
    image_size = 256 // 2
    crop_size = 224 // 2
    train_transform = Fitzpatrick_17k_Augmentations(is_training=True, image_size=image_size, input_size=crop_size).transforms
    test_transform = Fitzpatrick_17k_Augmentations(is_training=False, image_size=image_size, input_size=crop_size).transforms
    train_dataset = Fitzpatrick17k(df=train_df, root_dir=predefined_root_dir, transform=train_transform)
    val_dataset = Fitzpatrick17k(df=val_df, root_dir=predefined_root_dir, transform=test_transform)
    test_dataset = Fitzpatrick17k(df=test_df, root_dir=predefined_root_dir, transform=test_transform)
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, **kwargs)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_dataloader, val_dataloader, test_dataloader, train_df

def fitzpatric17k_dataloader_score_v2s(batch_size, workers, predefined_root_dir='fitzpatrick17k/dataset_images', csv_file_name='fitzpatrick17k/fitzpatrick17k.csv', ctype='low'):
    df = read_fitzpatrick17k_dataset_metainfo(csv_file_name=csv_file_name)
    train_df, val_df, _ = fitzpatrick17k_holdout_score_v2(df, ctype)
    image_size = 256 // 2
    crop_size = 224 // 2
    train_transform = Fitzpatrick_17k_Augmentations(is_training=True, image_size=image_size, input_size=crop_size).transforms
    test_transform = Fitzpatrick_17k_Augmentations(is_training=False, image_size=image_size, input_size=crop_size).transforms
    train_dataset = Fitzpatrick17k(df=train_df, root_dir=predefined_root_dir, transform=train_transform)
    val_dataset = Fitzpatrick17k(df=val_df, root_dir=predefined_root_dir, transform=test_transform)
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, **kwargs)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_dataloader, val_dataloader, df

def fitzpatric17k_dataloader_score_v3(batch_size, workers, predefined_root_dir='fitzpatrick17k/dataset_images', csv_file_name='fitzpatrick17k/fitzpatrick17k.csv', ctype='low', device="cpu", enable_train=True, enable_val=True, enable_test=True):
    df = read_fitzpatrick17k_dataset_metainfo(csv_file_name=csv_file_name)
    train_df, val_df, test_df = fitzpatrick17k_holdout_score_v2(df, ctype)
    image_size = 256
    crop_size = 224
    train_transform = Fitzpatrick_17k_Augmentations(is_training=True, image_size=image_size, input_size=crop_size).transforms
    test_transform = Fitzpatrick_17k_Augmentations(is_training=False, image_size=image_size, input_size=crop_size).transforms
    if enable_train:
        train_dataset = Fitzpatrick17kV2(df=train_df, root_dir=predefined_root_dir, transform=train_transform)
        train_dataset.to(device)
    else:
        train_dataset = Fitzpatrick17k(df=train_df, root_dir=predefined_root_dir, transform=train_transform)
    if enable_val:
        val_dataset = Fitzpatrick17kV2(df=val_df, root_dir=predefined_root_dir, transform=test_transform)
        val_dataset.to(device)
    else:
        val_dataset = Fitzpatrick17k(df=val_df, root_dir=predefined_root_dir, transform=test_transform)
    if enable_test:
        test_dataset = Fitzpatrick17kV2(df=test_df, root_dir=predefined_root_dir, transform=test_transform)
        test_dataset.to(device)
    else:
        test_dataset = Fitzpatrick17k(df=test_df, root_dir=predefined_root_dir, transform=test_transform)
    kwargs = {'num_workers': workers, 'pin_memory': True} if device != "cpu" else {}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, **kwargs if not enable_train else {})
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs if not enable_val else {})
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs if not enable_test else {})

    return train_dataloader, val_dataloader, test_dataloader, train_df

def fitzpatric17k_test_deploy(batch_size, predefined_root_dir='fitzpatrick17k/dataset_images', csv_file_name='fitzpatrick17k/selected.csv', ctype='mid'):
    df = fitzpatrick17k_holdout_test_deploy(csv_file_name=csv_file_name, ctype=ctype)
    image_size = 256 // 2
    crop_size = 224 // 2
    test_transform = Fitzpatrick_17k_Augmentations(is_training=False, image_size=image_size, input_size=crop_size).transforms
    test_dataset = Fitzpatrick17k(df=df, root_dir=predefined_root_dir, transform=test_transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_dataloader

def fitzpatric17k_pruning_examples(predefined_root_dir="pruning_examples/fitzpatrick17k", csv_file_name="pruning_examples/fitzpatrick17k.csv", ctype="low", minimum=False):
    df = fitzpatrick17k_holdout_test_deploy(csv_file_name=csv_file_name, ctype=ctype)
    image_size = 256 // 2
    crop_size = 224 // 2
    test_transform = Fitzpatrick_17k_Augmentations(is_training=False, image_size=image_size, input_size=crop_size).transforms

    images = []
    labels = []
    for idx in tqdm(range(len(df))):
        img_name = os.path.join(predefined_root_dir, df.loc[df.index[idx], 'hasher'] + '.jpg')
        image = io.imread(img_name)
        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)
        image = test_transform(image)
        images.append(image)
        labels.append(df.loc[df.index[idx], ctype])
        if minimum:
            break

    return images, labels

if __name__ == "__main__":
    # df = read_fitzpatrick17k_dataset_metainfo(csv_file_name='fitzpatrick17k/fitzpatrick17k.csv')
    # for index, row in df.iterrows():
    #     try:
    #         os.system("wget " + row["image_path"] + " -O " + os.path.join("fitzpatrick17k", "dataset_images", row["hasher"] + ".jpg"))
    #     except:
    #         print("FAIL:", row["hasher"])
    fitzpatric17k_dataloader_score(8, 8)
