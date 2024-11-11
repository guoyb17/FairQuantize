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


def read_isic2019_dataset_metainfo_raw(csv_file_name='ISIC_2019_train/ISIC_2019_Training_metadata.csv'):
    df = pd.read_csv(csv_file_name)
    return df

def read_isic2019_dataset_metainfo(csv_file_name):
    df = pd.read_csv(csv_file_name)
    return df

def isic2019_holdout_score(df):
    try:
        train_and_score = read_isic2019_dataset_metainfo(csv_file_name="ISIC_2019_train/train_score.csv")
        val = read_isic2019_dataset_metainfo(csv_file_name="ISIC_2019_train/all_val.csv")
        test = read_isic2019_dataset_metainfo(csv_file_name="ISIC_2019_train/all_test.csv")
    except:
        train_and_score, val_and_test, _, _ = train_test_split(
                                        df,
                                        df["class_id"],
                                        test_size=0.2,
                                        random_state=4242,
                                        stratify=df["class_id"])
        val, test, _, _ = train_test_split(
                                        val_and_test,
                                        val_and_test["class_id"],
                                        test_size=0.5,
                                        random_state=4242,
                                        stratify=val_and_test["class_id"])
        val.to_csv("ISIC_2019_train/all_val.csv")
        test.to_csv("ISIC_2019_train/all_test.csv")
        train_and_score.to_csv("ISIC_2019_train/train_score.csv")
        # print("[DEBUG]", "Train_and_score", train_and_score.shape[0]) # 19957

    score_size = 1248 # 19957 / 0.8 * 0.05 = 1247.3125

    try:
        male_data = read_isic2019_dataset_metainfo(csv_file_name="ISIC_2019_train/male_score.csv")
        female_data = read_isic2019_dataset_metainfo(csv_file_name="ISIC_2019_train/female_score.csv")
    except:
        male_data = train_and_score
        female_data = train_and_score
        male_data = male_data.drop(male_data[male_data["sex_id"] == 0].index)
        # print("[DEBUG]", "Male", male_data.shape[0]) # 9332
        male_data = male_data.sample(n=score_size)
        male_data.to_csv("ISIC_2019_train/male_score.csv")
        female_data = female_data.drop(female_data[female_data["sex_id"] == 1].index)
        # print("[DEBUG]", "Female", female_data.shape[0]) # 10625
        female_data = female_data.sample(n=score_size)
        female_data.to_csv("ISIC_2019_train/female_score.csv")

    train_and_score["class_id"].value_counts().sort_index() # 80% (19957)
    val["class_id"].value_counts().sort_index() # 10%
    test["class_id"].value_counts().sort_index() # 10%
    male_data["class_id"].value_counts().sort_index() # ~5% (1248)
    female_data["class_id"].value_counts().sort_index() # ~5% (1248)

    return train_and_score, female_data, male_data, val, test # Female is major

def isic2019_holdout_score_v2(df):
    try:
        train_and_score = read_isic2019_dataset_metainfo(csv_file_name="ISIC_2019_train/train_score.csv")
        val = read_isic2019_dataset_metainfo(csv_file_name="ISIC_2019_train/all_val.csv")
        test = read_isic2019_dataset_metainfo(csv_file_name="ISIC_2019_train/all_test.csv")
    except:
        train_and_score, val_and_test, _, _ = train_test_split(
                                        df,
                                        df["class_id"],
                                        test_size=0.4,
                                        random_state=4242,
                                        stratify=df["class_id"])
        val, test, _, _ = train_test_split(
                                        val_and_test,
                                        val_and_test["class_id"],
                                        test_size=0.5,
                                        random_state=4242,
                                        stratify=val_and_test["class_id"])
        val.to_csv("ISIC_2019_train/all_val.csv")
        test.to_csv("ISIC_2019_train/all_test.csv")
        train_and_score.to_csv("ISIC_2019_train/train_score.csv")
        # print("[DEBUG]", "Train_and_score", train_and_score.shape[0]) # 19957

    train_and_score["class_id"].value_counts().sort_index() # 80% (19957)
    val["class_id"].value_counts().sort_index() # 10%
    test["class_id"].value_counts().sort_index() # 10%

    return train_and_score, val, test # Female = 0, Male = 1

def get_weighted_sampler(df, label_level):
    class_sample_count = np.array(df[label_level].value_counts().sort_index())
    class_weight = 1. / class_sample_count
    samples_weight = np.array([class_weight[t] for t in df[label_level]])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)
    return sampler

class ISIC2019(Dataset):
    def __init__(self, df=None, root_dir=None, transform=None, s=False):
        """
        Args:
            train: True for training, False for testing
            transform (callable, optional): Optional transform to be applied
                on a sample.
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        assert df is not None
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.s = s

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.df.loc[self.df.index[idx], "image"] + ".jpg")
        image = io.imread(img_name)
        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)

        fair_attr = self.df.loc[self.df.index[idx], "sex_id"]
        y_attr = self.df.loc[self.df.index[idx], "class_id"]
        if self.transform:
            image = self.transform(image)
        sample = {
                    'image': image,
                    'fair_attr': fair_attr,
                    'y_attr': y_attr,
                }
        if self.s:
            sample["img"] = self.df.loc[self.df.index[idx], "image"]
        # sample = (image, int(val))
        return sample

class ISIC2019V2(Dataset):
    def __init__(self, df=None, root_dir=None, transform=None):
        """
        Args:
            train: True for training, False for testing
            transform (callable, optional): Optional transform to be applied
                on a sample.
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        assert df is not None
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

        self.images = []
        for idx in tqdm(range(len(self.df))):
            img_name = os.path.join(self.root_dir, self.df.loc[self.df.index[idx], "image"] + ".jpg")
            image = io.imread(img_name)
            if(len(image.shape) < 3):
                image = skimage.color.gray2rgb(image)
            if self.transform:
                image = self.transform(image)
                # init_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256 // 2), transforms.CenterCrop(256 // 2), transforms.ToTensor()]) # TODO: change image size
                # image = init_transforms(image)
            self.images.append(image)
        self.images = stack(self.images)

    def to(self, device):
        self.images = self.images.to(device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.images[idx]
        # if self.transform:
        #     image = self.transform(image)
        fair_attr = self.df.loc[self.df.index[idx], "sex_id"]
        y_attr = self.df.loc[self.df.index[idx], "class_id"]
        sample = {
                    'image': image,
                    'fair_attr': fair_attr,
                    'y_attr': y_attr,
                }
        # sample = (image, int(val))
        return sample

def isic2019_dataloader_score(batch_size, workers, predefined_root_dir='ISIC_2019_train/ISIC_2019_Training_Input', csv_file_name='ISIC_2019_train/ISIC_2019_Training_Metadata.csv'):
    df = read_isic2019_dataset_metainfo_raw(csv_file_name=csv_file_name)
    train_df, major_score_df, minor_score_df, val_df, test_df = isic2019_holdout_score(df)
    image_size = 256 // 2
    crop_size = 224 // 2
    train_transform = ISIC2019_Augmentations(is_training=True, image_size=image_size, input_size=crop_size).transforms
    test_transform = ISIC2019_Augmentations(is_training=False, image_size=image_size, input_size=crop_size).transforms
    train_dataset = ISIC2019(df=train_df, root_dir=predefined_root_dir, transform=train_transform)
    major_score_dataset = ISIC2019(df=major_score_df, root_dir=predefined_root_dir, transform=test_transform)
    minor_score_dataset = ISIC2019(df=minor_score_df, root_dir=predefined_root_dir, transform=test_transform)
    val_dataset = ISIC2019(df=val_df, root_dir=predefined_root_dir, transform=test_transform)
    test_dataset = ISIC2019(df=test_df, root_dir=predefined_root_dir, transform=test_transform)
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, **kwargs)
    major_score_dataloader = torch.utils.data.DataLoader(major_score_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    minor_score_dataloader = torch.utils.data.DataLoader(minor_score_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_dataloader, major_score_dataloader, minor_score_dataloader, val_dataloader, test_dataloader

def isic2019_dataloader_score_v2(batch_size, workers, predefined_root_dir='ISIC_2019_train/ISIC_2019_Training_Input', csv_file_name='ISIC_2019_train/ISIC_2019_Training_Metadata.csv', use_val=False, use_test=False):
    df = read_isic2019_dataset_metainfo_raw(csv_file_name=csv_file_name)
    train_df, val_df, test_df = isic2019_holdout_score_v2(df)
    image_size = 256 // 2
    crop_size = 224 // 2
    train_transform = ISIC2019_Augmentations(is_training=True, image_size=image_size, input_size=crop_size).transforms
    test_transform = ISIC2019_Augmentations(is_training=False, image_size=image_size, input_size=crop_size).transforms
    train_dataset = ISIC2019(df=train_df, root_dir=predefined_root_dir, transform=train_transform)
    if use_val:
        val_dataset = ISIC2019V2(df=val_df, root_dir=predefined_root_dir, transform=test_transform)
        val_dataset.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        val_dataset = ISIC2019(df=val_df, root_dir=predefined_root_dir, transform=test_transform)
    if use_test:
        test_dataset = ISIC2019V2(df=test_df, root_dir=predefined_root_dir, transform=test_transform)
        test_dataset.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        test_dataset = ISIC2019(df=test_df, root_dir=predefined_root_dir, transform=test_transform)
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, **kwargs)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs if not use_val else {})
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs if not use_test else {})

    return train_dataloader, val_dataloader, test_dataloader, train_df

def isic2019_dataloader_score_v3(batch_size, workers=8, predefined_root_dir='ISIC_2019_train/ISIC_2019_Training_Input', csv_file_name='ISIC_2019_train/ISIC_2019_Training_Metadata.csv', device="cpu", enable_train=True, enable_val=True, enable_test=True):
    df = read_isic2019_dataset_metainfo_raw(csv_file_name=csv_file_name)
    train_df, val_df, test_df = isic2019_holdout_score_v2(df)
    image_size = 256 // 2
    crop_size = 224 // 2
    train_transform = ISIC2019_Augmentations_v2(is_training=True, image_size=image_size, input_size=crop_size).transforms
    test_transform = ISIC2019_Augmentations_v2(is_training=False, image_size=image_size, input_size=crop_size).transforms
    if enable_train:
        train_dataset = ISIC2019V2(df=train_df, root_dir=predefined_root_dir, transform=train_transform)
        train_dataset.to(device)
    else:
        train_dataset = ISIC2019(df=train_df, root_dir=predefined_root_dir, transform=train_transform)
    if enable_val:
        val_dataset = ISIC2019V2(df=val_df, root_dir=predefined_root_dir, transform=test_transform)
        val_dataset.to(device)
    else:
        val_dataset = ISIC2019(df=val_df, root_dir=predefined_root_dir, transform=test_transform)
    if enable_test:
        test_dataset = ISIC2019V2(df=test_df, root_dir=predefined_root_dir, transform=test_transform)
        test_dataset.to(device)
    else:
        test_dataset = ISIC2019(df=test_df, root_dir=predefined_root_dir, transform=test_transform)
    kwargs = {'num_workers': workers, 'pin_memory': True} if device != "cpu" else {}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, **kwargs if not enable_train else {})
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs if not enable_val else {})
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs if not enable_test else {})

    return train_dataloader, val_dataloader, test_dataloader, train_df

def isic2019_pruning_examples(predefined_root_dir="pruning_examples/isic2019", csv_file_name="pruning_examples/isic2019.csv", minimum=False):
    df = read_isic2019_dataset_metainfo_raw(csv_file_name=csv_file_name)
    image_size = 256 // 2
    crop_size = 224 // 2
    test_transform = ISIC2019_Augmentations_v2(is_training=False, image_size=image_size, input_size=crop_size).transforms

    images = []
    labels = []
    for idx in tqdm(range(len(df))):
        img_name = os.path.join(predefined_root_dir, df.loc[df.index[idx], "image"] + ".jpg")
        image = io.imread(img_name)
        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)
        image = test_transform(image)
        images.append(image)
        labels.append(df.loc[df.index[idx], "class_id"])
        if minimum:
            break

    return images, labels

class ISIC2019_Augmentations():
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
            # mdlParams['full_color_distort'] = True
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

class ISIC2019_Augmentations_v2():
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
            # mdlParams['full_color_distort'] = True
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
                # transforms.Resize(self.image_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(np.float32(mdlParams['setMean']),np.float32(mdlParams['setStd']))]
        composed = transforms.Compose(all_transforms)
        return composed

    def get_train_augmentations(self, mdlParams):
        all_transforms = [transforms.ToPILImage()]
        # Normal train proc
        if self.same_sized_crop:
            # all_transforms.append(transforms.Resize(self.image_size))
            all_transforms.append(transforms.RandomCrop(self.input_size))
        elif self.only_downsmaple:
            raise NotImplementedError
            # all_transforms.append(transforms.Resize(self.input_size))
        else:
            raise NotImplementedError
            # all_transforms.append(transforms.RandomResizedCrop(self.input_size[0],scale=(mdlParams.get('scale_min',0.08),1.0)))
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

if __name__ == "__main__":
    # df = read_isic2019_dataset_metainfo_raw(csv_file_name='img_align_isic2019/list_attr_isic2019_modify.txt')
    # for index, row in df.iterrows():
    #     try:
    #         os.system("wget " + row["image_path"] + " -O " + os.path.join("img_align_isic2019", "img_align_isic2019", row["hasher"] + ".jpg"))
    #     except:
    #         print("FAIL:", row["hasher"])
    isic2019_dataloader_score(8, 8)
