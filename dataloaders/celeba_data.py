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


def read_celeba_dataset_metainfo_raw(csv_file_name='img_align_celeba/list_attr_celeba_modify.txt', valid_columns=["Image_Id", "Male", "Young", "Attractive", "Big_Nose", "Bags_Under_Eyes", "Mouth_Slightly_Open"]):
    df = pd.read_csv(csv_file_name, sep="\\s+")
    for idx, _ in df.items():
        if idx not in valid_columns:
            df = df.drop(idx, axis=1)
    # df.to_csv("img_align_celeba/debug0.csv")
    for idx in valid_columns:
        if idx == "Image_Id":
            continue
        df.loc[df[idx] == -1, idx] = 0
    df["Big_Nose_And_Bags_Under_Eyes"] = df["Big_Nose"] * 2 + df["Bags_Under_Eyes"]
    df["Attractive_And_Mouth_Slightly_Open"] = df["Attractive"] * 2 + df["Mouth_Slightly_Open"]
    # df = df.drop("Mouth_Slightly_Open", axis=1)
    # df.to_csv("img_align_celeba/debug1.csv")
    return df

def read_celeba_dataset_metainfo(csv_file_name):
    df = pd.read_csv(csv_file_name)
    return df

def celeba_holdout_score(df, fair_type, ctype):
    try:
        train_and_score = read_celeba_dataset_metainfo(csv_file_name="img_align_celeba/train_score.csv")
        val = read_celeba_dataset_metainfo(csv_file_name="img_align_celeba/all_val.csv")
        test = read_celeba_dataset_metainfo(csv_file_name="img_align_celeba/all_test.csv")
    except:
        train_and_score, val_and_test, _, _ = train_test_split(
                                        df,
                                        eval("df." + ctype),
                                        test_size=0.2,
                                        random_state=4242,
                                        stratify=eval("df." + ctype))
        val, test, _, _ = train_test_split(
                                        val_and_test,
                                        eval("val_and_test." + ctype),
                                        test_size=0.5,
                                        random_state=4242,
                                        stratify=eval("val_and_test." + ctype))
        val.to_csv("img_align_celeba/all_val.csv")
        test.to_csv("img_align_celeba/all_test.csv")
        train_and_score.to_csv("img_align_celeba/train_score.csv")
        # print("[DEBUG]", "Train_and_score", train_and_score.shape[0])

    score_size = 10130 # 162079 / 0.8 * 0.05 = 10129.9375

    try:
        male_data = read_celeba_dataset_metainfo(csv_file_name="img_align_celeba/male_score.csv")
        female_data = read_celeba_dataset_metainfo(csv_file_name="img_align_celeba/female_score.csv")
    except:
        male_data = train_and_score
        female_data = train_and_score
        male_data = male_data.drop(male_data[male_data["Male"] != 1].index)
        # print("[DEBUG]", "Male", male_data.shape[0])
        male_data = male_data.sample(n=score_size)
        male_data.to_csv("img_align_celeba/male_score.csv")
        female_data = female_data.drop(female_data[female_data["Male"] != 0].index)
        # print("[DEBUG]", "Female", female_data.shape[0])
        female_data = female_data.sample(n=score_size)
        female_data.to_csv("img_align_celeba/female_score.csv")

    try:
        young_data = read_celeba_dataset_metainfo(csv_file_name="img_align_celeba/young_score.csv")
        old_data = read_celeba_dataset_metainfo(csv_file_name="img_align_celeba/old_score.csv")
    except:
        young_data = train_and_score
        old_data = train_and_score
        young_data = young_data.drop(young_data[young_data["Young"] != 1].index)
        # print("[DEBUG]", "Young", young_data.shape[0])
        young_data = young_data.sample(n=10130)
        young_data.to_csv("img_align_celeba/young_score.csv")
        old_data = old_data.drop(old_data[old_data["Young"] != 0].index)
        # print("[DEBUG]", "Old", old_data.shape[0])
        old_data = old_data.sample(n=10130)
        old_data.to_csv("img_align_celeba/old_score.csv")

    if fair_type == "Male":
        major_data = male_data
        minor_data = female_data
    elif fair_type == "Young":
        major_data = young_data
        minor_data = old_data
    else:
        raise NotImplementedError
    train_and_score[ctype].value_counts().sort_index() # 80% (162079)
    val[ctype].value_counts().sort_index() # 10%
    test[ctype].value_counts().sort_index() # 10%
    major_data[ctype].value_counts().sort_index() # ~5% (10130)
    minor_data[ctype].value_counts().sort_index() # ~5% (10130)

    return train_and_score, major_data, minor_data, val, test

def celeba_holdout_score_v2(df, ctype):
    try:
        train_and_score = read_celeba_dataset_metainfo(csv_file_name="img_align_celeba/train_score.csv")
        val = read_celeba_dataset_metainfo(csv_file_name="img_align_celeba/all_val.csv")
        test = read_celeba_dataset_metainfo(csv_file_name="img_align_celeba/all_test.csv")
    except:
        train_and_score, val_and_test, _, _ = train_test_split(
                                        df,
                                        eval("df." + ctype),
                                        test_size=0.2,
                                        random_state=4242,
                                        stratify=eval("df." + ctype))
        val, test, _, _ = train_test_split(
                                        val_and_test,
                                        eval("val_and_test." + ctype),
                                        test_size=0.5,
                                        random_state=4242,
                                        stratify=eval("val_and_test." + ctype))
        val.to_csv("img_align_celeba/all_val.csv")
        test.to_csv("img_align_celeba/all_test.csv")
        train_and_score.to_csv("img_align_celeba/train_score.csv")
        # print("[DEBUG]", "Train_and_score", train_and_score.shape[0])

    train_and_score[ctype].value_counts().sort_index() # 80% (162079)
    val[ctype].value_counts().sort_index() # 10%
    test[ctype].value_counts().sort_index() # 10%

    return train_and_score, val, test

def get_weighted_sampler(df, label_level):
    class_sample_count = np.array(df[label_level].value_counts().sort_index())
    class_weight = 1. / class_sample_count
    samples_weight = np.array([class_weight[t] for t in df[label_level]])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)
    return sampler

class CelebA(Dataset):

    def __init__(self, fair_attr, y_attr, df=None, root_dir=None, transform=None, s=False):
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
        self.fair_attr = fair_attr
        self.y_attr = y_attr
        for idx, _ in self.df.items():
            if idx != "Image_Id" and idx != self.fair_attr and idx != self.y_attr:
                self.df = self.df.drop(idx, axis=1)
        self.root_dir = root_dir
        self.transform = transform
        self.s = s

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.df.loc[self.df.index[idx], "Image_Id"])
        image = io.imread(img_name)
        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)

        fair_attr = self.df.loc[self.df.index[idx], self.fair_attr]
        y_attr = self.df.loc[self.df.index[idx], self.y_attr]
        if self.transform:
            image = self.transform(image)
        sample = {
                    'image': image,
                    'fair_attr': fair_attr,
                    'y_attr': y_attr,
                }
        if self.s:
            sample['img'] = self.df.loc[self.df.index[idx], "Image_Id"]
        # sample = (image, int(val))
        return sample

class CelebAV2(Dataset):

    def __init__(self, fair_attr, y_attr, df=None, root_dir=None, transform=None, s=False):
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
        self.fair_attr = fair_attr
        self.y_attr = y_attr
        for idx, _ in self.df.items():
            if idx != "Image_Id" and idx != self.fair_attr and idx != self.y_attr:
                self.df = self.df.drop(idx, axis=1)
        self.root_dir = root_dir
        self.transform = transform
        self.s = s

        self.images = []
        for idx in tqdm(range(len(self.df))):
            img_name = os.path.join(self.root_dir, self.df.loc[self.df.index[idx], "Image_Id"])
            image = io.imread(img_name)
            if(len(image.shape) < 3):
                image = skimage.color.gray2rgb(image)
            if self.transform:
                image = self.transform(image)
            self.images.append(image)
        self.images = stack(self.images)

    def to(self, device):
        self.images = self.images.to(device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.images[idx]
        fair_attr = self.df.loc[self.df.index[idx], self.fair_attr]
        y_attr = self.df.loc[self.df.index[idx], self.y_attr]
        sample = {
                    'image': image,
                    'fair_attr': fair_attr,
                    'y_attr': y_attr,
                }
        if self.s:
            sample['img'] = self.df.loc[self.df.index[idx], "Image_Id"]
        # sample = (image, int(val))
        return sample

class CelebA_Augmentations():
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
            # # mdlParams['full_color_distort'] = True
            # mdlParams['autoaugment'] = True     
            # mdlParams['flip_lr_ud'] = True
            # mdlParams['full_rot'] = 180
            # mdlParams['scale'] = (0.8,1.2)
            # mdlParams['shear'] = 10
            # mdlParams['cutout'] = 16

            # For PAD-UFES-20 dataset
            # mdlParams['autoaugment'] = False     
            # mdlParams['flip_lr_ud'] = True
            # mdlParams['full_rot'] = 180
            # mdlParams['scale'] = (0.9,1.1)
            # mdlParams['shear'] = 10
            # mdlParams['cutout'] = 10
            
            # For celebA dataset
            mdlParams['autoaugment'] = False     
            mdlParams['flip_lr_ud'] = True
            mdlParams['full_rot'] = 180
            mdlParams['scale'] = (0.95,1.05)
            mdlParams['shear'] = 5
            mdlParams['cutout'] = 10

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

def celeba_dataloader_score(batch_size, workers, predefined_root_dir='img_align_celeba/img_align_celeba', csv_file_name='img_align_celeba/list_attr_celeba_modify.txt', fair_type="Young", ctype="Attractive"):
    df = read_celeba_dataset_metainfo_raw(csv_file_name=csv_file_name)
    train_df, major_score_df, minor_score_df, val_df, test_df = celeba_holdout_score(df, fair_type, ctype)
    image_size = 256 // 2
    crop_size = 224 // 2
    train_transform = CelebA_Augmentations(is_training=True, image_size=image_size, input_size=crop_size).transforms
    test_transform = CelebA_Augmentations(is_training=False, image_size=image_size, input_size=crop_size).transforms
    train_dataset = CelebA(df=train_df, fair_attr=fair_type, y_attr=ctype, root_dir=predefined_root_dir, transform=train_transform)
    major_score_dataset = CelebA(df=major_score_df, fair_attr=fair_type, y_attr=ctype, root_dir=predefined_root_dir, transform=test_transform)
    minor_score_dataset = CelebA(df=minor_score_df, fair_attr=fair_type, y_attr=ctype, root_dir=predefined_root_dir, transform=test_transform)
    val_dataset = CelebA(df=val_df, fair_attr=fair_type, y_attr=ctype, root_dir=predefined_root_dir, transform=test_transform)
    test_dataset = CelebA(df=test_df, fair_attr=fair_type, y_attr=ctype, root_dir=predefined_root_dir, transform=test_transform)
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, **kwargs)
    major_score_dataloader = torch.utils.data.DataLoader(major_score_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    minor_score_dataloader = torch.utils.data.DataLoader(minor_score_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_dataloader, major_score_dataloader, minor_score_dataloader, val_dataloader, test_dataloader

def celeba_dataloader_score_v2(batch_size, workers, predefined_root_dir='img_align_celeba/img_align_celeba', csv_file_name='img_align_celeba/list_attr_celeba_modify.txt', fair_type="Young", ctype="Attractive"):
    df = read_celeba_dataset_metainfo_raw(csv_file_name=csv_file_name)
    train_df, val_df, test_df = celeba_holdout_score_v2(df, ctype)
    image_size = 256 // 2
    crop_size = 224 // 2
    train_transform = CelebA_Augmentations(is_training=True, image_size=image_size, input_size=crop_size).transforms
    test_transform = CelebA_Augmentations(is_training=False, image_size=image_size, input_size=crop_size).transforms
    train_dataset = CelebA(df=train_df, fair_attr=fair_type, y_attr=ctype, root_dir=predefined_root_dir, transform=train_transform)
    val_dataset = CelebA(df=val_df, fair_attr=fair_type, y_attr=ctype, root_dir=predefined_root_dir, transform=test_transform)
    test_dataset = CelebA(df=test_df, fair_attr=fair_type, y_attr=ctype, root_dir=predefined_root_dir, transform=test_transform)
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, **kwargs)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_dataloader, val_dataloader, test_dataloader, train_df

def celeba_dataloader_score_v2s(batch_size, workers, predefined_root_dir='img_align_celeba/img_align_celeba', csv_file_name='img_align_celeba/list_attr_celeba_modify.txt', fair_type="Young", ctype="Attractive"):
    df = read_celeba_dataset_metainfo_raw(csv_file_name=csv_file_name)
    train_df, val_df, _ = celeba_holdout_score_v2(df, ctype)
    image_size = 256 // 2
    crop_size = 224 // 2
    train_transform = CelebA_Augmentations(is_training=True, image_size=image_size, input_size=crop_size).transforms
    test_transform = CelebA_Augmentations(is_training=False, image_size=image_size, input_size=crop_size).transforms
    train_dataset = CelebA(df=train_df, fair_attr=fair_type, y_attr=ctype, root_dir=predefined_root_dir, transform=train_transform, s=True)
    val_dataset = CelebA(df=val_df, fair_attr=fair_type, y_attr=ctype, root_dir=predefined_root_dir, transform=test_transform, s=True)
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, **kwargs)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_dataloader, val_dataloader, df

def celeba_test_deploy(batch_size, predefined_root_dir='img_align_celeba/img_align_celeba', csv_file_name='img_align_celeba/selected.csv', fair_type="Male", ctype="Bags_Under_Eyes"):
    df = read_celeba_dataset_metainfo(csv_file_name=csv_file_name)
    image_size = 256 // 2
    crop_size = 224 // 2
    test_transform = CelebA_Augmentations(is_training=False, image_size=image_size, input_size=crop_size).transforms
    test_dataset = CelebA(df=df, fair_attr=fair_type, y_attr=ctype, root_dir=predefined_root_dir, transform=test_transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_dataloader

def celeba_pruning_examples(predefined_root_dir="pruning_examples/celeba_Male_Big_Nose", csv_file_name="pruning_examples/celeba_Male_Big_Nose.csv", minimum=False, ctype="Big_Nose"):
    df = read_celeba_dataset_metainfo(csv_file_name=csv_file_name)
    image_size = 256 // 2
    crop_size = 224 // 2
    test_transform = CelebA_Augmentations(is_training=False, image_size=image_size, input_size=crop_size).transforms

    images = []
    labels = []
    for idx in tqdm(range(len(df))):
        img_name = os.path.join(predefined_root_dir, df.loc[df.index[idx], "Image_Id"])
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
    # df = read_celeba_dataset_metainfo_raw(csv_file_name='img_align_celeba/list_attr_celeba_modify.txt')
    # for index, row in df.iterrows():
    #     try:
    #         os.system("wget " + row["image_path"] + " -O " + os.path.join("img_align_celeba", "img_align_celeba", row["hasher"] + ".jpg"))
    #     except:
    #         print("FAIL:", row["hasher"])
    celeba_dataloader_score(8, 8)
