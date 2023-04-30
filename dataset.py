from PIL import Image
import numpy as np
import os
import config
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import clip
import torchvision.transforms as transforms
import torch.nn.functional as F


class DeblurDataset(Dataset):
    def __init__(self, raw_dir, target_dir, text_file_path):
        self.raw_dir = raw_dir
        self.target_dir = target_dir
        self.text_file_path = text_file_path

        train_raw_path = 'datasets/raw/train'
        train_target_path = 'datasets/sharp/train'
        train_text_path = 'datasets/text/sharp.txt'

        # Due to the data structure of the given dataset, we input all image names from the 'ld' folder
        # When loading, we'll look up the corresponding file in 'hd' folder
        self.raw_files_list = os.listdir(self.raw_dir)
        # self.target_files_list = os.listdir(self.target_dir)
        with open(self.text_file_path, 'r') as file:
            self.text = file.read().strip()

    # we need to implement __len__() and __getitem__() methods to use torch.utils.data.DataLoader later,
    # for train and test purpose
    def __len__(self):
        return len(self.raw_files_list)

    def __getitem__(self, index):
        img_file = self.raw_files_list[index]

        inp_img_path = os.path.join(self.raw_dir, img_file)
        tar_img_path = os.path.join(self.target_dir, img_file)
        # Load the input image
        input_image = np.array(Image.open(inp_img_path))

        # Crop out the margins
        input_image = input_image[config.MARGIN_WIDTH:-config.MARGIN_WIDTH, config.MARGIN_WIDTH:-config.MARGIN_WIDTH, :]

        try:
            # For test data, we won't have target images and hence need to handle the error accordingly
            target_image = np.array(Image.open(tar_img_path))
            target_image = target_image[config.MARGIN_WIDTH:-config.MARGIN_WIDTH,
                           config.MARGIN_WIDTH:-config.MARGIN_WIDTH, :]
        except FileNotFoundError:
            target_image = input_image

        # Normalize, resize and convert both images to tensor (channels first)
        clip_img = np.copy(input_image)
        input_image = config.resize_and_normalize(image=input_image)["image"]
        target_image = config.resize_and_normalize(image=target_image)["image"]
        clip_img = config.clip_transform(image=clip_img)["image"]

        return input_image, target_image, self.text, clip_img


def test():

    def normalize_0_1(tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        return (tensor - min_val) / (max_val - min_val)

    root_dir = "./datasets/Sharpen/train"
    text_dir = "./datasets/Sharpen/train/edit.txt"
    data = DeblurDataset(root_dir, text_dir)
    loader = DataLoader(data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    # print(len(data))  # -- should print num of train images --
    # # 获取第一个样本
    # input_image, target_image = data[0]
    #
    # # 输出第一个样本的形状
    # print("Input image shape: ", input_image.shape)
    # print("Target image shape: ", target_image.shape)

    # 获取一个迭代器
    iterator = iter(loader)

    # 获取第一个批次的数据
    batch = next(iterator)

    # 检查批次中第一个元素的形状
    input_image, target_image, text, clip_img= batch
    # print("Input image shape: ", input_image.shape)
    # # print(input_image[0])
    # print("Target image shape: ", target_image.shape)
    # # print(target_image[0])
    # # print(target_image[0])
    # # print("Text: ", text)

    # 1. 加载预训练的 CLIP 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 4. 准备文本输入
    text_input = clip.tokenize([t for t in text]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(clip_img)
        text_features = model.encode_text(text_input)

    print("Input image shape: ", input_image.shape)
    print("CLIP image shape: ", clip_img.shape)
    print("image feature dimension:", image_features.shape)
    print("text feature dimension:", text_features.shape)

    print(input_image[0])
    print(image_features[0])
    print(text_features[0])

    img_feature_normalized = normalize_0_1(image_features)
    text_feature_normalized = normalize_0_1(text_features)

    print(img_feature_normalized[0])
    print(text_feature_normalized[0])

    # img_edit = image_features.view(image_features.size(0), 512, 1, 1)
    # print("image feature dimension:", img_edit.shape)


if __name__ == '__main__':
    test()
