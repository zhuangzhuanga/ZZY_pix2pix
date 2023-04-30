import torch
from torch.utils.data import DataLoader
import numpy as np
from generator import Generator
from dataset import DeblurDataset
import config
import torch.optim as optim
import os
import clip
from utils import load_checkpoint
from torchvision.utils import save_image
from PIL import Image
import sys


def load_model(model_path):
    gen = Generator(in_channels=3).to(config.DEVICE)

    # optimizer isn't required but the load_checkpoint function loads it,
    # so it's easier to just initialize it as well
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    # Load the generator
    load_checkpoint(model_path, gen, opt_gen, config.LEARNING_RATE)

    gen.eval()
    return gen


def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    return (tensor * std) + mean


def main():
    # load pre-trained CLIP
    clip_model, _ = clip.load("ViT-B/32", device=config.DEVICE)

    # Load the generator model
    model_path = 'save_model/sharpen/Final_generator.pt'
    gen = load_model(model_path)

    # Load the test data
    test_img_path = 'datasets/Sharpen/train'
    test_text_path = 'datasets/Sharpen/train/edit.txt'
    test_dataset = DeblurDataset(test_img_path, test_text_path)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    # create the output directory if it doesn't exist
    output_dir = '23_first_test'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Test the model and save output images
    with torch.no_grad():
        for idx, (raw_img, target_img, text, clip_img) in enumerate(test_loader):
            raw_img, clip_img, target_img= raw_img.to(config.DEVICE), clip_img.to(config.DEVICE), target_img.to(config.DEVICE)
            text_input = clip.tokenize([t for t in text]).to(config.DEVICE)

            # calculate image and text feature using CLIP
            image_features = clip_model.encode_image(clip_img)
            text_features = clip_model.encode_text(text_input)

            with torch.cuda.amp.autocast():
                y_fake = gen(raw_img, image_features, text_features)

            # 取消归一化
            mean = [0.481, 0.456, 0.406]
            std = [0.268, 0.245, 0.263]
            y_fake_unnormalized = unnormalize(y_fake, mean, std)
            raw_img_unnormalized = unnormalize(raw_img, mean, std)
            target_img_unnormalized = unnormalize(target_img, mean, std)

            # 将其从GPU传输到CPU并将其从[0, 1]范围转换为[0, 255]范围
            # y_fake_unnormalized = (y_fake_unnormalized.cpu() * 255).clamp(0, 255).byte()

            # Save the images
            for i in range(y_fake_unnormalized.size(0)):
                img_y_fake = y_fake_unnormalized[i].unsqueeze(0)
                img_raw = raw_img_unnormalized[i].unsqueeze(0)
                img_target = target_img_unnormalized[i].unsqueeze(0)
                grid = torch.cat([img_y_fake, img_raw, img_target])
                save_image(grid, f"23_first_test/gridimage_{i}.png")

            # Only process one batch
            break


if __name__ == "__main__":
    main()
