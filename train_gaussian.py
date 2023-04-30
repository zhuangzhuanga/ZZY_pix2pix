import torch
import torch.nn as nn
import torch.optim as optim
from utils import save_checkpoint, load_checkpoint, save_some_examples
import config
from dataset import DeblurDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from ignite.metrics import FID, InceptionScore
from torchvision.models import inception_v3

import numpy as np
import clip

from torch.optim.lr_scheduler import StepLR


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1, bce, g_scalar, d_scalar, epoch, clip_model):
    writer = SummaryWriter(log_dir='./23_log/gaussian/logs_gaussian_train')
    # bce g_scalar d_scalar不确定是什么玩意
    loop = tqdm(loader, leave=True)

    # is_metric = InceptionScore()
    # fid = FID()

    # tqdm is optional
    # we can just enumerate over loader but then the console output will be a bit messier
    for idx, (raw_img, target_img, text, clip_img) in enumerate(loop):
        raw_img, target_img, clip_img = raw_img.to(config.DEVICE), \
                                        target_img.to(config.DEVICE), \
                                        clip_img.to(config.DEVICE)

        text_input = clip.tokenize([t for t in text]).to(config.DEVICE)

        # calculate image and text feature using CLIP
        with torch.no_grad():
            image_features = clip_model.encode_image(clip_img)
            text_features = clip_model.encode_text(text_input)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(raw_img, image_features, text_features)

            D_real = disc(raw_img, target_img)
            D_fake = disc(raw_img, y_fake.detach())
            D_real_loss = bce(D_real, torch.ones_like(D_real))  # 这个牛逼
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        opt_disc.zero_grad()
        d_scalar.scale(D_loss).backward()
        d_scalar.step(opt_disc)
        d_scalar.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            D_fake = disc(raw_img, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1(y_fake, target_img) * config.L1_LAMBDA  # L1 distance
            G_loss = G_fake_loss + L1

        # # is_metric.update(y_fake.detach()
        # is_metric.update(y_fake.float().detach())
        # # fid.update((y_fake.detach(), target_img.detach()))
        # fid.update((y_fake.float().detach(), target_img.float().detach()))

        opt_gen.zero_grad()
        g_scalar.scale(G_loss).backward()
        g_scalar.step(opt_gen)
        g_scalar.update()
        writer.add_scalar(tag='train_fakeloss', scalar_value=D_fake_loss, global_step=epoch)
        writer.add_scalar(tag='train_realloss', scalar_value=D_real_loss, global_step=epoch)
        writer.add_scalar(tag='train_G_total_loss', scalar_value=G_loss, global_step=epoch)
        writer.add_scalar(tag='train_G_partial_loss', scalar_value=G_loss, global_step=epoch)

    # with torch.cuda.amp.autocast():
    #     is_mean = is_metric.compute()
    #     fid_value = fid.compute()

    # # 将IS和FID添加到tensorboard
    # writer.add_scalar(tag='val_InceptionScore', scalar_value=is_mean, global_step=epoch)
    # writer.add_scalar(tag='val_FID', scalar_value=fid_value, global_step=epoch)


def eval_fn(gen, loader, epoch, clip_model):
    val_writer = SummaryWriter(log_dir='./23_log/gaussian/logs_gaussian_test')
    gen.eval()
    loop = tqdm(loader, leave=True)

    is_metric = InceptionScore()
    fid = FID()

    with torch.no_grad():
        for idx, (raw_img, target_img, text, clip_img) in enumerate(loop):
            raw_img, target_img, clip_img = raw_img.to(config.DEVICE), \
                                            target_img.to(config.DEVICE), \
                                            clip_img.to(config.DEVICE)

            text_input = clip.tokenize([t for t in text]).to(config.DEVICE)

            # calculate image and text feature using CLIP
            image_features = clip_model.encode_image(clip_img)
            text_features = clip_model.encode_text(text_input)

            with torch.cuda.amp.autocast():
                y_fake = gen(raw_img, image_features, text_features)

            # is_metric.update(y_fake)
            # fid.update((y_fake, target_img))
            is_metric.update(y_fake.float().detach())
            fid.update((y_fake.float().detach(), target_img.float().detach()))

    with torch.cuda.amp.autocast():
        is_mean = is_metric.compute()
        fid_value = fid.compute()

    # 将IS和FID添加到tensorboard
    val_writer.add_scalar(tag='test_InceptionScore', scalar_value=is_mean, global_step=epoch)
    val_writer.add_scalar(tag='test_FID', scalar_value=fid_value, global_step=epoch)

    gen.train()


def main():
    # load pre-trained CLIP
    clip_model, _ = clip.load("ViT-B/32", device=config.DEVICE)

    # initialize the discriminator, generator and their optimizers
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    # initialize learning rate scheduler
    scheduler_gen = StepLR(opt_gen, step_size=50, gamma=0.5)
    scheduler_disc = StepLR(opt_disc, step_size=50, gamma=0.5)

    # initialize loss objects
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    # load previously trained model
    # to prevent this, set 'LOAD_MODEL' in congif file to 'False'
    # if config.LOAD_MODEL:
    #     load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
    #     load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    # Load the training and validation data
    train_raw_path = 'datasets/raw/train'
    train_target_path = 'datasets/gaussian/train'
    train_text_path = 'datasets/text/gaussian.txt'
    train_dataset = DeblurDataset(train_raw_path, train_target_path, train_text_path)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    g_scalar = torch.cuda.amp.GradScaler()
    d_scalar = torch.cuda.amp.GradScaler()

    val_raw_path = 'datasets/raw/test'
    val_target_path = 'datasets/gaussian/test'
    val_text_path = 'datasets/text/gaussian.txt'
    val_dataset = DeblurDataset(val_raw_path, val_target_path, val_text_path)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # training loop
    for epoch in range(config.NUM_EPOCHS):
        print("\nEpoch - {}".format(epoch + 1))
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scalar, d_scalar, epoch, clip_model)
        eval_fn(gen, val_loader, epoch, clip_model)

        # update learning rate
        scheduler_gen.step()
        scheduler_disc.step()

        # save model every 5 epochs
        if config.SAVE_MODEL and epoch % 20 == 0 and epoch > 0:
            filename_gen = '23_model/gaussian/gaussian_gen_{}.pt'.format(epoch)
            filename_disc = '23_model/gaussian/gaussian_dis_{}.pt'.format(epoch)
            save_checkpoint(gen, opt_gen, filename_gen)
            save_checkpoint(disc, opt_disc, filename_disc)
        #
        # # generate output on validation data
        # save_some_examples(gen, val_loader, epoch, folder="evaluation/gaussianen")

    save_checkpoint(gen, opt_gen, '23_model/final/gaussian_generator.pt')
    save_checkpoint(disc, opt_disc, '23_model/final/gaussian_discriminator.pt')


if __name__ == "__main__":
    main()
