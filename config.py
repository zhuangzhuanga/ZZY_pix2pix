# contains all h-params of the Pix2Pix model + all the constants used across the scripts,
# and also image-procesing functions
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Universal Variables + h-params
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TRAIN_DIR = "./datasets/Sharpen/train/"
# VAL_DIR = "./datasets/Sharpen/val/"
# TEST_DIR = "./datasets/Sharpen/test/"
LEARNING_RATE = 3e-4
BATCH_SIZE = 128
NUM_WORKERS = 2
NUM_RUNS = 10
IMAGE_SIZE = 256
CHANNELS_IMG = 3
MARGIN_WIDTH = 10  # 10% of 100
L1_LAMBDA = 100
NUM_EPOCHS = 300
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_DISC = "Trained-Model/disc.pth.tar"
CHECKPOINT_GEN = "Trained-Model/gen.pth.tar"

# Image processing
# Resizes input image to fit generator input, normalizes values,
# and converts numpy array to tensor object (channels first)
resize_and_normalize = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.481, 0.456, 0.406], std=[0.268, 0.245, 0.263], max_pixel_value=255.0, ),
        ToTensorV2(),
    ]
)

clip_transform = A.Compose(
    [
        A.Resize(width=224, height=224),
        A.Normalize(mean=[0.481, 0.456, 0.406], std=[0.268, 0.245, 0.263], max_pixel_value=255.0, ),
        ToTensorV2(),
    ]
)


