import torch

BATCH_SIZE = 32
NUM_RES_BLOCKS = 3
HIDDEN_DIM = 4
LR = 0.0002
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_aug = dict(
    transform=dict(
        __class_fullname__='albumentations.core.composition.Compose',
        bbox_params=None,
        keypoint_params=None,
        p=1,
        transforms=[
            dict(
                __class_fullname__='albumentations.augmentations.transforms.HorizontalFlip',
                p=0.5,
            ),
            dict(
                __class_fullname__='albumentations.augmentations.transforms.VerticalFlip',
                p=0.5,
            ),
            dict(
                __class_fullname__='albumentations.pytorch.transforms.ToTensorV2',
                always_apply=True,
                p=1,
            )
        ]
    )
)


test_aug = dict(
    transform=dict(
        __class_fullname__='albumentations.core.composition.Compose',
        bbox_params=None,
        keypoint_params=None,
        p=1,
        transforms=[
            dict(
                __class_fullname__='albumentations.pytorch.transforms.ToTensorV2',
                always_apply=True,
                p=1,
            )
        ]
    )
)