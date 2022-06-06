import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import albumentations as A
import numpy as np
import os
from PIL import Image
from sklearn.utils import compute_class_weight

val_transform = A.Compose(  # validation image transforms
    [A.Resize(256, 256)]
)

def get_loaders(
    train_dir,
    train_mask_dir,
    val_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=0,
    pin_memory=True,
):
    """
    This method creates the dataloader objects for the training loops

    :train_dir: directory of training images
    :train_mask_dir: directory of training masks
    :val_dir: validation image directory

    :returns: training and validation dataloaders
    recall
    """
    train_ds = Dataset(train_dir,
                       train_mask_dir,
                       transform=train_transform
                       )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = Dataset(
        image_dir=val_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


class Dataset(Dataset):
    """This method creates the dataset from given directories"""

    def __init__(self, image_dir, mask_dir, transform=None):
        """initialize directories

        :image_dir: TODO
        :mask_dir: TODO
        :transform: TODO

        """
        self._image_dir = image_dir
        self._mask_dir = mask_dir
        self._transform = transform
        self.images = os.listdir(image_dir)
        self.mapping = {(0, 0, 0): 0,  # background class (black)
                        (0, 0, 255): 1,  # 0 = class 1
                        (225, 0, 225): 2,  # 1 = class 2
                        (255, 0, 0): 3,  # 2 = class 3
                        (255, 225, 225): 4,  # 3 = class 4
                        (255, 255, 0): 5}  # 4 = class 5

    def __len__(self):
        """returns length of images
        :returns: TODO

        """
        return len(self.images)

    def mask_to_class_rgb(self, mask):
        h = 256  # ========================================================================================================================
        w = 256  # ========================================================================================================================
        mask = torch.from_numpy(mask)
        mask = torch.squeeze(mask)  # remove 1

        class_mask = mask
        class_mask = class_mask.permute(2, 0, 1).contiguous()
        h, w = class_mask.shape[1], class_mask.shape[2]
        mask_out = torch.zeros(h, w, dtype=torch.long)

        for k in self.mapping:
            idx = (class_mask == torch.tensor(
                k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(0) == 3)
            mask_out[validx] = torch.tensor(self.mapping[k], dtype=torch.long)

        return mask_out
    def __getitem__(self, index):
        """TODO: Docstring for __getitem__.
        :returns: TODO

        """
        img_path = os.path.join(self._image_dir, self.images[index])
        mask_path = os.path.join(self._mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))
        mask = self.mask_to_class_rgb(mask).cpu().detach().numpy()


        if self._transform is not None:
            augmentations = self._transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            

        return image, mask


y_pred = []
y_true = []

VAL_IMG_DIR = '/storage/hpc/27/thomann/coastal_segmentation_data/current_data/valannot3/'
VAL_MASK_DIR="/storage/hpc/27/thomann/coastal_segmentation_data/current_data/valannot3/"

valDL, valDL2 = get_loaders(VAL_IMG_DIR, VAL_MASK_DIR,
                            VAL_IMG_DIR, VAL_MASK_DIR,
                            1, val_transform,
                            val_transform, num_workers=1, pin_memory=True,)

for x, y in valDL:
    y = y.to("cuda").unsqueeze(1)
    # y = y.to(torch.int64)
    # ===========================================================================
    y_true.append(y)

y_true = torch.cat(y_true, dim=2)

fop = y_true.squeeze()
fop = torch.reshape(fop, (-1,)).cpu()
print(fop.shape)
print(np.unique(fop.numpy()))

class_weights = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(fop.numpy()),
                                        y=fop.numpy()                                                    
                                    )
class_weights = dict(zip(np.unique(fop), class_weights))

# class_weights=torch.tensor(class_weights,dtype=torch.float)
 
print(class_weights) #([1.0000, 1.0000, 4.0000, 1.0000, 0.5714])

# print(class_weights)
print("hi")