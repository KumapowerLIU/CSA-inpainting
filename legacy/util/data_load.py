import random
import torch
from PIL import Image
from glob import glob


class Data_load(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform):
        super(Data_load, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform


        self.paths = glob('{:s}/*'.format(img_root),
                              recursive=True)


        self.mask_paths = glob('{:s}/*.png'.format(mask_root))

        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img , mask

    def __len__(self):
        return len(self.paths)
