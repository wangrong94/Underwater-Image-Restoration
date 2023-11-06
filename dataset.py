import torch.utils.data as data
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def get_patch(img_in, img_tar, patch_size, scale=1, ix=-1, iy=-1):
    (ih, iw) = img_in.size

    patch_mult = scale
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, info_patch
# def get_patch(img_in, img_tar, patch_size, scale=1, ix=-1, iy=-1):
#     (ih, iw) = img_in.size
#     patch_mult = scale
#     if ih > 256 and iw > 256:
#         tp = patch_mult * patch_size
#         ip = tp // scale
#
#         if ix == -1:
#             ix = random.randrange(0, iw - ip + 1)
#         if iy == -1:
#             iy = random.randrange(0, ih - ip + 1)
#
#         (tx, ty) = (scale * ix, scale * iy)
#
#         img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
#         img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))
#
#         info_patch = {
#             'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}
#     else:
#         tp = patch_mult * patch_size // 2
#         ip = tp // scale
#
#         if ix == -1:
#             ix = random.randrange(0, iw - ip + 1)
#         if iy == -1:
#             iy = random.randrange(0, ih - ip + 1)
#
#         (tx, ty) = (scale * ix, scale * iy)
#
#         img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
#         img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))
#
#         info_patch = {
#             'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}
#
#     return img_in, img_tar, info_patch


def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, info_aug


def re_size(img_in, img_tar, res_w=128, res_h=128):
    res = (res_w, res_h)
    img_in = img_in.resize(res)
    img_tar = img_tar.resize(res)
    info_res = {'resize_w': res_w, 'resize_h': res_h}
    return img_in, img_tar, info_res


class DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, label_dir, patch_size, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        label_filenames = [join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)]
        label_filenames.sort()
        self.label_filenames = label_filenames
        # self.im_resize = im_resize
        # self.res_w = res_w
        # self.res_h = res_h
        self.patch_size = patch_size
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        target = load_img(self.label_filenames[index])
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.label_filenames[index])
        input, target, _ = get_patch(input, target, self.patch_size)

        if self.data_augmentation:
            input, target, _ = augment(input, target)

        # if self.im_resize:
        #     input, target, _ = re_size(input, target, self.res_w, self.res_h)

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        return input, target, file

    def __len__(self):
        return len(self.label_filenames)


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        label_filenames = [join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)]
        label_filenames.sort()
        self.label_filenames = label_filenames
        self.transform = transform

    def __getitem__(self, index):
        target = load_img(self.label_filenames[index])
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.label_filenames[index])

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        return input, target, file

    def __len__(self):
        return len(self.label_filenames)
