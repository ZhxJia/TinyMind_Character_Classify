import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageEnhance
import PIL.ImageOps
from sklearn.model_selection import train_test_split
import os.path as osp
from torchvision import datasets, transforms
import random
import yaml
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def gaussianNoisy(im, mean=0.2, sigma=0.3):
    for _i in range(len(im)):
        im[_i] += random.gauss(mean, sigma)
    return im


def RandomGaussian(image, mean=0.2, sigma=0.3):
    img = np.asarray(image)
    # img.flags.writeable = True
    width, height = img.shape[:2]
    img_n = gaussianNoisy(img.flatten(), mean, sigma)
    img = img_n.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def RandomColor(image):
    random_factor = np.random.randint(7, 18) / 10.
    brightness_image = ImageEnhance.Brightness(image).enhance(random_factor)  # 亮度
    random_factor = np.random.randint(8, 18) / 10.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 对比度
    return contrast_image


def imshow(imgs, labels=None):
    imgs_ = imgs.cpu().numpy()
    print(imgs.shape)
    plt.imshow(np.transpose(imgs_, (1, 2, 0)))
    plt.show()


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


class CharacterDateset(Dataset):

    def __init__(self, root_dir, type, data_dir='train', transform=transforms.ToTensor(), converted=False):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.transform = transform
        self.image_names = []
        self.labels = []
        self.converted = converted
        # get map of character labels and index
        self.label_names = []
        if os.path.exists("../datasets/label_map.yaml"):
            print("Load exist label map")
            self.labels_map = self._load_label_map()
            label_range = len(self.labels_map)
            for i in range(label_range):
                self.label_names.append(get_key(self.labels_map, i)[0])
        else:
            self.label_names = os.listdir(osp.join(self.root_dir, 'train'))
            self.labels_map = {label: index for index, label in enumerate(self.label_names)}
            self.save_label_map()
            print("Save label map: label_map.yaml Success ")

        print(f"mapping of labels and index:\n{self.labels_map.items()}")

        if self.data_dir == 'test2' and type == 'test':
            for img in os.listdir(osp.join(self.root_dir, self.data_dir)):
                self.image_names.append(osp.join(self.root_dir, self.data_dir, img))

        elif self.data_dir == 'train':
            images = []
            labels = []
            img_path = osp.join(self.root_dir, self.data_dir)
            for label in self.label_names:
                for img in os.listdir(osp.join(img_path, label)):
                    images.append(osp.join(img_path, label, img))
                    labels.append(label)
            labels = [self.labels_map[label] for label in labels]
            img_train, img_val, label_train, label_val = train_test_split(images, labels, test_size=0.01,
                                                                          random_state=0)
            if type == 'train':
                self.image_names = img_train
                self.labels = label_train

                # for i, l in zip(self.image_names, self.labels):
                #     print(f"img:{i},label:{l}\n")

            elif type == 'val':
                self.image_names = img_val
                self.labels = label_val

                # for i, l in zip(self.image_names, self.labels):
                #     print(f"img:{i},label:{l}\n")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.data_dir == 'test2':
            img_dir = self.image_names[idx]
            img = Image.open(img_dir).convert("L").resize((256, 256))

            img = self.transform(img)
            return img
        else:
            img_dir = self.image_names[idx]
            img = Image.open(img_dir).convert("L").resize((256, 256)).rotate(random.randint(-5, 5))
            if self.converted:
                img = PIL.ImageOps.invert(img)
            img = RandomColor(img)
            img = RandomGaussian(img, mean=0.2, sigma=0.3)
            img = self.transform(img)
            label = self.labels[idx]
            return img, label

    def get_label_dict(self):
        return self.labels_map

    def get_img_path_list(self):
        return self.image_names

    def save_label_map(self):
        fp = open('../datasets/label_map.yaml', 'w')
        fp.write(yaml.dump(self.labels_map))
        fp.close()

    def _load_label_map(self):
        fp = open('../datasets/label_map.yaml', 'r')
        st = fp.read()
        fp.close()
        return yaml.safe_load(st)

    def get_labelname(self):
        return self.label_names


if __name__ == "__main__":
    data_transform = transforms.Compose(
        [transforms.ToTensor(), ])

    train_datasets = CharacterDateset('../data', type='train')
    train_dataloader = DataLoader(train_datasets, batch_size=100, shuffle=True, num_workers=2)
    val_datasets = CharacterDateset('../data', type='val')
    val_dataloader = DataLoader(val_datasets, batch_size=4, shuffle=True, num_workers=2)
    # print(val_datasets.get_img_path_list())
    # train_datasets.save_label_map()
    # for i, (img, label) in enumerate(val_dataloader):
    #     print(f"{i}:{img}")
    labels_maps = val_datasets.get_labelname()
    dataiter = iter(val_dataloader)
    images, labels = dataiter.next()
    print(images.shape)
    imshow(make_grid(images))
    print(' '.join('%5s' % labels_maps[labels[j]] for j in range(4)))
    print(len(val_dataloader))

    im = Image.open("../data/test1_/5bf00c19cc06f66070a6aad27336b62e40f14fab.jpg").convert("L").resize((256, 256))
    # im.show(title="origin")
    im_1 = RandomColor(im)
    im_2 = RandomGaussian(im_1, mean=0.2, sigma=0.3)
    # im_2.show(title="convert")
    print(val_datasets.get_labelname())
