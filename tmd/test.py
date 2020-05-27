import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torchvision import transforms
from datasets.characeter_dataset import CharacterDateset
from PIL import Image
from net import resnet34
from tqdm import tqdm

def load_image(image):
    img = Image.open(image).convert("L").resize((256, 256))
    img = transforms.ToTensor(img)
    img = img.reshape((1,) + img.shape + (1,))
    return img


def get_label_predict_top1(image, model):
    predict_proprely = model(image)
    predict_label = np.argmax(predict_proprely, axis=1)
    return predict_label


def get_label_predict_top_k(image, model, top_k):
    predict_proprely = model(image)
    predict_list = list(predict_proprely[0])
    min_label = min(predict_list)
    label_k = []
    for i in range(top_k):
        label = np.argmax(predict_list)
        predict_list.remove(predict_list[label])
        predict_list.insert(label, min_label)
        label_k.append(label)
    return label_k


def test_image_predict_top1(model, test_loader, label_list):
    predict_label = []
    nb = len(test_loader)
    pbar = tqdm(enumerate(test_loader), total=nb)
    for t, img in pbar:
        img = img.to(device)
        label_index = get_label_predict_top1(img, model)
        label = label_list[label_index]
        predict_label.append(label)
    return predict_label

def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]

def test_image_predict_topk(model, test_loader, label_dict):
    predict_label = []
    nb = len(test_loader)
    pbar = tqdm(enumerate(test_loader), total=nb)
    for t, img in pbar:
        img = img.to(device)
        label_index = get_label_predict_top_k(img, model, top_k=5)
        label_value_dict = []
        for idx in label_index:
            label_value = get_key(label_dict,idx)[0]
            label_value_dict.append(str(label_value))
        predict_label.append(label_value_dict)

    return predict_label


def tran_list2str(predict_list_label):
    str_labels = []
    for row in range(len(predict_list_label)):
        str = ""
        for label in predict_list_label[row]:
            str += label
        str_labels.append(str)
    return str_labels





def save_csv(test_images_path, predict_labels):
    save_arr = np.empty((16343, 2), dtype=np.str)
    save_arr = pd.DataFrame(save_arr, columns=['filename', 'label'])
    predict_labels = tran_list2str(predict_labels)
    for i in range(len(test_images_path)):
        filename = test_images_path[i].split('/')[-1]
        filename = filename.split('\\')[-1]
        save_arr.values[i, 0] = filename
        save_arr.values[i, 1] = predict_labels[i]
    save_arr.to_csv('submit_test.csv', decimal=',', encoding='utf-8', index=False, index_label=False)
    print(f'complete -> csv')


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_file = "./weights/best_model.pt"
    test_datasets = CharacterDateset('./data', type='test', data_dir="test2")
    test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False)
    label_dict = test_datasets.get_label_dict()
    img_list = test_datasets.get_img_path_list()
    net = resnet34(False).to(device)
    model_dict = torch.load(ckpt_file, map_location=device)
    net.load_state_dict(model_dict['model_state_dict'])
    net.eval()
    predict_label = test_image_predict_topk(net, test_dataloader, label_dict)
    save_csv(img_list, predict_label)
