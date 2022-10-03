from re import split

import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from net import resnet18
from nets.vgg16 import Vgg16
import time
import os
import cv2
import PIL.Image as Image
from IPython.display import display
import numpy as np



dataset_dir = "../data/Rice_Image_Dataset/test/"

dir_save_path = "test_result"
                     
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    label_name = ["Arborio","Basmati","Ipsala","Jasmine","Karacadag"]
    label_to_idx = {label_name[i]: i for i in range(len(label_name))}


    print(label_name)
    print(label_to_idx)

    net = Vgg16(5).to(device)
    model_path = "logs/models_vgg16_pretrained_lr0.0001_bs32_1934/ep005-loss0.109-val_loss0.007.pth"
    net.load_state_dict(torch.load(model_path))
    # # switch the model to evaluation mode to make dropout and batch norm work in eval mode
    net.eval()


    loader = transforms.Compose([transforms.RandomCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.4914, 0.4822, 0.4465)])

    # path = "Ipsala/Ipsala (154).jpg"
    # path = "Arborio/Arborio (40).jpg"
    # mode = input("Input mode('predict','dir_predict'):")
    mode = "dir_predict"
    def detect_image(image):
        image_name = image
        image = loader(image).float()
        image = torch.autograd.Variable(image, requires_grad=True)
        image = image.unsqueeze(0)
        image = image.cuda()
        output = net(image)
        # print(image_name)
        # label_gt = image_name.split(" ")[0]
        conf, predicted = torch.max(output.data, 1)
        # pred_t_or_f = 0
        # if label_gt == str(label_name[predicted.item()]):
        #     pred_t_or_f = 1
        # print(label_name[predicted.item()], "confidence: ", conf.item() * 5, "%")
        res = "预测结果：" + str(label_name[predicted.item()]) + ",confidence: " + str(conf.item() * 5) + "%"
        print(res)
        return res


    if mode == "predict":
        while True:
            path = input('Input image filename:')
            try:
                image = Image.open(dataset_dir + path)
            except:
                print('Open Error! Try again!')
                continue
            else:
                detect_image(image)

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dataset_dir)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dataset_dir, img_name)
                image = Image.open(image_path)
                res = detect_image(image)

                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)

                with open(os.path.join(dir_save_path, "classify_res"), 'a') as f:
                    f.write(str(res))
                    f.write("\n")