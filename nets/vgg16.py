import torch.nn as nn
import torch
import torchvision.models as models

class Vgg16(nn.Module):
    def __init__(self,num_classes,pretrained=False):
        super(Vgg16,self).__init__()
        self.num_classes = num_classes

        # net = []
        #
        # # block 1
        # net.append(nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3, stride=1))
        # net.append(nn.ReLU())
        # net.append(nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1))
        # net.append(nn.ReLU())
        # net.append(nn.MaxPool2d(kernel_size=2, stride=2))
        #
        # # block 2
        # net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        # net.append(nn.ReLU())
        # net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        # net.append(nn.ReLU())
        # net.append(nn.MaxPool2d(kernel_size=2, stride=2))
        #
        # # block 3
        # net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1))
        # net.append(nn.ReLU())
        # net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        # net.append(nn.ReLU())
        # net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        # net.append(nn.ReLU())
        # net.append(nn.MaxPool2d(kernel_size=2, stride=2))
        #
        # # block 4
        # net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1))
        # net.append(nn.ReLU())
        # net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        # net.append(nn.ReLU())
        # net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        # net.append(nn.ReLU())
        # net.append(nn.MaxPool2d(kernel_size=2, stride=2))
        #
        # # block 5
        # net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        # net.append(nn.ReLU())
        # net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        # net.append(nn.ReLU())
        # net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
        # net.append(nn.ReLU())
        # net.append(nn.MaxPool2d(kernel_size=2, stride=2))
        #
        # # add net into class property
        # self.extract_feature = nn.Sequential(*net)
        #
        # # define an empty container for Linear operations
        # classifier = []
        # classifier.append(nn.Linear(in_features=512*7*7, out_features=4096))
        # classifier.append(nn.ReLU())
        # classifier.append(nn.Dropout(p=0.5))
        # classifier.append(nn.Linear(in_features=4096, out_features=4096))
        # classifier.append(nn.ReLU())
        # classifier.append(nn.Dropout(p=0.5))
        # classifier.append(nn.Linear(in_features=4096, out_features=self.num_classes))
        #
        # # add classifier into class property
        # self.classifier = nn.Sequential(*classifier)

        self.model = models.vgg16(pretrained=pretrained)
        # self.classify = nn.Linear(1000,5)
        self.fc1 = nn.Linear(in_features=1000,out_features=512)
        self.fc2 = nn.Linear(in_features=512,out_features=128)
        self.fc3 = nn.Linear(in_features=128,out_features=32)
        self.fc4 = nn.Linear(in_features=32,out_features=num_classes)



    def forward(self, x):
        # feature = self.extract_feature(x)
        # feature = feature.view(x.size(0), -1)
        # classify_result = self.classifier(feature)
        # return classify_result

        x = self.model(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        out = self.fc4(x)
        return out

# if __name__ == "__main__":
#     x = torch.rand(size=(8, 3, 224, 224))
#     vgg = Vgg16(num_classes=5)
#     out = vgg(x)
#     print(vgg)
#     print(out.size())