import torch.nn as nn
import torch
import torchvision.models as models



class resnet101(nn.Module):
    def __init__(self):
        super(resnet101, self).__init__()
        self.model = models.resnet101(pretrained=True)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features,176)



    def forward(self,x):
        out = self.model(x)
        return out





def pytorch_resnet101():
    return resnet101()

class resnet34(nn.Module):
    def __init__(self):
        super(resnet34, self).__init__()
        self.model = models.resnet34(pretrained=True)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features,176)

    def forward(self,x):
        out = self.model(x)
        return out

def pytorch_resnet34():
    return resnet34()


class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features,5)

    def forward(self,x):
        out = self.model(x)
        return out



def pytorch_resnet18():
    return resnet18()


class Vgg16(nn.Module):
    def __init__(self) -> None:
        super(Vgg16,self).__init__()
        