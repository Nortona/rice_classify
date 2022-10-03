import torch
import torch.nn as nn
import torch.nn.functional as F



class LeNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=5)

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)

        self.fc_1 = nn.Linear(16 * 53 * 53, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, output_dim)

    def forward(self, x):
        #(3, 224, 224) ---> input
        x = self.conv1(x)
        #(6, 220, 220) ---> output
        x = F.max_pool2d(x, kernel_size=2)
        #(6, 110, 110)
        x = F.relu(x)
        x = self.conv2(x)
        #(16, 106, 106)
        x = F.max_pool2d(x, kernel_size=2)
        #(16, 53, 53)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        h = x
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x
if __name__ == "__main__":
    x = torch.rand(size=(8, 3, 224, 224))
    lenet = LeNet(output_dim=5)
    out = lenet(x)
    print(out.size())