import torch.nn as nn
from torchsummary import summary

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2),
                nn.BatchNorm2d(32),
                nn.ReLU(32),
                nn.Conv2d(32, 64, 3, 2),
                nn.BatchNorm2d(64),
                nn.ReLU(64),
                nn.Conv2d(64, 64, 3, 2),
                nn.BatchNorm2d(64),
                nn.ReLU(64),
                nn.Conv2d(64, 128, 3, 2),
                nn.BatchNorm2d(128),
                nn.ReLU(128),
                nn.Conv2d(128, 128, 3, 2),
                nn.BatchNorm2d(128),
                nn.ReLU(128),
                nn.Conv2d(128, 256, 3, 2),
                nn.BatchNorm2d(256),
                nn.ReLU(256),
        )

        self.classifier = nn.Sequential(
                nn.Linear(256*2*2, 2),

        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = Model1()
    summary(model.cuda(), (3, 224, 224))
