import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
from nni.compression.torch import SlimPruner, L1FilterPruner, ActivationMeanRankFilterPruner
import dataset

class MobileModel(torch.nn.Module):
    def __init__(self):
        super(MobileModel, self).__init__()
        model = models.vgg16(pretrained=True)
        self.features = model.features
        
        # self.classifier = nn.Sequential(
        #         nn.Dropout(p=0.3),
        #        nn.AdaptiveAvgPool2d((1, 1)),
        # )
        self.maxpool = nn.MaxPool2d(7, 7)
        self.linear = nn.Linear(512, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.maxpool(x)
        # x = F.max_pool2d(x, 7, 7)
        # x = self.classifier(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.linear(x)

        return x

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        model = models.vgg16(pretrained=True)
        self.features = model.features

        self.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.linear = nn.Linear(1028, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.linear(x)

        return x

config_list = [{'sparsity': 0.5, 'op_types': ['Conv2d']}]
pretrain_epochs = 1
prune_epochs = 1
device = 'cuda'
train_path = './train'
test_path = './test'

train_data_loader = dataset.train_loader(train_path)
test_data_loader = dataset.test_loader(test_path)

criterion = torch.nn.CrossEntropyLoss()

def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('{:2.0f}% Loss {}'.format(100*batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        acc = 100 * correct / len(test_loader.dataset)

        print("Loss: {} Accuracy: {}%\n".format(test_loss, acc))

    return acc

model = MobileModel().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
print("start model training...")
for epoch in range(pretrain_epochs):
    train(model, device, train_data_loader, optimizer)
    test(model, device, test_data_loader)
torch.save(model.state_dict(), 'pretrained_model.pth')
print("start model pruning...")
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
best_top1 = 0
# pruner = SlimPruner(model, config_list, optimizer)
pruner = ActivationMeanRankFilterPruner(model, config_list, optimizer)
model = pruner.compress()

for epoch in range(prune_epochs):
    pruner.update_epoch(epoch)
    print("# Epoch {} #".format(epoch))
    train(model, device, train_data_loader, optimizer)
    top1 = test(model, device, test_data_loader)
    if top1 > best_top1:
        pruner.export_model(model_path='pruned_model.pth', mask_path='pruned_mask.pth')
        from nni.compression.torch import apply_compression_results
        from nni.compression.speedup.torch import ModelSpeedup
        model = MobileModel().cuda()
        model.eval()
        apply_compression_results(model, 'pruned_mask.pth', None)
        m_speedup = ModelSpeedup(model, torch.randn(1, 3, 224, 224).cuda(), 'pruned_mask.pth', None)
        m_speedup.speedup_model()
        torch.save(model.state_dict(), 'pruned_speedup_model.pth')
