import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data

def train_loader(path, batch_size=32, num_workers=8, pin_memory=True):
    return data.DataLoader(
            datasets.ImageFolder(path, 
                                 transforms.Compose([
                                     transforms.Scale(256),
                                     transforms.RandomSizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]),
                                     ])),
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory)


def test_loader(path, batch_size=32, num_workers=8, pin_memory=True):
    return data.DataLoader(
            datasets.ImageFolder(path, 
                                 transforms.Compose([
                                     transforms.Scale(256),
                                     transforms.RandomSizedCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]),
                                     ])),
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory)


