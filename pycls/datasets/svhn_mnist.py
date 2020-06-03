from torchvision.datasets import mnist
from torchvision.datasets import svhn
from torchvision import transforms
from torch.utils.data import ConcatDataset

def MNIST(data_path, split, *args, **kwargs):
    mnist_trans = transforms.Compose([
                           transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                           transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                       ])
    return mnist.MNIST(data_path, split=='train', mnist_trans, download=True)

def SVHN(data_path, split, *args, **kwargs):
    svhn_trans = transforms.Compose([
                        transforms.Resize(32),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.437, 0.4437, 0.4728), 
                            (0.1980, 0.2010, 0.1970)
                        ),
                    ])
    if split == 'test':
        return svhn.SVHN(data_path, 'test', svhn_trans, download=True)
    elif split == 'train':
        return svhn.SVHN(data_path, 'train', svhn_trans, download=True)
    elif split == 'train+extra':
        train_loader = svhn.SVHN(data_path, 'train', svhn_trans, download=True)
        extra_loader = svhn.SVHN(data_path, 'extra', svhn_trans, download=True)
        return ConcatDataset((train_loader, extra_loader))
