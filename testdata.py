data = datasets.CIFAR10("./data/", transform=transform, train=True, download=True)

class CustimDataset(data.Dataset):
    def __init__(self):
        # Todo
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        return 0


