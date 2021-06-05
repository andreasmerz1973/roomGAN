import torch.utils.data
from .dataset import Dataset

class DatasetDataLoader:
    def name(self): 
        return 'DatasetDataLoader'

    def initialize(self, opt):
        self.opt = opt
        self.dataset = Dataset()
        self.dataset.initialize(opt)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self.dataloader.__iter__()
