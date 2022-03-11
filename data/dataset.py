from .base_dataset import BaseDataset, get_transform
import numpy as np
from PIL import Image
import os
import torch
import pandas as pd

class Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        #self.dir_P = os.path.join(opt.dataroot, opt.phase) #Room images
        self.dir_P = '/mnt/gpid08/users/jorge.pueyo/ScanNet/images'
        #self.dir_K = os.path.join(opt.dataroot, opt.phase + 'K') #Layouts
        self.dir_K = '/mnt/gpid08/users/jorge.pueyo/ScanNet/layouts'
        #self.dir_SP = opt.dirSem #semantic

        self.SP_input_nc = opt.SP_input_nc

        self.init_categories('/mnt/gpid07/imatge/jorge.pueyo/ScanNet/pairs/pair_list.csv')
        self.transform = get_transform()

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        self.size = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            self.pairs.append(pair)

        print('Loading data pairs finished ...')


    def __getitem__(self, index):
        #if self.opt.phase == 'train':
        #    index = np.random.randint(0, self.size-1)


        P1_name, P2_name = self.pairs[index]

        #Gets path for room 1
        P1_path = os.path.join(self.dir_P, P1_name + '.jpg') # Room 1
        BP1_path = os.path.join(self.dir_K, P1_name + '.npy') # Semantic 1

        #Gets path for room 2
        P2_path = os.path.join(self.dir_P, P2_name + '.jpg') # Room 2
        BP2_path = os.path.join(self.dir_K, P2_name + '.npy') # Semantics 2


        P1_img = Image.open(P1_path).convert('RGB')
        BP1_img = np.load(BP1_path)

        P2_img = Image.open(P2_path).convert('RGB')
        BP2_img = np.load(BP2_path)

        BP1 = torch.from_numpy(BP1_img).float() #h, w, c
        #print("Tamaño de la pose 1:", BP1.size()) 

        BP2 = torch.from_numpy(BP2_img).float()
        #print("Tamaño de la pose 2:", BP2.size())


        P1 = self.transform(P1_img)
        P2 = self.transform(P2_img)

        #Stacked masks
        SP1 = np.load(BP1_path)

        return {'P1': P1, 'BP1': BP1, 'SP1': SP1, 'P2': P2, 'BP2': BP2,
                'P1_path': P1_name, 'P2_path': P2_name}

                

    def __len__(self):
        return self.size

    def name(self):
        return 'Dataset'
