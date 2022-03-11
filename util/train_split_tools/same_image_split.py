import pandas as pd
import numpy as np
import os

mask_folder = '/mnt/gpid08/users/jorge.pueyo/masks'
save_folder = 'train_split'

d = {
    'from': [],
    'to': []
}

images = []
for folder in os.listdir(mask_folder):

    if len(os.listdir(os.path.join(mask_folder, folder))) > 4:
        images.append(folder)

for image in images:

        d['from'].append(image)
        d['to'].append(image)


df = pd.DataFrame(data = d)
df.to_csv(os.path.join(save_folder, 'same_pair_list.csv'), index = False)
