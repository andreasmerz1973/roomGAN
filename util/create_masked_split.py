import pandas as pd
import numpy as np
import os
from tqdm import tqdm

save_folder = 'train_split'

d = {
    'from': [],
    'to': []
}

image_folder_1 = '/mnt/gpid08/users/jorge.pueyo/masks'
image_folder_2 = '/mnt/gpid08/users/jorge.pueyo/ADEChallengeData2016/bedroom_resize/masks'
layout_folder = '/mnt/gpid08/users/jorge.pueyo/layout'
images = {}

for folder in tqdm(os.listdir(image_folder_1)):
    images[folder] = set()

    for mask_file in os.listdir(os.path.join(image_folder_1, folder)):
    
        label = (mask_file.split('-')[-1]).split('.')[0]
        images[folder].add(label)

"""
for folder in tqdm(os.listdir(image_folder_2)):
    images[folder] = set()

    for mask_file in os.listdir(os.path.join(image_folder_2, folder)):
    
        label = mask_file.split('-')[-1].split('.')[0]
        images[folder].add(label)
"""

for image_1, labels_1 in tqdm(images.items()):

    if not os.path.exists(os.path.join(layout_folder, image_1 + '.npy')):
        continue

    counter = 0
    for image_2, labels_2 in images.items():

        if not os.path.exists(os.path.join(layout_folder, image_2 + '.npy')):
                continue

        if (image_1 is not image_2) and (labels_1 == labels_2):

            d['from'].append(image_1)
            d['to'].append(image_2)

            counter += 1
        if counter == 1:
            break


df = pd.DataFrame(data = d)
df.to_csv(os.path.join(save_folder, 'masked_pair_list.csv'), index = False)
