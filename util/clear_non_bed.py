import os
from tqdm import tqdm

dataset_path = '/mnt/gpid08/users/jorge.pueyo/masks'

for folder in tqdm(os.listdir(dataset_path)):
    if not os.path.exists(os.path.join(dataset_path, folder, folder + '-bed.png')):

        os.system('rm -R /mnt/gpid08/users/jorge.pueyo/masks/{}'.format(folder))
