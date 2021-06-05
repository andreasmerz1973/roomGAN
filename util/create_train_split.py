import pandas as pd
import numpy as np
import os

save_folder = 'train_split'

d = {
    'from': [],
    'to': []
}
from_list = []

images = []
for folder in os.listdir('/mnt/gpid08/users/jorge.pueyo/ADEChallengeData2016/bedroom_resize/masks'):
    images.append(folder)

number_repetitions = 8

for image in images:
    for i in range(number_repetitions):

        random_index = np.random.randint(0, len(images))
        to_image = images[random_index]

        d['from'].append(image)
        d['to'].append(to_image)


df = pd.DataFrame(data = d)
df.to_csv(os.path.join(save_folder, 'pair_list.csv'), index = False)
