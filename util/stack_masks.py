import os
from PIL import Image
import numpy as np
from tqdm import tqdm



label_to_id = {
    "bed": 0,
    "wall": 1,
    "window": 2,
    "chair": 3,
    "lamp": 4,
    "pillow": 5,
    "floor": 6,
    "door": 7,
    "curtain": 8,
    "blanket": 9,
    "cabinet": 10,
    "ceiling": 11,
    "table": 12,
    "mirror": 13,
    "plant": 14,
    "shelf": 15,
    "painting": 16,
    "carpet": 17
}


#Stack masks script
if __name__ == "__main__": 

    folder_path = '/mnt/gpid08/users/jorge.pueyo/masks'

    for subfolder in tqdm(os.listdir(folder_path)):

        if os.path.exists(os.path.join('/mnt/gpid08/users/jorge.pueyo/layout', subfolder + '.npy')):
            continue

        mask = np.zeros((18, 256, 256), dtype=np.uint8)

        image_name = subfolder
        
        for filename in os.listdir(os.path.join(folder_path, subfolder)):

            label = '-'.join(filename.split("-")[1:]).replace(".png","")

            mask_image = Image.open(os.path.join(folder_path, subfolder, filename)).convert('L')

            mask_data = np.asarray(mask_image)

            mask[label_to_id[label]] = mask_data


        np.save(os.path.join('/mnt/gpid08/users/jorge.pueyo/layout', image_name), mask)




