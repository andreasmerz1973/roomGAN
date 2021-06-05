import os
from PIL import Image
from tqdm import tqdm

masks_path = '/mnt/gpid08/users/aleix.clemens/masks'

for filename in tqdm(os.listdir(masks_path)):
    img = Image.open(os.path.join(masks_path, filename))

    img = img.crop(256, 0, 512, 256)
    img.save(os.path.join(masks_path, filename))
