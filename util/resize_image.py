from PIL import Image
import os

save_path = '/mnt/gpid08/users/jorge.pueyo/ADEChallengeData2016/bedroom_resize/images'
img_path = '/mnt/gpid08/users/jorge.pueyo/ADEChallengeData2016/bedroom_resize/images'

"""
for subfolder in os.listdir(img_path):
    for filename in os.listdir(os.path.join(img_path, subfolder)):
        img = Image.open(os.path.join(img_path, subfolder, filename))
        width, height = img.size

        if width > height:
            delta = width - height
            left = int(delta/2)
            upper = 0
            right = height + left
            lower = height
        else:
            delta = height - width
            left = 0
            upper = int(delta/2)
            right = width
            lower = width + upper

        img = img.crop((left, upper, right, lower))
        img = img.resize((256,256))

        img.save(os.path.join(save_path, subfolder, filename))
"""
for filename in os.listdir(os.path.join(img_path)):
    img = Image.open(os.path.join(img_path, filename))
    width, height = img.size

    if width > height:
        delta = width - height
        left = int(delta/2)
        upper = 0
        right = height + left
        lower = height
    else:
        delta = height - width
        left = 0
        upper = int(delta/2)
        right = width
        lower = width + upper

    img = img.crop((left, upper, right, lower))
    img = img.resize((256,256))

    img.save(os.path.join(save_path, filename))