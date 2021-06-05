from PIL import Image
import numpy as np
import os

pixel_to_label = {
    8: 'bed',
    1: 'wall',
    9: 'window',
    20: "chair",
    37: "lamp",
    58: "pillow",
    4: "floor",
    15: "door",
    19: "curtain",
    11: "cabinet",
    6: "ceiling",
    16: "table",
    28: "mirror",
    18: "plant",
    25: "shelf",
    23: "painting",
    29: "carpet",


    31: "chair",
    132: "bed",
    40: "pillow",
    36: "cabinet"

}

dataset_path = '/mnt/gpid08/users/jorge.pueyo/ADEChallengeData2016/bedroom_resize'

for image in os.listdir(os.path.join(dataset_path, 'annotations')):

    filename = image.split('.')[0]
    print(filename)

    img = Image.open(os.path.join(dataset_path, 'annotations', image)).convert('RGB')
    img_array = np.array(img)

    objects = set(img.getdata())
    colors = img.getcolors()

    if not os.path.exists(os.path.join(dataset_path, 'masks', filename)):
        os.makedirs(os.path.join(dataset_path, 'masks', filename))

    for index, color_count in enumerate(colors):
        
        count, color = color_count
        pixel_value = color[0]

        if pixel_value in pixel_to_label.keys() and count > 100:
            label = pixel_to_label[pixel_value]
        else:
            continue
        
        print(pixel_value, label)

        height, width, depth = img_array.shape

        data = np.copy(img_array)

        if os.path.exists(os.path.join(dataset_path, 'masks', filename, filename + '-' + label + '.png')):
            blank = np.array(Image.open(os.path.join(dataset_path, 'masks', filename, filename + '-' + label + '.png')).convert('RGB'))
        else:
            blank = np.array(Image.new( 'RGB', (width, height), 'black'))

        blank[(data == color).all(axis = -1)] = (255,255,255)

        mask = Image.fromarray(blank).convert('L')
        mask.save(os.path.join(dataset_path, 'masks', filename, filename + '-' + label + '.png'))

    
