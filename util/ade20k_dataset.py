from PIL import Image
import os

dataset_path = 'ADE20K_Bedroom'
image_path = 'ADE20K_Bedroom/Images'
annotation_path = 'ADE20K_Bedroom/Annotation'

for image in os.listdir(dataset_path):
    if image.endswith('.jpg'):
        os.system("mv {} {}".format(
            os.path.join(dataset_path, image),
            os.path.join(image_path, image)
        ))

    elif image.endswith('_seg.png'):
        os.system("mv {} {}".format(
            os.path.join(dataset_path, image),
            os.path.join(annotation_path, image)
        ))
