import subprocess

images = '0c97e6362df003288e706622da6f8f332de8ae04,0e62c12ba24af35b6e6782bb354d107790a80351,0e426640a8548e7c102425b639ce876fc1210ac8,08bb122cd79b0fe7507d4a7497b9a31c32cfd14a'
images_folder = '/mnt/gpid08/users/jorge.pueyo/bedroom_train/'


for image in images.split(','):
    subprocess.run(
        f'cp {images_folder}{image}.jpg failure/{image}.jpg',
        shell=True
    )