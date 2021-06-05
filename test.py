import sys
import time
import os
from options.test_options import TestOptions
from data.data_loader import DatasetDataLoader
from models.adgan import TransferModel
from util.visualizer import Visualizer
from util import html
import time

print("bp1", flush=True)

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

print("bp2", flush=True)

data_loader = DatasetDataLoader()
data_loader.initialize(opt)

dataset = data_loader.load_data()
dataset_size = len(data_loader)


model = TransferModel()
model.initialize(opt)

visualizer = Visualizer(opt)

print("bp3", flush=True)

# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

print(opt.how_many)
print(len(dataset))

model = model.eval()
print(model.training)

opt.how_many = 10
# test
for i, data in enumerate(dataset):
    print(' process %d/%d img ..'%(i,opt.how_many), flush=True)
    if i >= opt.how_many:
        break
    model.set_input(data)
    startTime = time.time()
    model.test()
    endTime = time.time()
    print(endTime-startTime, flush=True)
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    img_path = [img_path]
    print(img_path, flush=True)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()




