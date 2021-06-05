import re
import xml.etree.ElementTree as ET
import json
import sys
import argparse
from os import listdir, makedirs
import os
from os.path import isfile, join, exists
from PIL import Image, ImageDraw

size_dir = 'bedroom/images'
savedir = 'masks'
valid_labels = ['bed', 'wall', 'window', 'chair', 'lamp', 'pillow', 'dresser', 'art', 't-shirt', 'blanket', 'cabinet', 'ceiling', 'cloth', 'mirror', 'plant', 'shelf']

polygons = []
numFound = 0
filename = ''
imageWidth = 640
imageHeight = 480

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
colors = ['black', 'white', 'red', 'green', 'blue']

bgcolor = black
fgcolor = white

# Create an image with the data in the polygons array
def generateImage(file_dir, filename, polygons):
    #TODO: Hardcoded image dimensions
    original_image = Image.open(os.path.join(size_dir, file_dir + '.jpg'))
    print(original_image.size)
    img = Image.new('RGB', original_image.size, bgcolor)
    pixels = img.load()
    draw = ImageDraw.Draw(img)
    for polygon in polygons:
        draw.polygon(polygon, fill=fgcolor)

    print('Saving image ' + filename)
    if not os.path.exists(os.path.join(savedir, file_dir)):
        os.makedirs(os.path.join(savedir, file_dir))
    img.save(str(join(savedir, file_dir, filename)))


# Check if a file is either json or xml
def isValidFile(file):
    if bool(re.search(r'\.json', file)) or bool(re.search(r'\.xml', file)):
        return True
    return False

# Reset vars to default values
# Used when loading a new file
def clearVars():
    global polygons, numFound, filename, imageWidth, imageHeight
    polygons = []
    numFound = 0
    filename = ''
    imageWidth = 0
    imageHeight = 0

# Parse the xml file and fill in the polygons array
def parseXML(file):
    global polygons, numFound, filename, labels
    labels = {}
    tree = ET.parse(file)
    root = tree.getroot()
    for child in root:
        if child.tag == 'object':
            objType = child[0].text.strip().lower()

            if objType in valid_labels:
                if objType not in labels.keys():
                    labels[objType] = []
                numFound += 1

                for item in child:
                    if item.tag == 'polygon':
                        points = []
                        for item_under_polygon in item:
                            if item_under_polygon.tag == 'pt':
                                x = item_under_polygon[0].text.strip()
                                y = item_under_polygon[1].text.strip()
                                points.append((float(x), float(y)))
                        #polygons.append(points)
                        labels[objType].append(points)

        elif child.tag == 'filename':
            filename = child.text.strip()
    
    return labels


parser = argparse.ArgumentParser(description='Convert LabelMe XML/JSON files to binary images.')

# Required arguments
parser.add_argument('file', metavar='file/folder', type=str, help='path to input file/folder (json/xml/folder)')
parser.add_argument('output', type=str, help='output file type', 
                    choices=['png', 'jpg'])
parser.add_argument('labels', type=str, nargs='+',
                    help='labels to include in the image')

# Optional flags
parser.add_argument('--savedir', required=False, help='directory to save images in (default: masks)')
parser.add_argument('--nosave', required=False, help='dont save image', 
                    action='store_true')
parser.add_argument('--preview', required=False, help='show image preview', 
                    action='store_true')
parser.add_argument('--bgcolor', required=False, help='background color (default: white)', 
                    choices=colors)
parser.add_argument('--fgcolor', required=False, help='foreground/label color (default: black)', 
                    choices=colors)

args = parser.parse_args()

if args.savedir:
    savedir = args.savedir

if not args.nosave:
    if not exists(savedir):
        makedirs(savedir)

if args.bgcolor:
    bgcolor = args.bgcolor

if args.fgcolor:
    fgcolor = args.fgcolor

# List of files to convert
files = []
if isfile(args.file):
    files.append(args.file)
else:
    # Dir passed in
    print('Start parsing items from directory')
    filesInDir = [f for f in listdir(args.file) if isfile(join(args.file, f))]
    for f in filesInDir:
        files.append(str(join(args.file, f)))
for f in files:
    if not isValidFile(f):
        print('Skipping ' + f)
    else:
        labels = parseXML(f)
        filename = re.sub(r'\.\w+', '', filename)

        for label, polygons in labels.items():
            filename_mask = filename + '-' + label + '.png'
            generateImage(filename, filename_mask, polygons)