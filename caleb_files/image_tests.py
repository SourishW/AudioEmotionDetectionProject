from PIL import Image
import os

def files_iterator(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            yield filepath

for file in files_iterator('./img'):
    img = Image.open(file)
    width, height = img.size

    print(width, height)
