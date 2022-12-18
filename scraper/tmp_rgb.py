import os
from PIL import Image

# Specify the folder containing the PNG images
folder = '/home/ubuntu/riffusion/riffusion-manipulation/scraper/spectogram_dataset'

# Iterate through all files in the folder
for filename in os.listdir(folder):
  # Check if the file is a PNG image
  if filename.endswith('.png'):
    # Open the image
    image = Image.open(os.path.join(folder, filename))
    # Convert the image to RGB
    image = image.convert('RGB')
    # Save the image with the same file name
    image.save(os.path.join(folder, filename))

print('All images converted to RGB')