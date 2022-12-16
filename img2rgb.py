from PIL import Image
init_image = Image.open('invader.png').convert('RGB')
init_image.save('RGB.png')