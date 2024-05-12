import cv2
import numpy as np
import os
from PIL import Image
import random
"""
this will be the agumentation for the images. for train the model
to clean the images take look on the autoencoder model in my github
"""
def pixelate_image(image_path, pixel_size):
    # Read the image
    image = cv2.imread(image_path)

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the number of pixels in each block
    block_size = int(pixel_size)

    # Resize the image to a smaller size
    small_image = cv2.resize(image, (width // block_size, height // block_size))

    # Resize the small image back to the original size
    pixelated_image = cv2.resize(small_image, (width, height), interpolation=cv2.INTER_NEAREST)

    # convert the pixelated image to a PIL image
    pixelated_image = cv2.cvtColor(pixelated_image, cv2.COLOR_BGR2RGB) # Convert the image from BGR to RGB
    pixelated_image = Image.fromarray(pixelated_image) # Convert the image to a PIL Image
    return pixelated_image

def add_noise_to_directory(directory, output_directory1=None, output_directory2=None):
  """
  Adds Gaussian noise to all images in a directory and saves them optionally to a new directory.

  Args:
      directory: Path to the directory containing images.
      output_directory: Path to the directory for saving noisy images (optional, defaults to original directory).
  """
  if not output_directory1:
    output_directory1 = directory

  num_images = 0
  for filename in os.listdir(directory):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
      image_path = os.path.join(directory, filename)
      try:
        image = Image.open(image_path)
        input_filename = os.path.join(output_directory2, str(num_images)+".jpg")
        image.save(input_filename)
        noisy_image = pixelate_image(input_filename,pixel_size=20) # noisy should be from type PIL
        
        output_filename = os.path.join(output_directory1, str(num_images)+".jpg")
        noisy_image.save(output_filename)
        num_images += 1
      except Exception as e:
        raise
        print(f"Error processing image: {image_path} - {e}")


if __name__ == "__main__":
  # Replace 'path/to/images' with the actual directory containing your images
  image_directory = os.path.join(os.getcwd(), 'dataset4\\image_add_noise') 
  output_directory1 = os.path.join(os.getcwd(), 'dataset4\\train')
  output_directory2 = os.path.join(os.getcwd(), 'dataset4\\train_cleaned')
  # Optional: Specify a different directory to save noisy images (default: overwrite originals)
  # noisy_image_directory = "path/to/noisy/images"
  add_noise_to_directory(image_directory,output_directory1, output_directory2)
