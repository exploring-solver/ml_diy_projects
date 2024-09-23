from PIL import Image
import os

# Define the folder where your images are located
image_folder = "c2s_ss_1.0.5"  # Update this with your folder path
output_folder = "output_ss_1.0.5"     # Folder to save the cropped images

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Target size for the width
new_width = 1320

# Iterate through all the files in the image folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        # Open the image
        image_path = os.path.join(image_folder, filename)
        img = Image.open(image_path)

        # Get the original size
        width, height = img.size

        # Ensure the image size matches the expected height (optional)
        if height == 2868 and width == 1324:
            # Crop the image: (left, top, right, bottom)
            left = (width - new_width) / 2  # Crop equally from both sides
            right = width - left
            top = 0
            bottom = height

            cropped_img = img.crop((left, top, right, bottom))

            # Save the cropped image to the output folder
            output_path = os.path.join(output_folder, filename)
            cropped_img.save(output_path)

            print(f"Cropped and saved: {output_path}")
        else:
            print(f"Skipping {filename}, image size is not 1324x2868")

print("All images processed!")
