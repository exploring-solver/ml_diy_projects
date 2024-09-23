from PIL import Image
import os

# Define the folder where your images are located
image_folder = "output_ss_1.0.5"  # Update this with your folder path
output_folder = "output_ss_6.5in_1.0.5"     # Folder to save the processed images

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Target size for resize and crop
target_width_resize = 1242
target_height_resize = 2699
target_height_crop = 2688

# Iterate through all the files in the image folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        # Open the image
        image_path = os.path.join(image_folder, filename)
        img = Image.open(image_path)

        # Get the original size
        width, height = img.size

        # Step 1: Resize if the image is 1320x2868
        if width == 1320 and height == 2868:
            print(f"Resizing {filename} from 1320x2868 to 1242x2699...")
            # Resize to 1242x2699 using LANCZOS instead of ANTIALIAS
            img = img.resize((target_width_resize, target_height_resize), Image.LANCZOS)
            width, height = img.size  # Update width and height after resizing

        # Step 2: Crop if the image is already 1242x2699
        if width == 1242 and height == target_height_resize:
            print(f"Cropping {filename} from 1242x2699 to 1242x2688...")
            # Crop the top 11 pixels to make the height 2688
            left = 0
            top = 11
            right = width
            bottom = target_height_crop + 11

            img = img.crop((left, top, right, bottom))

        # Save the processed image to the output folder
        output_path = os.path.join(output_folder, filename)
        img.save(output_path)
        print(f"Processed and saved: {output_path}")
        
print("All images processed!")
