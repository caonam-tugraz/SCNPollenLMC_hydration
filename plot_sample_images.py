import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Function to get a list of subfolders in the given folder
def get_subfolders(folder_path):
    return [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]

# Function to get a list of image files in the given folder
def get_image_files(folder_path):
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]

# Main function to randomly select and plot images from each subfolder
def plot_random_images_from_subfolders(main_folder):
    subfolders = get_subfolders(main_folder)
    num_subfolders = len(subfolders)
    fig, axs = plt.subplots(num_subfolders, 3, figsize=(15, 5 * num_subfolders))  # Adjusting for an additional column for folder names
    # fig.suptitle('Hydrated', fontsize=60)

    for idx, subfolder in enumerate(subfolders):
        image_files = get_image_files(subfolder)
        # axs[idx, 0].text(0.5, 0.5, os.path.basename(subfolder), verticalalignment='center', horizontalalignment='center', fontsize=60)
        # axs[idx, 0].axis('off')  # Hide the axis for the text cell
        if len(image_files) >= 3:
            selected_images = random.sample(image_files, 3)
            # for col, image_path in enumerate(selected_images, start=1):
            for col, image_path in enumerate(selected_images):
                img = Image.open(image_path).resize((224, 224))  # Resize the image to 224x224
                axs[idx, col].imshow(img)
                axs[idx, col].axis('off')
        else:
            print(f"Not enough images in folder: {subfolder}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Specify the main folder path
# main_folder_path = 'data/images_3_types_dry_7030_train'

main_folder_path = 'data/images_3_types_half_hydrated_7030_train'

# main_folder_path = 'data/images_3_types_hydrated_7030_train'


# Plot random images from each subfolder
plot_random_images_from_subfolders(main_folder_path)






