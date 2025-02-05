# data_preparation.py
import os
import matplotlib.pyplot as plt
from PIL import Image

def list_image_paths(directory):
    """
    List image file paths (JPEG) in the given directory.
    """
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return sorted(image_files)

def display_images(image_paths, titles=None):
    """
    Plot the images provided in the list.
    """
    for idx, path in enumerate(image_paths):
        image = Image.open(path)
        plt.imshow(image)
        if titles and idx < len(titles):
            plt.title(titles[idx])
        else:
            plt.title(f"Image {idx+1}")
        plt.show()

def main():
    # Assuming directories for cracks (positive) and no-cracks (negative)
    positive_dir = "./data/Crack"  # update path as needed
    negative_dir = "./data/NoCrack"  # update path as needed

    # Example: List the first few positive samples
    positive_files = list_image_paths(positive_dir)
    negative_files = list_image_paths(negative_dir)

    # Plot the second image with no cracks (index 1 since Python indexing is zero based)
    if len(negative_files) > 1:
        image2 = Image.open(negative_files[1])
        plt.imshow(image2)
        plt.title("2nd Image With No Cracks")
        plt.show()

    # Question 2: Plot the first three images for the dataset with cracks.
    if len(positive_files) >= 3:
        titles = ["1st Crack Image", "2nd Crack Image", "3rd Crack Image"]
        display_images(positive_files[:3], titles)
    else:
        print("Not enough crack images available.")

if __name__ == "__main__":
    main()
