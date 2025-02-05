# data_preparation_pytorch.py
import matplotlib.pyplot as plt
from dataset import CrackDataset

def display_dataset_samples(dataset, sample_indices):
    """
    Displays samples from the dataset given the indices.
    """
    for sample in sample_indices:
        image, label = dataset[sample]
        plt.imshow(image.squeeze(), cmap='gray')  # use cmap if images are grayscale
        plt.xlabel("y=" + str(label.item()))
        plt.title(f"Validation data, sample {sample}")
        plt.show()

def main():
    # Create dataset objects for training and validation.
    train_dataset = CrackDataset(train=True)
    validation_dataset = CrackDataset(train=False)
    
    # For example: display the 16th sample and sample 103 from the validation data.
    sample_indices = [16, 103]
    display_dataset_samples(validation_dataset, sample_indices)

if __name__ == "__main__":
    main()
