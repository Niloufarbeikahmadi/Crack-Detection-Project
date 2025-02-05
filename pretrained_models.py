# pretrained_models.py
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import CrackDataset

def prepare_model():
    """
    Question 1:
      1. Load pre-trained resnet18.
      2. Freeze its parameters.
      3. Replace the final fully-connected layer to classify 2 classes.
    """
    model = models.resnet18(pretrained=True)
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    # Replace the output layer (model.fc) with a new Linear layer for 2 classes.
    num_features = model.fc.in_features  # should be 512 for resnet18
    model.fc = nn.Linear(num_features, 2)
    print("Modified model architecture:")
    print(model)
    return model

def train_model(model, train_dataset, validation_dataset, n_epochs=1, batch_size=100, lr=0.001):
    """
    Question 2:
      Train the model on the training dataset and evaluate accuracy on the validation dataset.
    """
    # Create the loss function
    criterion = nn.CrossEntropyLoss()

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer only for parameters that require gradients
    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr)

    loss_list = []
    accuracy_list = []
    N_test = len(validation_dataset)

    start_time = time.time()
    model.train()
    for epoch in range(n_epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        # Validation phase
        model.eval()
        correct = 0
        with torch.no_grad():
            for x_test, y_test in validation_loader:
                outputs = model(x_test)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_test).sum().item()
        accuracy = correct / N_test
        accuracy_list.append(accuracy)
        print(f"Epoch {epoch+1}/{n_epochs} - Validation Accuracy: {accuracy:.4f}")
        model.train()  # set back to training mode for next epoch

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    # Plot the loss curve
    plt.plot(loss_list)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()
    return model

def find_misclassified(model, validation_dataset, num_samples=4):
    """
    Question 3:
      Identify the first few misclassified samples.
    """
    model.eval()
    misclassified = []
    with torch.no_grad():
        for idx in range(len(validation_dataset)):
            x, y = validation_dataset[idx]
            # Add batch dimension
            x_unsqueezed = x.unsqueeze(0)
            output = model(x_unsqueezed)
            _, predicted = torch.max(output, 1)
            if predicted.item() != y.item():
                misclassified.append((idx, x, y, predicted.item()))
            if len(misclassified) >= num_samples:
                break

    if not misclassified:
        print("No misclassified samples found.")
        return

    for idx, img, true_label, pred_label in misclassified:
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"Index: {idx}, True: {true_label}, Predicted: {pred_label}")
        plt.show()

def main():
    # Create dataset objects. Ensure that the tensor files have been unzipped properly.
    train_dataset = CrackDataset(train=True)
    validation_dataset = CrackDataset(train=False)

    # Prepare the model (Question 1)
    model = prepare_model()

    # Train the model (Question 2)
    model = train_model(model, train_dataset, validation_dataset, n_epochs=1, batch_size=100, lr=0.001)

    # Find misclassified samples (Question 3)
    find_misclassified(model, validation_dataset, num_samples=4)

if __name__ == "__main__":
    main()
