import os
# Fix OpenMP duplicate warning on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from dataset import MultiLabelDataset
from model import MultiLabelResNet


def train():
    # -----------------------
    # Device (CPU)
    # -----------------------
    device = torch.device("cpu")

    # -----------------------
    # Paths
    # -----------------------
    image_dir = "dataset/images"
    label_file = "dataset/labels.txt"

    # -----------------------
    # Transforms
    # -----------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -----------------------
    # Dataset & Loader
    # -----------------------
    dataset = MultiLabelDataset(image_dir, label_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # -----------------------
    # Model
    # -----------------------
    model = MultiLabelResNet(num_classes=4).to(device)

    # -----------------------
    # Loss & Optimizer
    # -----------------------
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # -----------------------
    # Training
    # -----------------------
    num_epochs = 5
    losses = []

    model.train()
    for epoch in range(num_epochs):
        for images, labels, mask in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss = loss * mask
            loss = loss.sum() / mask.sum()

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item():.4f}")

    # -----------------------
    # Save Model
    # -----------------------
    model_path = os.path.join(os.getcwd(), "multilabel_resnet18.pth")
    torch.save(model.state_dict(), model_path)
    print("Model saved at:", model_path)

    # -----------------------
    # Save Loss Curve (FORCED)
    # -----------------------
    plot_path = os.path.join(os.getcwd(), "training_loss.png")

    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("iteration_number")
    plt.ylabel("training_loss")
    plt.title("Aimonk_multilabel_problem")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("Loss curve saved at:", plot_path)
    print("Training completed successfully.")


if __name__ == "__main__":
    train()
