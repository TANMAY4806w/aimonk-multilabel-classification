import torch
from PIL import Image
import torchvision.transforms as transforms

from model import MultiLabelResNet

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ATTRIBUTES = ["Attr1", "Attr2", "Attr3", "Attr4"]


def infer(image_path, threshold=0.5):
    device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model = MultiLabelResNet(num_classes=4)
    model.load_state_dict(torch.load("multilabel_resnet18.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs)[0]

    present_attrs = [
        ATTRIBUTES[i] for i, p in enumerate(probs) if p >= threshold
    ]

    print("Attributes present:", present_attrs)


if __name__ == "__main__":
    # Example usage
    infer("dataset/images/image_0.jpg")
