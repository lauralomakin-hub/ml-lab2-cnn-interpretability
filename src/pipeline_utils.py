# chatGPT: "Kan jag inte göra en SRC-fil för detta?

import json

import torch
import matplotlib.pyplot as plt

from torchvision.transforms.v2.functional import to_pil_image
from torchcam.utils import overlay_mask

from src.model_utils import model
from src.image_utils import load_image
from src.cam_utils import get_activation_map
from src.prediction_utils import get_predicted_class_name  

def run_example(image_path, json_path="data/imagenet_class_index.json", target_layer="layer4"):
    # Load and preprocess the image
    img, input_tensor = load_image(image_path)
    
    # Get the activation map and model output
    activation_map, out, class_index = get_activation_map(
        model, 
        input_tensor,
        target_layer=target_layer
    )

    #logits is the raw output from the model before applying softmax
    logits = out.squeeze(0)
    top_values, top_indices = torch.topk(logits, 5)

    #chatGPT: "Är det verkligen rätt med tanke på hur min pipline ser ut?"

    with open(json_path, "r") as f:
        class_mapping = json.load(f)
    
    print("´\nTop 5 logits")
    for value, index in zip(top_values, top_indices):
        class_name = class_mapping[str(index.item())][1]
        print(class_name, value.item())



    # Get prediction info
    prediction = torch.softmax(out.squeeze(0), dim=0)

    prediction_info = get_predicted_class_name(
        prediction.detach(),
        json_path
    )

    #create overlay
    result = overlay_mask(
        img, 
        to_pil_image(activation_map[0].squeeze(0), mode='F'), 
        alpha=0.5
    )

    #show image
    plt.imshow(result)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return prediction_info
    
def print_top_logits(image_path, json_path="data/imagenet_class_index.json", top_k=5):
    # Load and preprocess the image
    img, input_tensor = load_image(image_path)

    # Run the model without creating an attribution map
    with torch.no_grad():
        out = model(input_tensor.unsqueeze(0))

    logits = out.squeeze(0)
    top_values, top_indices = torch.topk(logits, top_k)

    with open(json_path, "r") as f:
        class_mapping = json.load(f)

    print(f"Top {top_k} logits")
    for value, index in zip(top_values, top_indices):
        class_name = class_mapping[str(index.item())][1]
        print(class_name, value.item())

    