import json
import torch

def get_predicted_class_name(output_tensor, json_path):
    # Get the predicted class index
    predicted_index = output_tensor.argmax().item()
    
    # Load the ImageNet class index mapping
    with open(json_path) as f:
        class_index = json.load(f)

    class_id = class_index[str(predicted_index)][0]  # This is the class ID (e.g., "n02106662")
    class_name = class_index[str(predicted_index)][1]
    confidence = output_tensor[predicted_index].item()

    return {
        "class_index": predicted_index,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence        
    }