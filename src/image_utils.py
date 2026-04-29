from torchvision.io import decode_image

#chatGPT: "Ska denna del in till image_utils?

#img = decode_image("data/images/dog_german_shepherd.jpg")

#input_tensor = preprocess(img)

from PIL import Image
from src.model_utils import preprocess

def load_image(image_path):
    # Load the image using torchvision's decode_image
    img = Image.open(image_path)
    
    # Preprocess the image using the provided preprocess function
    input_tensor = preprocess(img)
    
    return img, input_tensor