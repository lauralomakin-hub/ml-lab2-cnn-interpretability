from torchvision.models import get_model, get_model_weights

weights = get_model_weights("resnet18").DEFAULT
model = get_model("resnet18", weights=weights).eval()

preprocess= weights.transforms()