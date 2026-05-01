# CNN Interpretability with Attribution Maps

This project is part of Assignment 2 in the Machine Learning course.

The purpose of the project is to explore how a pretrained image classification model makes predictions, and how attribution maps can be used to interpret the result.

## Project description

The project uses a pretrained ResNet18 model from TorchVision. The model is trained on ImageNet and is used to classify different images.

LayerCAM is used to create attribution maps. These maps show which parts of the image the model focuses on when making a prediction.

## What the project includes

- Image classification with ResNet18
- Mapping ImageNet class indexes to class names
- Positive and negative examples
- Logits and confidence values
- Attribution maps with LayerCAM
- A comparison between different layers in the model

## Files

```text
data/
  images/
  imagenet_class_index.json

src/
  Python files used in the project

report.ipynb
README.md
pyproject.toml