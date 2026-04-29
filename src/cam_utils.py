from torchcam.methods import LayerCAM

def get_activation_map(model, input_tensor, target_layer="layer4"):
    with LayerCAM(model, target_layer=target_layer) as CAM_extractor:
        out = model(input_tensor.unsqueeze(0))
        class_index = out.squeeze(0).argmax().item()
        activation_map = CAM_extractor(class_index, out)
    
    return activation_map, out, class_index

#chatGPT: JAg har layerCam i srcfil som heter cam_utils" 
# (ang. lägga target_layer på rätt ställe)