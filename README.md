# Things to try
- Freeze Pretrained layers
for param in model.resnet.parameters():
    param.requires_grad = False
for param in model.resnet.fc.parameters():
    param.requires_grad = True
- Add data augmentation for more robustness


# Smile Detection ideas
- Get patch of mouth from keypoints - Template Matching
- Detect from that