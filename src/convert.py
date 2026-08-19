import torch

from resnet import ResNet

model = ResNet()
model.load_state_dict(torch.load("./41270.pt", map_location="cpu"))
scripted = torch.jit.script(model)
scripted.save("./41270.scripted.pt")
