import torch
import torch.nn as nn
import torch.nn.functional as F


class ContactClassifier(nn.Module):
	def __init__(self, backbone):
		super(ContactClassifier, self).__init__()
		self.feat_extractor = backbone
		self.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

	def forward(self, x):
		x = self.feat_extractor(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)

		output = F.log_softmax(x, dim=1)
		return output


resnet50 = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
modules = list(resnet50.children())[:-1]
resnet50 = nn.Sequential(*modules)
model = ContactClassifier(resnet50)
print(model)


random_data = torch.rand((1, 3, 512, 256))
result = model(random_data)
print(result)

