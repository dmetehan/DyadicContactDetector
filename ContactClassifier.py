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


def init_model(option=1):
	resnet50 = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
	print(list(resnet50.children())[0])
	resnet50.conv1 = nn.Conv2d(34 if option == 1 else 37, 64, kernel_size=7, stride=2, padding=3, bias=False)
	print(list(resnet50.children())[0])
	modules = list(resnet50.children())[:-1]
	resnet50 = nn.Sequential(*modules)
	model = ContactClassifier(resnet50)
	return model


def main():
	model = init_model()
	random_data = torch.rand((1, 3, 512, 256))
	result = model(random_data)
	print(result)


if __name__ == '__main__':
	main()
