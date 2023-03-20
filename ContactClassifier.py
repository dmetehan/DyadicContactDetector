import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.utils import Options


class ContactClassifier(nn.Module):
	def __init__(self, backbone="resnet50", weights="IMAGENET1K_V2", option=Options.jointmaps, dropout=(None, None), copy_rgb_weights=False):
		super(ContactClassifier, self).__init__()
		resnet50 = torch.hub.load("pytorch/vision", backbone, weights=weights)
		conv1_pretrained = list(resnet50.children())[0]
		self.conv1 = nn.Conv2d(34 if option in [Options.gaussian, Options.jointmaps] else
							   (37 if option in [Options.gaussian_rgb, Options.jointmaps_rgb] else 52), 64, kernel_size=7, stride=2, padding=3, bias=False)
		if copy_rgb_weights and self.conv1.weight.shape[1] >= 37:
			self.conv1.weight.data[:, 34:37, :, :] = conv1_pretrained.weight  # copies the weights to the rgb channels
		# 	with torch.no_grad():  # seems like the weights do not train that much.
		# 		self.conv1.weight[:, -3:, :, :].copy_(conv1_pretrained.weight)
		self.dropout_first = nn.Dropout(dropout[0]) if dropout[0] else None
		modules = list(resnet50.children())[(2 if self.dropout_first else 1):-1]  # gets rid of the batchnorm if dropout
		resnet50 = nn.Sequential(*modules)
		self.feat_extractor = resnet50
		self.dropout_last = nn.Dropout(dropout[1]) if dropout[1] else None  # dropout last seems to be a terrible idea.
		self.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

	def forward(self, x):
		x = self.conv1(x)
		if self.dropout_first:
			x = self.dropout_first(x)
		x = self.feat_extractor(x)
		x = torch.flatten(x, 1)
		if self.dropout_last:
			x = self.dropout_last(x)
		x = self.fc(x)

		# x = F.log_softmax(x, dim=1)
		return x


def init_model(option=Options.jointmaps, dropout=(None, None)):
	model = ContactClassifier(backbone="resnet50", weights="IMAGENET1K_V2", option=option, dropout=dropout)
	return model


def main():
	model = init_model(option=Options.jointmaps_rgb, dropout=(0.1, 0.5))
	random_data = torch.rand((1, 37, 512, 256))
	result = model(random_data)
	print(result)
	print(model)
	loss_fn = nn.BCEWithLogitsLoss()
	# Compute the loss and its gradients
	loss = loss_fn(result, torch.zeros(len(result), dtype=int))
	print(list(list(list(model.children())[0].children())[4].children()))
	loss.backward()
	for bneck in list(list(model.children())[0].children())[4].children():
		for name, layer in bneck.named_children():
			try:
				if layer.weight.grad is not None:
					print(name, layer)
			except AttributeError:
				continue


if __name__ == '__main__':
	main()
