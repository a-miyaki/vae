import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class EncoderBlock(nn.Module):
	def __init__(self, in_ch, out_ch, kernel_size, padding, stride):
		super(EncoderBlock, self).__init__()
		self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
		self.bn = nn.BatchNorm2d(num_features=out_ch, momentum=0.9)
	
	def forward(self, ten, out=False, t=False):
		if out:
			ten = self.conv(ten)
			ten_out = ten
			ten = self.bn(ten)
			ten = F.relu(ten, False)
			return ten, ten_out
		else:
			ten = self.conv(ten)
			ten = self.bn(ten)
			ten = F.relu(ten, True)
			return ten


class DecoderBlock(nn.Module):
	def __init__(self, in_ch, out_ch, kernel_size, padding, stride):
		super(DecoderBlock, self).__init__()
		self.conv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
		self.bn = nn.BatchNorm2d(num_features=out_ch, momentum=0.9)

	def forward(self, ten):
		ten = self.conv(ten)
		ten = self.bn(ten)
		ten = F.relu(ten, True)
		return ten


class Encoder(nn.Module):
	def __init__(self, in_ch=1, z_size=256):
		super(Encoder, self).__init__()
		self.size = in_ch
		layers_list = []
		for i in range(5):
			if i == 0:
				layers_list.append(EncoderBlock(in_ch=self.size, out_ch=64, kernel_size=3, padding=1, stride=1))
				self.size = 64
			elif i <= 3:
				layers_list.append(EncoderBlock(in_ch=self.size, out_ch=self.size*2, kernel_size=3, padding=1, stride=1))
				self.size *= 2
			else:
				layers_list.append(EncoderBlock(in_ch=self.size, out_ch=self.size, kernel_size=4, padding=1, stride=2))
		
		# calcuration image size by cnn per every layer : out_size = (in_size + (2 * padding) - filter_size) / stride + 1
		self.conv = nn.Sequential(*layers_list)
		self.fc = nn.Sequential(nn.Linear(in_features=16 * 16 * self.size, out_features=1024, bias=False),
								nn.BatchNorm1d(num_feature=1024, momentum=0.9),
								nn.ReLU(True))
		self.mu = nn.Linear(in_features=1024, out_features=z_size)
		self.var = nn.Linear(in_features=1024, out_features=z_size)
		
	def forward(self, ten):
		ten = self.conv(ten)
		ten = ten.view(len(ten), -1)
		ten = self.fc(ten)
		mu = self.mu(ten)
		logvar = self.var(ten)
		return mu, logvar
	
	def __call__(self, *args, **kwargs):
		return super(Encoder, self).__call__(*args, **kwargs)


class Decoder(nn.Module):
	def __init__(self, z_size, size):
		super(Decoder, self).__init__()
		self.fc = nn.Sequential(nn.Linear(in_features=z_size, out_features=16 * 16 * size, bias=False),
					nn.BatchNorm1d(num_features=16 * 16 * size, momentum=0.9),
					nn.ReLU(True))
		self.size = size
		layers_list = []
		for i in range(5):
			if i == 0:
				layers_list.append(DecoderBlock(in_ch=self.size, out_ch=self.size, kernel_size=3, padding=1, stride=1))
			elif i <= 3:
				layers_list.append(DecoderBlock(in_ch=self.size, out_ch=self.size // 2, kernel_size=3, padding=1, stride=1))
				self.size = self.size // 2
			else:
				layers_list.append(DecoderBlock(in_ch=self.size, out_ch=self.size, kernel_size=5, padding=1, stride=1))
		layers_list.append(nn.Sequential(
			nn.Conv2d(in_channels=self.size, out_channels=1, kernel_size=5, stride=1, padding=2),
			nn.Tanh()
		))
		
		self.conv = nn.Sequential(*layers_list)
		
	def forwars(self, ten):

		ten = self.fc(ten)
		ten = ten.view(len(ten), -1, 16, 16)
		ten = self.conv(ten)
		return ten
		
	def __call__(self, *args, **kwargs):
		return super(Decoder, self).__call__(*args, **kwargs)


class Discriminator(nn.Module):
	def __init__(self, in_ch=1, recon_level=3):
		super(Discriminator, self).__init__()
		self.size = in_ch
		self.recon_level = recon_level
		self.conv = nn.ModuleList()
		self.conv.append(nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
			nn.ReLU(inplace=True)))
		self.size = 32
		self.conv.append(EncoderBlock(in_ch=self.size, out_ch=128, kernel_size=3, padding=1, stride=1))
		self.size = 128
		self.conv.append(EncoderBlock(in_ch=self.size, out_ch=256, kernel_size=3, padding=1, stride=1))
		self.size = 256
		self.conv.append(EncoderBlock(in_ch=self.size, out_ch=256, kernel_size=3, padding=1, stride=1))
		self.fc = nn.Sequential(
			nn.Linear(in_features=16 * 16 * self.size, out_features=512, bias=False),
			nn.BatchNorm1d(num_features=512, momentum=0.9),
			nn.ReLU(inplace=True),
			nn.Linear(in_features=512, out_features=1))
		
		def forward(self, ten, other_ten, mode='REC'):
			if mode == "REC":
				ten = torch.cat((ten, other_ten), 0)
				for i, lay in enumerate(self.conv):
					if i == self.recon_level:
						ten, layer_ten = lay(ten, True)
						layer_ten = layer_ten.view(len(layer_ten), -1)
						return layer_ten
					else:
						ten = lay(ten)
			else:
				ten = torch.cat((ten, other_ten), 0)
				for i, lay in enumerate(self.conv):
					ten = lay(ten)
				
				ten = ten.view(len(ten), -1)
				ten = self.fc(ten)
				return F.sigmoid(ten)
		
		def __call__(self, *args, ** kwargs):
			return super(Discriminator, self).__call__(*args, **kwargs)


class VaeGan(nn.Module):
	def __init__(self, z_size=128, recon_level=3):
		super(VaeGan, self).__init()
		self.z_size = z_size
		self.encoder = Encoder(z_size=self.z_szie)
		self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size)
		self.discriminator = Discriminator(in_ch=1, recon_level=recon_level)
		self.init_parameters()
	
	def init_parameters(self):
		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
				if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
					scale = 1.0/np.sqrt(np.prod(m.weight.shape[1:]))
					scale /= np.sqrt(3)
					nn.init.uniform(m.weight, -scale, scale)
				if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
					nn.init.constant(m.bias, 0.0)
	
	def forward(self, ten, gen_size=10):
		if self.training:
			# save original image
			ten_original = ten
			# encode
			mus, log_variances = self.encoder(ten)
			# we need the true variance, not the log one
			variances = torch.exp(log_variances * 0.5)
			# sample from a gassian
			
			ten_from_normal = Variable(torch.randn(len(ten), self.z_size).cuda(), requires_grad=True)			
			# shift and scale using the means and variances
			
			ten = ten_from_normal * variances * mus
			# decode the tensor
			ten = self.decoder(ten)
			# discriminator for reconstruction
			ten_layer = self.discriminator(ten, ten_original, "REC")
			# decoder for samples
			
			ten_from_normal = Variable(torch.randn(len(ten), self.z_size).cuda(), requires_grad=True)
			
			ten = self.decoder(ten_from_normal)
			ten_class = self.discriminator(ten.original, ten, "GAN")
			return ten, ten_class, ten_layer, mus, log_variances
		else:
			if ten is None:
				# just sample and decode
				ten = Variable(torch.randn(gen_size, self.z_size).cuda(), requires_grad=False)
				ten = self.decoder(ten)
			else:
				mus, log_variances = self.encoder(ten)
				# we need the true variances, not the log variances
				variances = torch.exp(log_variances * 0.5)
				# sample from a gassian
				
				ten_from_normal = Variable(torch.randn(len(ten), self.z_size).cuda(), requires_grad=False)
				# shift and scale using the means and variances
				ten = ten_from_normal * variances + mus
				# decode the tensor
				ten = self.decoder(ten)
			return ten

	def __call__(self, *args, **kwargs):
		return super(VaeGan, self).__call__(*args, **kwargs)
	
	@staticmethod
	def loss(ten_original, ten_predict, layer_original, layer_predicted, labels_original, labels_sampled, mus, variances):
		"""
		:param ten_original: original images
		:param ten_predict:  predicted images (output of the decoder)
		:param layer_original:  intermediate layer for original (intermediate output of the discriminator)
		:param layer_predicted: intermediate layer for reconstructed (intermediate output of the discriminator)
		:param labels_original: labels for original (output of the discriminator)
		:param labels_predicted: labels for reconstructed (output of the discriminator)
		:param labels_sampled: labels for sampled from gaussian (0,1) (output of the discriminator)
		:param mus: tensor of means
		:param variances: tensor of diagonals of log_variances
		:return:
		"""

		# reconstruction error, not used for the loss but, useful to evaluate quality
		nle = 0.5 * (ten_original.view(len(ten_original), -1) - ten_predict.view(len(ten_predict), -1)) ** 2
		# kl-divergence
		kl = -0.5 * torch.sum(-variances.exp() - torch.pow(mus, 2) + variances + 1, 1)
		# mse between intermediate layers
		mse = torch.sum(0.5 * (layer_original - layer_predicted) ** 2, 1)
		# bce for decode and discriminator for original, sample and reconstructed
		# the only excluded is the bce_gen_original
		
		bce_dis_original = -torch.log(labels_original + 1e-3)
		bce_dis_sampled = -torch.log(1 - labels_sampled + 1e-3)
		
		bce_gen_original = -torch.log(1 - labels_original + 1e-3)
		bce_gen_sample = -torch.log(labels_sampled + 1e-3)
		
		'''
		bce_gen_predicted = nn.BCEWithLogitsLoss(size_average=False)(labels_predicted,
			Variable(torch.ones_like(labels_predicted.data).cuda(), requires_grad=False))
		bce_gen_sampled = nn.BCEWithLogitsLoss(size_average=False)(labels_sampled,
			Variable(torch.ones_like(labels_sampled.data).cuda(), requires_grad=False))
		bce_dis_original = nn.BCEWithLogitsLoss(size_average=False)(labels_original,
			Variable(torch.ones_like(labels_original.data).cuda(), requires_grad=False))
		bce_dis_predicted = nn.BCEWithLogitsLoss(size_average=False)(labels_sampled,
			Variable(torch.ones_like(labels_sampled.data).cuda(), requires_grad=False))
		bce_dis_sampled = nn.BCEWithLogitsLoss(size_average=False)(labels_sampled,
			Variable(torch.ones_like(labels_sampled.data).cuda(), requires_grad=False))
		'''
		return nle, kl, mse, bce_dis_original, bce_dis_sampled, bce_gen_original, bce_gen_sample
