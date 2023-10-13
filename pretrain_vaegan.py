import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, sampler
from torchvision import transforms as tfs
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tools import loadTifImage
import torchvision.transforms as transforms

def preprocess_img(x):
    x = tfs.ToTensor()(x)
    return (x - 0.5) / 0.5
def deprocess_img(x):
    return (x + 1.0) / 2.0

transform = transforms.Compose([
            transforms.ToTensor(),
    transforms.Normalize([0.5]*39,[0.5]*39)
        ])

train_path = r"/mnt/3.6T-DATA/CBN/DATA/train/" # path of training data
train_dataset = loadTifImage.DatasetFolder(root=train_path, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_path = r"/mnt/3.6T-DATA/CBN/DATA/test/" # path of test data
test_dataset = loadTifImage.DatasetFolder(root=test_path, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1,shuffle=False)

# The VAE structure is as follows, and it is also a generator
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(28*28*39, 400) # 28 is the size of the sample, which can be modified in the function loadTifImage, 39 is the number of channels
        self.fc21 = nn.Linear(400, 20) # mean
        self.fc22 = nn.Linear(400, 20) # var
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 28*28*39)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        if torch.cuda.is_available():
            eps = Variable(eps.cuda())
        else:
            eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x) # encode the data
        z = self.reparametrize(mu, logvar) # Reparameterize as a normal distribution
        return self.decode(z), mu, logvar, self.z # Decode, simultaneously outputting mean and variance

# discriminator
def discriminator():
    net = nn.Sequential(
            nn.Linear(28*28*39, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    return net

# Compute the loss
def ls_discriminator_loss(scores_real, scores_fake): # loss of discriminator
    loss = 0.5 * ((scores_real - 1) ** 2).mean() + 0.5 * (scores_fake ** 2).mean()
    return loss

def ls_generator_loss(recon_x, x, mu, logvar): # loss of generator
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    loss0 = 0.5 * ((recon_x - 1) ** 2).mean()
    # KL divergence
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return loss0+KLD

# setup the optimizer
def get_optimizer(net):
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    return optimizer

# a training epoch
def train_a_gan(D_net, G_net, D_optimizer, G_optimizer, discriminator_loss, generator_loss, show_every=250,
                num_epochs=30):
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in train_dataloader:
            bs = x.shape[0]
            # train the discriminator
            real_data = Variable(x).view(bs, -1).cuda()
            logits_real = D_net(real_data)
            fake_images, mu, logvar, z = G_net(real_data)
            logits_fake = D_net(fake_images)

            d_total_error = discriminator_loss(logits_real, logits_fake)
            D_optimizer.zero_grad()
            d_total_error.backward()
            D_optimizer.step()

            # train the generator
            fake_images, mu, logvar = G_net(real_data)
            gen_logits_fake = D_net(fake_images)
            g_error = generator_loss(gen_logits_fake, real_data, mu, logvar)
            G_optimizer.zero_grad()
            g_error.backward()
            G_optimizer.step()

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))
                imgs_numpy = deprocess_img(fake_images.data.cpu().numpy())
                print()
            iter_count += 1
            # save weights
            torch.save(G_net.state_dict(), './save_model/VAE_GAN_G.pth')
            torch.save(D_net.state_dict(), './save_model/VAE_GAN_D.pth')








