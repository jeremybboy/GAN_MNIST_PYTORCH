import os
from itertools import islice

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import plot_loss_dicos, plot_examples


class GANTrainer:
    def __init__(self,
                 directory,
                 generator,
                 discriminator,
                 train_loader,
                 test_loader,
                 cnn,
                 linear,
                 lr):
        # generator
        # dataset ou dataloader ?
        self.logdir = directory
        # Optimizer
        self.generator = generator
        self.z_dim = self.generator.z_dim
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.test_loader = test_loader
        #choose if you want a CNN or Linear Architecture
        self.cnn=cnn
        self.linear=linear


        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=lr)
        self.writer = SummaryWriter(log_dir=directory)


    def do_epoch(self, device, num_batches=None, train=True):


        dico = dict(loss_generator=0.0, loss_discriminator=0.0)


        loader = self.train_loader if train else self.test_loader
        self.generator.train(train)
        self.discriminator.train(train)

        for i, (images, true_labels) in tqdm(enumerate(islice(loader, num_batches))):
            if self.cnn:

                #les données pour entraîner le discriminator
                # Move tensors to the configured device
                images = images.to(device)
                # true_labels=true_labels.to(device)
                batch_size = images.size(0)
                real_samples_labels = torch.ones((batch_size,1)).to(device)
                #les données pour entraîner le generator

                latent_space_samples = torch.randn((batch_size, self.z_dim,1,1)).to(device)

                generated_samples = self.generator(latent_space_samples)
                generated_samples_labels = torch.zeros((batch_size, 1)).to(device)
                all_samples = torch.cat((images, generated_samples))
                all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))



               # Forward pass
                with torch.set_grad_enabled(train):
                    logits_discriminator = self.discriminator(all_samples)
                    loss_discriminator = self.loss(logits_discriminator, all_samples_labels)
                    dico['loss_discriminator'] += loss_discriminator.item()

                if train :
                    self.discriminator.zero_grad()
                    loss_discriminator.backward()
                    self.optimizer_discriminator.step()

                with torch.set_grad_enabled(train):
                    latent_space_samples = torch.randn((batch_size, self.z_dim,1,1)).to(device)
                    generated_samples = self.generator(latent_space_samples)
                    logits_discriminator_generated = self.discriminator(generated_samples)
                    loss_generator = self.loss(logits_discriminator_generated, real_samples_labels)
                    dico['loss_generator'] += loss_generator.item()

                if train :
                    self.generator.zero_grad()
                    loss_generator.backward()
                    self.optimizer_generator.step()

            if self.linear:
                # les données pour entraîner le discriminator
                # Move tensors to the configured device
                images = images.to(device)
                # true_labels=true_labels.to(device)
                batch_size = images.size(0)
                real_samples_labels = torch.ones((batch_size, 1)).to(device)
                # les données pour entraîner le generator

                latent_space_samples = torch.randn((batch_size, self.z_dim)).to(device)

                generated_samples = self.generator(latent_space_samples)
                generated_samples_labels = torch.zeros((batch_size, 1)).to(device)
                all_samples = torch.cat((images, generated_samples))
                all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

                # Forward pass
                with torch.set_grad_enabled(train):
                    logits_discriminator = self.discriminator(all_samples)
                    loss_discriminator = self.loss(logits_discriminator, all_samples_labels)
                    dico['loss_discriminator'] += loss_discriminator.item()

                if train:
                    self.discriminator.zero_grad()
                    loss_discriminator.backward()
                    self.optimizer_discriminator.step()

                with torch.set_grad_enabled(train):
                    latent_space_samples = torch.randn((batch_size, self.z_dim)).to(device)
                    generated_samples = self.generator(latent_space_samples)
                    logits_discriminator_generated = self.discriminator(generated_samples)
                    loss_generator = self.loss(logits_discriminator_generated, real_samples_labels)
                    dico['loss_generator'] += loss_generator.item()

                if train:
                    self.generator.zero_grad()
                    loss_generator.backward()
                    self.optimizer_generator.step()

        #fin de l'epoch
        dico = {
            key: (value / (i+1))
            for key, value in dico.items()
        }
        return dico, generated_samples


    def loss(self, outputs, labels):
        # return F.binary_cross_entropy_with_logits(outputs,labels)
        # sig = torch.sigmoid(outputs)
        loss=nn.BCELoss()
        return loss(outputs, labels)




    def train_model(self, num_epochs, device, num_batches=None):

        for epoch in range(num_epochs):
            train_dico, _ = self.do_epoch(device, num_batches, train=True)
            print(f'TRAIN : Epoch [{epoch + 1}/{num_epochs}], Loss discriminator: {train_dico["loss_discriminator"]:.4f}, Loss generator: {train_dico["loss_generator"]:.4f}')

            test_dico, generations = self.do_epoch(device, num_batches, train=False)
            print(f'TEST : Epoch [{epoch + 1}/{num_epochs}], Loss discriminator: {test_dico["loss_discriminator"]:.4f}, Loss generator: {test_dico["loss_generator"]:.4f}')

            plot_loss_dicos(self.writer, epoch, train_dico, test_dico)
            plot_examples(self.writer, epoch, generations)