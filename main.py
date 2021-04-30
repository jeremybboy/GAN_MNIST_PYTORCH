import numpy as np
import os
import torchvision.datasets
from torchvision import transforms
import torch
from model.LinearGenerator import LinearGenerator
from model.LinearDiscriminator import LinearDiscriminator
from model.CNNGenerator import CNNGenerator
from model.CNNDiscriminator import CNNDiscriminator
from training import GANTrainer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
torch.autograd.set_detect_anomaly(True)
import click


@click.command()
@click.option('-l', '--linear', is_flag=True)
@click.option('-c', '--cnn', is_flag=True)

def train_gan(linear, cnn):

    #Define device (cuda if it exists)
    # moves your models to train on your gpu if available else it uses your cpu
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #Prepare data
    batch_size = 128
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1207,), (0.3081,))])
    # data_path ="C:/Users/Jeremy UZAN/data"
    data_path ="/home/jeremy/data"


    train_set = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    images, true_labels= next(iter(train_loader))
    print("DATA IS LOADED")

    #If you choose linear architecture
    if linear:
        generator = LinearGenerator(z_dim=100)
        generator.to(device)
        print("MODEL IS READY")

        num_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        print('Number of parameters of the generator: %d' % num_params)

        # Create Discriminator

        discriminator = LinearDiscriminator()
        discriminator.to(device)
        print("MODEL IS READY")

        num_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        print('Number of parameters of the discriminator: %d' % num_params)

        # Create GANTrainer
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        gan_dir = "results/GAN" + timestamp
        os.makedirs(gan_dir, exist_ok=True)
        gan_trainer = GANTrainer(gan_dir, generator, discriminator, train_loader, test_loader,cnn,linear, lr=0.0002)
        print("TRAINER IS READY")

        # train GAN
        print("TRAINING MODEL")

        gan_trainer.train_model(100, device)

    #if you choose convolutional architecture
    if cnn:
        generator= CNNGenerator(z_dim=100)
        generator.to(device)
        print("MODEL IS READY")

        num_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        print('Number of parameters of the generator: %d' % num_params)

        # Create Discriminator

        discriminator = CNNDiscriminator()
        discriminator.to(device)
        print("MODEL IS READY")

        num_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        print('Number of parameters of the discriminator: %d' % num_params)

        # Create GANTrainer
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        gan_dir = "results/GAN" + timestamp
        os.makedirs(gan_dir, exist_ok=True)
        gan_trainer = GANTrainer(gan_dir, generator, discriminator, train_loader, test_loader,cnn,linear, lr=0.0002)
        print("TRAINER IS READY")

        # train GAN
        print("TRAINING MODEL")

        gan_trainer.train_model(100, device)



if __name__ == '__main__':
    train_gan()