from src.config import *
import torch
import torch.optim as optim
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import wandb
import torch.nn as nn
from src.models import Generator, Discriminator
from src.data_loader_regions import train_loader, test_loader
from pytorch_msssim import ssim

wandb.login(key="ab67e0f4c27fad7a0d47405f84a8a4deb80056ba")

wandb.init(
    project="ia2hl-gan"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ajout des modèles sur le GPU
generator = Generator(MODEL_SCALE).to(device)
discriminator = Discriminator(MODEL_SCALE).to(device)
# défnintion des fonctions de loss en utilisant le module nn
criterion_bce = nn.BCEWithLogitsLoss()
criterion_l1 = nn.L1Loss()
# les optimizers prennent les paramètres des deux modèles qu'ils doivent optimiser 
generator_optimizer = optim.Adam(generator.parameters(), lr=LRG, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LRD, betas=(0.5, 0.999))

epoch_counter = 1
best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
last_model_path = os.path.join(CHECKPOINT_DIR, "last_model.pth")
best_val_l1 = float('inf')  # Sélection sur L1 validation (standard)

# fonction pour calculer la loss du discriminateur
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = criterion_bce(disc_real_output, torch.ones_like(disc_real_output))
    generated_loss = criterion_bce(disc_generated_output, torch.zeros_like(disc_generated_output))
    disc_total_loss = real_loss + generated_loss
    return disc_total_loss

# fonction pour calculer la loss du générateur avec L1
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = criterion_bce(disc_generated_output, torch.ones_like(disc_generated_output))
    l1_loss = criterion_l1(gen_output, target)
    # Pix2Pix standard : GAN loss + λ * L1 loss (typiquement λ=100)
    gen_total_loss = gan_loss + (LAMBDA * l1_loss)
    return gen_total_loss

def train_step(input_image, target):
    # ajout des data sur le GPU
    input_image = input_image.to(device)
    target = target.to(device)
    # Train Discriminator
    discriminator_optimizer.zero_grad()
    gen_output = generator(input_image)
    disc_real_output = discriminator(input_image, target)
    disc_generated_output = discriminator(input_image, gen_output.detach())
    disc_total_loss = discriminator_loss(disc_real_output, disc_generated_output)
    disc_total_loss.backward()
    discriminator_optimizer.step()
    # Train Generator
    generator_optimizer.zero_grad()
    gen_output = generator(input_image)
    disc_generated_output = discriminator(input_image, gen_output)
    gen_total_loss = generator_loss(disc_generated_output, gen_output, target)
    gen_total_loss.backward()
    generator_optimizer.step()
    return gen_total_loss.item()

# Validation avec L1 (standard pour sélectionner le meilleur modèle)
def validate(test_loader):
    generator.eval()
    discriminator.eval()
    total_l1 = 0
    num_batches = 0
    with torch.no_grad():
        for input_image, target in test_loader:
            input_image = input_image.to(device)
            target = target.to(device)
            gen_output = generator(input_image)
            
            # L1 loss pour validation (standard en image-to-image translation)
            l1_val = criterion_l1(gen_output, target)
            total_l1 += l1_val.item()
            num_batches += 1
    
    avg_l1 = total_l1 / num_batches
    return avg_l1

# Training Loop
def fit(train_loader, test_loader, start_epoch, epochs):
    global epoch_counter, best_val_l1
    train_gen_losses = []
    val_l1_values = []
    epochs_list = []
    
    for epoch in range(start_epoch, epochs + 1):
        epoch_counter = epoch
        epochs_list.append(epoch)
        start = time.time()
        
        generator.train()
        discriminator.train()
        print(f"Epoch {epoch}/{epochs}")
        progress_bar = tqdm(train_loader, desc="Training", unit="batch")
        
        epoch_gen_loss = 0
        num_train_batches = 0
        
        for input_image, target in progress_bar:
            batch_gen_loss = train_step(input_image, target)
            epoch_gen_loss += batch_gen_loss
            num_train_batches += 1
            progress_bar.set_postfix({
                'gen_loss': f'{batch_gen_loss:.4f}',
            })
        
        # Calculer la moyenne de l'époque
        avg_gen_loss = epoch_gen_loss / num_train_batches
        
        # Validation après chaque époque
        print("Running validation...")
        avg_l1 = validate(test_loader)
        print(f"Train Gen Loss: {avg_gen_loss:.4f}, Val L1: {avg_l1:.4f}")

        # Stocker les metrics pour le graphique
        train_gen_losses.append(avg_gen_loss)
        val_l1_values.append(avg_l1)

        # Sauvegarder le meilleur modèle basé sur L1 validation (standard en image translation)
        if avg_l1 < best_val_l1:
            best_val_l1 = avg_l1
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                'generator': generator.state_dict(),
            }, best_model_path)
            print(f"Best model saved (Val L1: {best_val_l1:.4f})")
        
        # Sauvegarder le dernier modèle à chaque époque
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save({
            'generator': generator.state_dict(),
        }, last_model_path)
        print(f"Last model saved")
    
    create_loss_plots(epochs_list, train_gen_losses, val_l1_values)

def create_loss_plots(epochs, train_gen_loss, val_l1):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Generator Total Loss
    axes[0].plot(epochs, train_gen_loss, 'b-', label='Train Gen Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Generator Total Loss')
    axes[0].set_title('Training Generator Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # L1 Value (validation)
    axes[1].plot(epochs, val_l1, 'r-', label='Val L1', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('L1 Loss')
    axes[1].set_title('Validation L1 Loss (lower is better)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plot_path = os.path.join('results', 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Metrics plots saved to {plot_path}")
    wandb.log({"training_metrics_plot": wandb.Image(plot_path)})
    plt.close()

# Start Training
fit(train_loader, test_loader, start_epoch=epoch_counter, epochs=EPOCHS)

wandb.finish()