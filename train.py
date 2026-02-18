import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import wandb
import lpips
import subprocess
from src import config
from src.models import Generator, Discriminator
from src.data_loader import create_dataloaders

img_width = config.IMG_WIDTH
img_height = config.IMG_HEIGHT
lrg = config.LRG
lrd = config.LRD
base_hes_path = config.BASE_HES_PATH
base_ihc_path = config.BASE_IHC_PATH
checkpoint_dir = config.CHECKPOINT_DIR
batch_size = config.BATCH_SIZE
epochs = config.EPOCHS
lambda_l1 = config.LAMBDA
model_scale = config.MODEL_SCALE

# Créer les dataloaders avec les arguments
train_loader, valid_loader, test_loader = create_dataloaders(
    img_height,
    img_width,
    batch_size
)

commit = subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode().strip()
best_model_name=f"batch-{batch_size}-scale-{model_scale}-lambda-{lambda_l1}-width-{img_width}-lrg-{lrg}-lrd-{lrd}"

device = config.device
print(f"Using device: {device}")

# ajout des modèles sur le GPU
generator = Generator(model_scale).to(device)
discriminator = Discriminator(model_scale).to(device)
# défnintion des fonctions de loss en utilisant le module nn
criterion_bce = nn.BCEWithLogitsLoss()
criterion_l1 = nn.L1Loss()
# les optimizers prennent les paramètres des deux modèles qu'ils doivent optimiser 
generator_optimizer = optim.Adam(generator.parameters(), lr=lrg, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lrd, betas=(0.5, 0.999))

epoch_counter = 1
best_model_path = os.path.join(checkpoint_dir, f'{best_model_name}.pth')  # Meilleur modèle basé sur LPIPS
best_val_lpips = float('inf')  # Sélection sur LPIPS validation (lower is better)

# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='alex').to(device)

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
    gen_total_loss = gan_loss + (lambda_l1 * l1_loss)
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

# Validation avec LPIPS
def validate_lpips(valid_loader):
    generator.eval()
    discriminator.eval()
    lpips_model.eval()
    total_lpips = 0
    num_samples = 0
    with torch.no_grad():
        for idx, (input_image, target) in enumerate(valid_loader):
            input_image = input_image.to(device)
            target = target.to(device)
            gen_output = generator(input_image)
            lpips_value = lpips_model(gen_output, target)
            total_lpips += lpips_value.mean().item()
            num_samples += 1
    avg_lpips = total_lpips / num_samples
    return avg_lpips

# Training Loop
def fit(train_loader, valid_loader, start_epoch, epochs):

    global epoch_counter, best_val_lpips
    train_gen_losses = []
    val_lpips_values = []
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
        print(f"Train Gen Loss: {avg_gen_loss:.4f}")

        # Stocker les metrics pour le graphique
        train_gen_losses.append(avg_gen_loss)

        # Validation LPIPS à chaque époque
        print("Running LPIPS validation...")
        avg_lpips = validate_lpips(valid_loader)
        print(f"Val LPIPS: {avg_lpips:.4f}")
        val_lpips_values.append(avg_lpips)
        
        # Sauvegarder le meilleur modèle basé sur LPIPS
        if avg_lpips < best_val_lpips:
            best_val_lpips = avg_lpips
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'generator': generator.state_dict(),
                'epoch': epoch,
                'lpips': avg_lpips
            }, best_model_path)
            print(f"Best model saved based on LPIPS (Val LPIPS: {best_val_lpips:.4f})")
    create_loss_plots(epochs_list, train_gen_losses, val_lpips_values)

def create_loss_plots(epochs, train_gen_loss, val_lpips):
    # Créer 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Generator Total Loss
    axes[0].plot(epochs, train_gen_loss, 'b-', label='Train Gen Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Generator Total Loss')
    axes[0].set_title('Training Generator Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # LPIPS Value (validation)
    axes[1].plot(epochs, val_lpips, 'r-', marker='o', label='Val LPIPS', linewidth=2, markersize=8)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('LPIPS Score')
    axes[1].set_title('Validation LPIPS Score (lower is better)')
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
fit(train_loader, valid_loader, start_epoch=epoch_counter, epochs=epochs)