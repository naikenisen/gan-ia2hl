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

wandb.login(key="ab67e0f4c27fad7a0d47405f84a8a4deb80056ba")


wandb.init(
    project="ia2hl-gan",
    config={
        "base_hande_path": BASE_HandE_PATH,
        "base_ihc_path": BASE_IHC_PATH,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "img_width": IMG_WIDTH,
        "img_height": IMG_HEIGHT,
        "epochs": EPOCHS,
        "lambda_l1": LAMBDA,
        "model_scale": MODEL_SCALE,
        "checkpoint_dir": CHECKPOINT_DIR,
        "learning_rate_generator": LRG,
        "learning_rate_discriminator": LRD,
        "device": str(device),
    }
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
best_val_loss = float('inf')

# fonction pour calculer la loss du discriminateur
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = criterion_bce(disc_real_output, torch.ones_like(disc_real_output))
    generated_loss = criterion_bce(disc_generated_output, torch.zeros_like(disc_generated_output))
    total_loss = (real_loss + generated_loss) / 2 # TODO essayer de voir avant et après division par 2
    return total_loss
# fonction pour calculer la loss du générateur
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = criterion_bce(disc_generated_output, torch.ones_like(disc_generated_output))
    l1_loss = criterion_l1(gen_output, target)
    return gan_loss + (LAMBDA * l1_loss), l1_loss

def train_step(input_image, target):
    # ajout des data sur le GPU
    input_image = input_image.to(device)
    target = target.to(device)

    # Train Discriminator
    discriminator_optimizer.zero_grad()
    gen_output = generator(input_image) # générer une image IHC à partir de l'image HandE
    disc_real_output = discriminator(input_image, target) # passe l'image HandE et l'image IHC réelle au discriminateur
    disc_generated_output = discriminator(input_image, gen_output.detach()) # passe l'image HandE et l'image IHC générée au discriminateur
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output) # calcul de la loss du discriminateur
    disc_loss.backward() # backpropagation sur la loss du discriminateur
    discriminator_optimizer.step() # mise à jour des poids du discriminateur (gradient descent)
    
    # Train Generator
    generator_optimizer.zero_grad()
    gen_output = generator(input_image) # générer une image IHC à partir de l'image HandE
    disc_generated_output = discriminator(input_image, gen_output) # passe l'image HandE et l'image IHC générée au discriminateur
    gen_total_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target) # calcul de la loss du générateur
    gen_total_loss.backward() # backpropagation sur la loss du générateur
    generator_optimizer.step() # mise à jour des poids du générateur (gradient descent)
    return disc_loss.item(), gen_total_loss.item(), gen_l1_loss.item()

# Validation step
def validate(test_loader):
    generator.eval()
    discriminator.eval()
    
    total_l1_loss = 0
    total_disc_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for input_image, target in test_loader:
            input_image = input_image.to(device)
            target = target.to(device)
            
            # Génération d'image
            gen_output = generator(input_image)
            
            # Calculer uniquement la L1 loss (reconstruction quality)
            l1_loss = criterion_l1(gen_output, target)
            
            # Calculer la loss du discriminateur (optionnel mais informatif)
            disc_real_output = discriminator(input_image, target)
            disc_generated_output = discriminator(input_image, gen_output)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
            
            total_l1_loss += l1_loss.item()
            total_disc_loss += disc_loss.item()
            num_batches += 1
    
    avg_l1_loss = total_l1_loss / num_batches
    avg_disc_loss = total_disc_loss / num_batches
    
    return avg_l1_loss, avg_disc_loss

# Training Loop
def fit(train_loader, test_loader, start_epoch, epochs):
    global epoch_counter, best_val_loss
    
    # Listes pour stocker les losses
    train_disc_losses = []
    train_gen_losses = []
    train_l1_losses = []
    val_l1_losses = []
    val_disc_losses = []
    epochs_list = []
    
    # boule sur les époques pour l'entrainement
    for epoch in range(start_epoch, epochs + 1):
        epoch_counter = epoch
        start = time.time()
        
        # Passage des modèles en mode entraînement (hérité de nn.Module)
        generator.train()
        discriminator.train()
        
        print(f"Epoch {epoch}/{epochs}")
        progress_bar = tqdm(train_loader, desc="Training", unit="batch")
        
        # Accumulateurs pour les losses moyennes de l'époque
        epoch_disc_loss = 0
        epoch_gen_loss = 0
        epoch_l1_loss = 0
        num_train_batches = 0
        
        for input_image, target in progress_bar:
            disc_loss, gen_total_loss, gen_l1_loss = train_step(input_image, target)
            
            # Accumuler les losses
            epoch_disc_loss += disc_loss
            epoch_gen_loss += gen_total_loss
            epoch_l1_loss += gen_l1_loss
            num_train_batches += 1
            
            progress_bar.set_postfix({
                'disc_loss': f'{disc_loss:.4f}',
                'gen_total': f'{gen_total_loss:.4f}',
                'l1_loss': f'{gen_l1_loss:.4f}'
            })
            wandb.log({
                'disc_loss': disc_loss,
                'gen_total_loss': gen_total_loss,
                'gen_l1_loss': gen_l1_loss
            })
        
        # Calculer les moyennes de l'époque
        avg_train_disc_loss = epoch_disc_loss / num_train_batches
        avg_train_gen_loss = epoch_gen_loss / num_train_batches
        avg_train_l1_loss = epoch_l1_loss / num_train_batches
        
        # Validation après chaque époque
        print("Running validation...")
        val_l1_loss, val_disc_loss = validate(test_loader)
        print(f"Validation - L1 Loss: {val_l1_loss:.4f}, Disc Loss: {val_disc_loss:.4f}")
        
        # Stocker les losses pour le graphique
        epochs_list.append(epoch)
        train_disc_losses.append(avg_train_disc_loss)
        train_gen_losses.append(avg_train_gen_loss)
        train_l1_losses.append(avg_train_l1_loss)
        val_l1_losses.append(val_l1_loss)
        val_disc_losses.append(val_disc_loss)
        
        wandb.log({
            'val_l1_loss': val_l1_loss,
            'val_disc_loss': val_disc_loss,
            'epoch': epoch,
            'avg_train_disc_loss': avg_train_disc_loss,
            'avg_train_gen_loss': avg_train_gen_loss,
            'avg_train_l1_loss': avg_train_l1_loss
        })
        
        # Sauvegarder le meilleur modèle basé sur la L1 loss de validation
        # TODO "img flou": https://doi.org/10.1038/s41746-025-01741-9 tester métrique composite (L1, PSNR, SSIM, FID)
        if val_l1_loss < best_val_loss: 
            best_val_loss = val_l1_loss
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'generator_optimizer': generator_optimizer.state_dict(),
                'discriminator_optimizer': discriminator_optimizer.state_dict(),
                'val_l1_loss': best_val_loss,
            }, best_model_path)
            print(f"New best model saved! Val L1 Loss: {best_val_loss:.4f} (epoch {epoch})")
            wandb.save(best_model_path)
        
        # Sauvegarder le dernier modèle à chaque époque
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'generator_optimizer': generator_optimizer.state_dict(),
            'discriminator_optimizer': discriminator_optimizer.state_dict(),
            'val_l1_loss': val_l1_loss,
            'best_val_l1_loss': best_val_loss,
        }, last_model_path)
        print(f"Last model saved (epoch {epoch})")

        print(f"Time taken for epoch {epoch} is {time.time()-start:.2f} sec\n")
        wandb.log({"epoch_time": time.time()-start})
    
    # Créer et sauvegarder les graphiques de losses
    create_loss_plots(epochs_list, train_disc_losses, train_gen_losses, train_l1_losses, 
                     val_l1_losses, val_disc_losses)

def create_loss_plots(epochs, train_disc, train_gen, train_l1, val_l1, val_disc):
    """Crée des graphiques pour visualiser l'évolution des losses"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: L1 Loss (Train vs Val)
    axes[0, 0].plot(epochs, train_l1, 'b-', label='Train L1 Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_l1, 'r-', label='Val L1 Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('L1 Loss')
    axes[0, 0].set_title('L1 Loss - Train vs Validation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Discriminator Loss (Train vs Val)
    axes[0, 1].plot(epochs, train_disc, 'b-', label='Train Disc Loss', linewidth=2)
    axes[0, 1].plot(epochs, val_disc, 'r-', label='Val Disc Loss', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Discriminator Loss')
    axes[0, 1].set_title('Discriminator Loss - Train vs Validation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Generator Total Loss
    axes[1, 0].plot(epochs, train_gen, 'g-', label='Train Gen Total Loss', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Generator Total Loss')
    axes[1, 0].set_title('Generator Total Loss (GAN + λ*L1)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Toutes les losses de validation
    axes[1, 1].plot(epochs, val_l1, 'r-', label='Val L1 Loss', linewidth=2)
    axes[1, 1].plot(epochs, val_disc, 'orange', label='Val Disc Loss', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('All Validation Losses')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    os.makedirs('results', exist_ok=True)
    plot_path = os.path.join('results', 'training_losses.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Loss plots saved to {plot_path}")
    
    # Log vers WandB
    wandb.log({"training_losses_plot": wandb.Image(plot_path)})
    
    plt.close()

# Start Training
fit(train_loader, test_loader, start_epoch=epoch_counter, epochs=EPOCHS)

wandb.finish()