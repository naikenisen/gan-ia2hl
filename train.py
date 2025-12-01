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
best_val_loss = float('inf')

# fonction pour calculer la loss du discriminateur
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = criterion_bce(disc_real_output, torch.ones_like(disc_real_output))
    generated_loss = criterion_bce(disc_generated_output, torch.zeros_like(disc_generated_output))
    total_loss = (real_loss + generated_loss) / 2
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
    
    # boule sur les époques pour l'entrainement
    for epoch in range(start_epoch, epochs + 1):
        epoch_counter = epoch
        start = time.time()
        
        # Passage des modèles en mode entraînement (hérité de nn.Module)
        generator.train()
        discriminator.train()
        
        print(f"Epoch {epoch}/{epochs}")
        progress_bar = tqdm(train_loader, desc="Training", unit="batch")
        for input_image, target in progress_bar:
            disc_loss, gen_total_loss, gen_l1_loss = train_step(input_image, target)
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
        
        # Validation après chaque époque
        print("Running validation...")
        val_l1_loss, val_disc_loss = validate(test_loader)
        print(f"Validation - L1 Loss: {val_l1_loss:.4f}, Disc Loss: {val_disc_loss:.4f}")
        
        wandb.log({
            'val_l1_loss': val_l1_loss,
            'val_disc_loss': val_disc_loss,
            'epoch': epoch
        })
        
        # Sauvegarder le meilleur modèle basé sur la L1 loss de validation
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

        print(f"Time taken for epoch {epoch} is {time.time()-start:.2f} sec\n")
        wandb.log({"epoch_time": time.time()-start})

# Start Training
fit(train_loader, test_loader, start_epoch=epoch_counter, epochs=EPOCHS)

wandb.finish()