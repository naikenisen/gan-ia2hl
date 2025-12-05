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
import scipy.linalg
from torch.nn import functional as F

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
best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")  # Meilleur modèle basé sur FID
best_val_fid = float('inf')  # Sélection sur FID validation (lower is better)

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

# Fonction pour calculer FID pendant la validation
def calculate_fid_from_images(real_images, generated_images, device):
    """Calculate FID score using InceptionV3 features"""
    from torchvision.models import inception_v3
    
    # Load InceptionV3 model in feature extraction mode
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.fc = nn.Identity()  # Remove final classification layer
    inception_model.eval()
    
    def get_features(images):
        features = []
        with torch.no_grad():
            for img in images:
                # Resize to 299x299 for InceptionV3
                img_resized = F.interpolate(
                    img.unsqueeze(0), size=(299, 299), mode='bilinear', align_corners=False
                )
                # InceptionV3 expects images normalized to [-1, 1] which we already have
                feat = inception_model(img_resized)
                # feat is now a 1D tensor of features from the last pooling layer
                features.append(feat.squeeze().cpu().numpy())
        return np.array(features)
    
    # Get features
    real_features = get_features(real_images)
    gen_features = get_features(generated_images)
    
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(gen_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    # Calculate FID
    diff = mu_real - mu_gen
    covmean = scipy.linalg.sqrtm(sigma_real.dot(sigma_gen))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid

# Validation avec FID
def validate_fid(test_loader, max_samples=50):
    """
    Calcule le FID score sur un sous-ensemble de données de validation
    max_samples: nombre maximum d'échantillons à utiliser pour calculer le FID
    """
    generator.eval()
    discriminator.eval()
    
    real_images_for_fid = []
    generated_images_for_fid = []
    
    with torch.no_grad():
        for idx, (input_image, target) in enumerate(test_loader):
            if idx >= max_samples:
                break
                
            input_image = input_image.to(device)
            target = target.to(device)
            gen_output = generator(input_image)
            
            # Store images for FID calculation (batch size of 1 assumed)
            real_images_for_fid.append(target[0])
            generated_images_for_fid.append(gen_output[0])
    
    if len(real_images_for_fid) > 1:
        fid_score = calculate_fid_from_images(real_images_for_fid, generated_images_for_fid, device)
        return fid_score
    else:
        return float('inf')

# Training Loop
def fit(train_loader, test_loader, start_epoch, epochs, fid_frequency=5):
    """
    Entraîne le modèle et sauvegarde le meilleur basé sur le FID score
    fid_frequency: calculer le FID tous les X epochs (FID est coûteux en calcul)
    """
    global epoch_counter, best_val_fid
    train_gen_losses = []
    val_fid_values = []
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

        # Validation FID (moins fréquent car coûteux)
        calculate_fid_now = (epoch % fid_frequency == 0 or epoch == epochs)
        if calculate_fid_now:
            print("Running FID validation...")
            avg_fid = validate_fid(test_loader, max_samples=50)
            print(f"Val FID: {avg_fid:.4f}")
            val_fid_values.append(avg_fid)
            
            # Sauvegarder le meilleur modèle basé sur FID
            if avg_fid < best_val_fid:
                best_val_fid = avg_fid
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                torch.save({
                    'generator': generator.state_dict(),
                    'epoch': epoch,
                    'fid': avg_fid
                }, best_model_path)
                print(f"Best model saved based on FID (Val FID: {best_val_fid:.4f})")
        else:
            val_fid_values.append(None)  # Placeholder pour les epochs sans calcul FID
    
    create_loss_plots(epochs_list, train_gen_losses, val_fid_values)

def create_loss_plots(epochs, train_gen_loss, val_fid):
    # Filtrer les valeurs FID (enlever les None)
    fid_epochs = [e for e, f in zip(epochs, val_fid) if f is not None]
    fid_values = [f for f in val_fid if f is not None]
    
    # Créer 2 subplots
    has_fid = len(fid_values) > 0
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Generator Total Loss
    axes[0].plot(epochs, train_gen_loss, 'b-', label='Train Gen Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Generator Total Loss')
    axes[0].set_title('Training Generator Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # FID Value (validation)
    if has_fid:
        axes[1].plot(fid_epochs, fid_values, 'g-', marker='o', label='Val FID', linewidth=2, markersize=8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('FID Score')
        axes[1].set_title('Validation FID Score (lower is better)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No FID data yet', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Validation FID Score')
    
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