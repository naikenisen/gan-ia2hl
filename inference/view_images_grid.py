import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Charger les images
img1 = mpimg.imread('inference/hes.jpg')
img2 = mpimg.imread('inference/cd30.jpg')
img3 = mpimg.imread('inference/inference.png')

# Créer la figure avec 3 sous-plots côte à côte
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Espacer légèrement les images
plt.subplots_adjust(wspace=0.05)

# Paramètres du quadrillage
grid_spacing = 10  # Nombre de lignes/colonnes dans le quadrillage

# Afficher les images et ajouter le quadrillage
for idx, ax in enumerate(axes):
    if idx == 0:
        ax.imshow(img1)
    elif idx == 1:
        ax.imshow(img2)
    else:
        ax.imshow(img3)
    
    # Obtenir les dimensions de l'image affichée
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Créer le quadrillage
    x_step = (xlim[1] - xlim[0]) / grid_spacing
    y_step = (ylim[0] - ylim[1]) / grid_spacing
    
    # Lignes verticales
    for i in range(grid_spacing + 1):
        x = xlim[0] + i * x_step
        ax.axvline(x=x, color='white', linewidth=0.8, alpha=0.8)
    
    # Lignes horizontales
    for i in range(grid_spacing + 1):
        y = ylim[1] + i * y_step
        ax.axhline(y=y, color='white', linewidth=0.8, alpha=0.8)
    
    # Ajouter les coordonnées sur les axes
    # Coordonnées X (en haut)
    x_positions = [xlim[0] + (i + 0.5) * x_step for i in range(grid_spacing)]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(range(grid_spacing), fontsize=10, color='black', weight='bold')
    ax.tick_params(axis='x', length=0, pad=5, color='black', labelcolor='black')
    ax.xaxis.tick_top()
    
    # Coordonnées Y (à gauche)
    y_positions = [ylim[1] + (i + 0.5) * y_step for i in range(grid_spacing)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(range(grid_spacing), fontsize=10, color='black', weight='bold')
    ax.tick_params(axis='y', length=0, pad=5, color='black', labelcolor='black')
    
    # S'assurer que les axes sont visibles
    ax.spines['top'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

# Supprimer les marges
plt.tight_layout(pad=0.5)

# Afficher la figure
plt.savefig('inference/view_images_grid.png', dpi=300)
