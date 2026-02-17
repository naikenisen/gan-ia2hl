import os
from pathlib import Path
from PIL import Image

def tile_dataset(src_root, dst_root, tile_size=512):
    # Conversion en objets Path pour faciliter la manipulation
    src_path = Path(src_root).expanduser()
    dst_path = Path(dst_root).expanduser()

    # Extensions d'images supportées
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif')

    print(f"Début du découpage... de {src_path} vers {dst_path}")

    # Parcours récursif de toutes les images du dataset d'origine
    for img_file in src_path.rglob('*'):
        if img_file.suffix.lower() in valid_extensions:
            
            # 1. Recréer la structure des dossiers (ex: train/CD30)
            relative_path = img_file.relative_to(src_path)
            target_dir = dst_path / relative_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)

            # 2. Ouvrir l'image
            with Image.open(img_file) as img:
                width, height = img.size
                
                # Récupérer le nom de base (ex: c_patch_x4000_y30000)
                base_name = img_file.stem 

                # 3. Calculer le nombre de patches possibles
                # On ignore les bords s'ils sont plus petits que 512 (plus propre pour le GAN)
                for y in range(0, height - tile_size + 1, tile_size):
                    for x in range(0, width - tile_size + 1, tile_size):
                        
                        # Définir la zone de découpe (left, top, right, bottom)
                        box = (x, y, x + tile_size, y + tile_size)
                        tile = img.crop(box)

                        # 4. Construire le nouveau nom
                        # Format: c_patch_x4000_y30000_tile_x0_y512.jpg
                        new_name = f"{base_name}_tile_x{x}_y{y}{img_file.suffix}"
                        
                        # Sauvegarder
                        tile.save(target_dir / new_name)

            print(f"Traité : {relative_path}")

if __name__ == "__main__":
    SOURCE = "~/coding/gan-ia2hl/dataset"
    DESTINATION = "~/coding/gan-ia2hl/dataset_tiled_512"
    
    tile_dataset(SOURCE, DESTINATION)
    print("\nTerminé ! Tes images sont découpées et alignées.")