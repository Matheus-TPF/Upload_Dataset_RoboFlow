import random
import shutil
from pathlib import Path

# ==========================
# CONFIGURAÇÃO
# ==========================

# Pasta onde estão as imagens
IMAGES_DIR = Path("rgb")

# Pasta de saída
OUTPUT_DIR = Path("datargb")

TRAIN = 0.70
VALID = 0.20
TEST = 0.10

# Reprodutibilidade
random.seed(42)

# ==========================

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Cria as pastas
for split in ["train", "valid", "test"]:
    (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)

# Lista todas as imagens
images = [
    img for img in IMAGES_DIR.iterdir()
    if img.is_file() and img.suffix.lower() in IMG_EXTS
]

random.shuffle(images)

total = len(images)

n_train = int(total * TRAIN)
n_valid = int(total * VALID)

train_imgs = images[:n_train]
valid_imgs = images[n_train:n_train + n_valid]
test_imgs = images[n_train + n_valid:]


def copiar(imagens, split):
    for img in imagens:
        destino = OUTPUT_DIR / split / "images" / img.name
        shutil.copy2(img, destino)

    print(f"{split:<5} -> {len(imagens)} imagens")


copiar(train_imgs, "train")
copiar(valid_imgs, "valid")
copiar(test_imgs, "test")

print("\n===================================")
print("Dataset organizado com sucesso!")
print(f"Total de imagens: {total}")
print(f"Train: {len(train_imgs)}")
print(f"Valid: {len(valid_imgs)}")
print(f"Test : {len(test_imgs)}")
print(f"Saída: {OUTPUT_DIR}")
print("===================================")