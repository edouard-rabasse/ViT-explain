# Instructions pour charger et préparer les données pour l’entraînement du modèle

Ce document décrit les étapes nécessaires pour télécharger, préparer et charger les données issues du dataset *covidqu* afin de les utiliser pour l’entraînement d’un modèle de classification. Dans cet exemple, nous avons choisi d’utiliser le dossier **Lung Segmentation Data**.

---


!pip install --quiet torchvision

!pip install kaggle

!kaggle datasets download -d anasmohammedtahir/covidqu

import zipfile
with zipfile.ZipFile('covidqu.zip', 'r') as zip_ref:
    zip_ref.extractall('covidqu')
import os


dataset_root = "/content/covidqu"
print_directory_tree_dirs(dataset_root)

self.split_dir = os.path.join(self.rootpath,
                              "Lung Segmentation Data",
                              "Lung Segmentation Data",
                              self.mode)

import torchvision.transforms as T

self.transform = T.Compose([
    T.ToPILImage(),
    T.Resize(self.cropsize),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

from torch.utils.data import Dataset
import cv2
import os
import torch
import torchvision.transforms as T

class InfectionBinaryDataset(Dataset):
    def __init__(self, rootpath, mode='Train', cropsize=(224,224)):
        """
        Args:
            rootpath (str): Chemin vers le dossier racine du dataset (ex: "/content/covidqu")
            mode (str): "Train", "Val" ou "Test"
            cropsize (tuple): taille de redimensionnement des images, ici (224,224)
        """
        self.rootpath = rootpath
        self.mode = mode  # "Train", "Val" ou "Test" (avec majuscule)
        self.cropsize = cropsize

        # Utilisation du dossier "Lung Segmentation Data"
        self.split_dir = os.path.join(self.rootpath,
                                      "Lung Segmentation Data",
                                      "Lung Segmentation Data",
                                      self.mode)

        # On ne garde que les classes "COVID-19" et "Normal"
        self.classes_to_use = ["COVID-19", "Normal"]
        # Mapping : "COVID-19" -> 1, "Normal" -> 0
        self.label_map = {"COVID-19": 1, "Normal": 0}

        # Création de la liste des échantillons : (chemin complet de l'image, label)
        self.samples = []
        for cls in self.classes_to_use:
            cls_img_dir = os.path.join(self.split_dir, cls, "images")
            if not os.path.isdir(cls_img_dir):
                continue
            img_files = sorted([f for f in os.listdir(cls_img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
            for img in img_files:
                self.samples.append((os.path.join(cls_img_dir, img), self.label_map[cls]))

        # Mélanger l'ordre des échantillons
        import random
        random.shuffle(self.samples)

        # Définition de la transformation (conversion en PIL, redimensionnement, mise en tenseur et normalisation)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.cropsize),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)
        return img, label

from torch.utils.data import DataLoader

dataset_root = "/content/covidqu"

# Création des datasets pour les splits Train, Val et Test
data_train = InfectionBinaryDataset(rootpath=dataset_root, mode="Train", cropsize=(224,224))
data_val   = InfectionBinaryDataset(rootpath=dataset_root, mode="Val", cropsize=(224,224))
data_test  = InfectionBinaryDataset(rootpath=dataset_root, mode="Test", cropsize=(224,224))

# Création des DataLoaders (exemple avec un batch_size de 64)
batch_size = 64
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(data_val, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(data_test, batch_size=batch_size, shuffle=False)
