import torch
import torch.nn as nn

def load_deit(weights_path):
    """
    Charge le modèle tiny DEIT de Facebook, modifie sa dernière couche (head)
    pour qu'elle ait deux sorties, et y affecte les poids contenus dans le fichier
    spécifié par weights_path.

    Parameters:
        weights_path (str): Chemin vers le fichier contenant les poids pour la tête.

    Returns:
        model: Le modèle DEIT modifié avec la tête à 2 sorties et les poids chargés.
    """
    # Charger le modèle pré-entraîné depuis torch.hub
    model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
    
    # Modifier la couche de classification (head) pour avoir 2 sorties.
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, 2)
    
    # Charger les poids personnalisés pour la nouvelle tête depuis le fichier
    head_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.head.load_state_dict(head_state_dict)
    
    return model


## Exemple d'utilisation
'''
from load_deit import load_deit
weights_path = 'deit_tiny_head_weights.pth'
model = load_deit(weights_path)
print(model)
'''
# Output