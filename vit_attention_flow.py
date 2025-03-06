import networkx as nx
import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2

def get_adjmat(att_mat, input_tokens):
    n_layers, length, _ = att_mat.shape
    adj_mat = np.zeros(((n_layers + 1) * length, (n_layers + 1) * length))
    labels_to_index = {}
    
    # Construction du dictionnaire de labels
    for k in np.arange(length):
        labels_to_index[str(k) + "_" + input_tokens[k]] = k

    # Remplissage de la matrice d'adjacence
    for i in np.arange(1, n_layers + 1):
        for k_f in np.arange(length):
            index_from = (i) * length + k_f
            label = "L" + str(i) + "_" + str(k_f)
            labels_to_index[label] = index_from
            for k_t in np.arange(length):
                index_to = (i - 1) * length + k_t
                adj_mat[index_from][index_to] = att_mat[i - 1][k_f][k_t]

    return adj_mat, labels_to_index

def get_attention_flow(adjmat, labels_to_index, input_nodes, length):
    """
    Calcule les flux d'attention à partir de la matrice d'adjacence
    et des labels associés.
    """
    flow_values = np.zeros((len(labels_to_index), len(labels_to_index)))
    
    # Créer un graph dirigé à partir de la matrice d'adjacence
    G = nx.from_numpy_matrix(adjmat, create_using=nx.DiGraph())

    # Boucler sur les nœuds et calculer les flux
    for label, u in labels_to_index.items():
        if u not in input_nodes:
            for v in input_nodes:
                flow_value = nx.maximum_flow_value(G, u, v, flow_func=nx.algorithms.flow.edmonds_karp)
                flow_values[u, v] = flow_value

    return flow_values

class VITAttentionFlow:
    def __init__(self, model, discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())
        print(f"Attention capturée : {output.shape}")

    def attach_attention_hook(self):
        for name, module in self.model.named_modules():
            if 'attn' in name:
                module.register_forward_hook(self.get_attention)
                print(f"Hook attaché à: {name}")

    def compute_attention_flow(self, attentions, input_tokens):
        adjmat, labels_to_index = get_adjmat(attentions[0], input_tokens)
        input_nodes = ['class']
        flow_values = get_attention_flow(adjmat, labels_to_index, input_nodes, len(input_tokens))
        return flow_values

    def __call__(self, input_tensor, input_tokens, np_img):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)
            if not self.attentions:
                print("Aucune attention capturée après l'inférence.")
                return None

        flow_values = self.compute_attention_flow(self.attentions, input_tokens)
        mask = flow_values[0, 1:]

        # Vérification du masque
        print("Mask shape:", mask.shape)
        if mask.size == 0:
            raise ValueError("Le masque est vide. Vérifie les flux d'attention.")

        mask = mask / np.max(mask)  # Normalisation

        # Vérification de l'image np_img avant redimensionnement
        if np_img.size == 0:
            raise ValueError("L'image d'entrée est vide.")
        print("Image dimensions:", np_img.shape)

        # Redimensionnement avec une vérification des dimensions
        try:
            mask_resized = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        except cv2.error as e:
            raise ValueError(f"Erreur lors du redimensionnement: {e}")

        return mask_resized
