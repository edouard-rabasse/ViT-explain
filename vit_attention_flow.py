import networkx as nx
import numpy as np
import torch

def get_adjmat(att_mat, input_tokens):
    """
    Crée la matrice d'adjacence à partir de la matrice d'attention et des tokens d'entrée.
    """
    n_layers, length, _ = att_mat.shape
    adj_mat = np.zeros(((n_layers + 1) * length, (n_layers + 1) * length))
    labels_to_index = {}
    
    # Ajout des tokens d'entrée
    for k in np.arange(length):
        labels_to_index[str(k) + "_" + input_tokens[k]] = k

    # Pour chaque couche, associer chaque token avec son index
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
        if label in input_nodes:  # Vérifiez que le label est bien dans les nœuds d'entrée
            for v in labels_to_index.values():
                flow_value = nx.maximum_flow_value(G, u, v, flow_func=nx.algorithms.flow.edmonds_karp)
                flow_values[u, v] = flow_value

    return flow_values

class VITAttentionFlow:
    def __init__(self, model, discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        self.attentions = []

    def get_attention(self, module, input, output):
        """
        Récupère les matrices d'attention à partir du modèle.
        """
        self.attentions.append(output.cpu())

    def compute_attention_flow(self, attentions, input_tokens):
        """
        Cette méthode va calculer les flux d'attention sur la base des matrices d'attention.
        """
        # Obtenir la matrice d'adjacence et les indices des labels
        adjmat, labels_to_index = get_adjmat(attentions[0], input_tokens)

        # Définir les tokens d'entrée, ici la première position est la classe
        input_nodes = ['class']

        # Calcul des flux d'attention
        flow_values = get_attention_flow(adjmat, labels_to_index, input_nodes, len(input_tokens))

        return flow_values

    def __call__(self, input_tensor, input_tokens):
        """
        Calcule les flux d'attention pour une image ou un tensor d'entrée.
        """
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        # Calculer les flux d'attention
        flow_values = self.compute_attention_flow(self.attentions, input_tokens)

        # Récupérer le masque de l'attention (flux vers les patches)
        mask = flow_values[0, 1:]  # On suppose que la classe est le premier élément
        mask = mask / np.max(mask)  # Normalisation du masque

        return mask