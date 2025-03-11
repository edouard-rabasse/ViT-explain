import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

def get_adjmat(mat, input_tokens):
    n_layers, length, _ = mat.shape
    adj_mat = np.zeros(((n_layers+1)*length, (n_layers+1)*length))
    labels_to_index = {f"{k}_{input_tokens[k]}": k for k in range(length)}

    for i in range(1, n_layers+1):
        for k_f in range(length):
            index_from = (i)*length+k_f
            label = f"L{i}_{k_f}"
            labels_to_index[label] = index_from
            for k_t in range(length):
                index_to = (i-1)*length+k_t
                adj_mat[index_from][index_to] = mat[i-1][k_f][k_t]

    return adj_mat, labels_to_index

def compute_flow(G, u, v):
    try:
        return nx.maximum_flow_value(G, u, v, flow_func=nx.algorithms.flow.edmonds_karp)
    except nx.NetworkXUnbounded:
        return 0

def compute_flows_parallel(G, labels_to_index, input_nodes, length):
    number_of_nodes = len(labels_to_index)
    flow_values = np.zeros((number_of_nodes, number_of_nodes))

    with ThreadPoolExecutor() as executor:
        futures = []
        for key in labels_to_index:
            if key not in input_nodes:
                current_layer = labels_to_index[key] // length
                pre_layer = current_layer - 1
                u = labels_to_index[key]
                for inp_node_key in input_nodes:
                    v = labels_to_index[inp_node_key]
                    futures.append(executor.submit(compute_flow, G, u, v))

        for future, (u, v) in zip(futures, [(labels_to_index[key], labels_to_index[inp_node_key]) for key in labels_to_index if key not in input_nodes for inp_node_key in input_nodes]):
            flow_value = future.result()
            flow_values[u][pre_layer*length+v] = flow_value
            if flow_values[u].sum() > 0:
                flow_values[u] /= flow_values[u].sum()

    return flow_values

class VITAttentionGraph:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
        discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output)

    def __call__(self, input_tensor, input_tokens):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        att_mat = torch.stack(self.attentions).squeeze(1)
        att_mat = att_mat.mean(dim=1)
        att_mat = att_mat.cpu().numpy()

        # Ignorer le jeton de classification
        att_mat = att_mat[:, 1:, 1:]

        adj_mat, labels_to_index = get_adjmat(att_mat, input_tokens)
        G = nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph())
        input_nodes = [f"{i}_{input_tokens[i]}" for i in range(len(input_tokens))]
        flow_values = compute_flows_parallel(G, labels_to_index, input_nodes, att_mat.shape[1])

        # Retourne la matrice d'attention moyenne pour la heatmap
        return np.mean(flow_values, axis=0).reshape(14, 14)
