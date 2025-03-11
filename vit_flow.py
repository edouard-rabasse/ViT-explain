import numpy as np
import torch
import cv2

def compute_joint_attention(att_mat, add_residual=True):
    print("Computing joint attention...")
    # Average over the heads
    att_mat = att_mat.mean(axis=2)  # Shape becomes [12, 197, 197]

    if add_residual:
        residual_att = np.eye(att_mat.shape[2])  # Shape [197, 197]
        aug_att_mat = att_mat + residual_att[None, ...]  # Broadcast to [12, 197, 197]
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[..., None]
    else:
        aug_att_mat = att_mat

    joint_attentions = np.zeros_like(aug_att_mat)  # Initialize with zeros
    joint_attentions[0] = aug_att_mat[0]

    layers = aug_att_mat.shape[0]
    for i in range(1, layers):
        joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i-1])

    print("Joint attention computed.")
    return joint_attentions

def generate_attention_mask(joint_attentions, discard_ratio, head_fusion):
    print("Generating attention mask...")
    # Sum over all layers to get a single attention mask
    attention_mask = joint_attentions.sum(axis=0)

    # Normalize the mask to the range [0, 1]
    attention_mask = (attention_mask - attention_mask.min()) / (attention_mask.max() - attention_mask.min())

    # Convert to PyTorch tensor for topk operation
    flat = torch.tensor(attention_mask.flatten())
    _, indices = flat.topk(int(flat.size(0) * discard_ratio), largest=False)

    # Set the lowest values to zero
    flat[indices] = 0

    # Reshape back to original shape
    attention_mask = flat.reshape(attention_mask.shape).numpy()

    # Apply Gaussian blur to smooth the mask
    attention_mask = cv2.GaussianBlur(attention_mask, (7, 7), 0)

    print("Attention mask generated.")
    return attention_mask
