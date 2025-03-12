# based on https://arxiv.org/pdf/2012.09838
# This is a simplified version. We did'nt implement the LRP method for the final layer.


import torch
import numpy as np
import cv2

class VITTransformerLRPmimic:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean"):
        """
        Initialize the LRP-based explainer.
        
        Args:
            model: The Vision Transformer model.
            attention_layer_name: Substring to match modules containing attention maps.
            head_fusion: Method to fuse heads (e.g., "mean", "max", "min").
        """
        self.model = model
        self.attention_layer_name = attention_layer_name
        self.head_fusion = head_fusion
        self.attentions = []
        self.attention_gradients = []
        
        # Register forward and backward hooks on all modules whose name contains the attention_layer_name.
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.forward_hook)
                module.register_full_backward_hook(self.backward_hook)
                
    def forward_hook(self, module, input, output):
        # Store the attention maps.
        self.attentions.append(output)
        
    def backward_hook(self, module, grad_input, grad_output):
        # Store gradients from the attention module.
        # grad_output is a tuple; we assume grad_output[0] contains the gradients.
        self.attention_gradients.append(grad_output[0])
        
    def __call__(self, input_tensor, target_class=None):
        """
        Compute the LRP-based explanation.
        
        Args:
            input_tensor: Preprocessed input tensor.
            target_class: (Optional) Target class index. If None, uses the predicted class.
        
        Returns:
            A spatial mask (numpy array) representing the explanation.
        """
        # Clear stored hooks (if any) from previous calls.
        self.attentions = []
        self.attention_gradients = []
        
        # Forward pass.
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Backward pass with respect to the target class.
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # List to hold the adjusted attention matrices for each attention block.
        A_bars = []
        for att, grad in zip(self.attentions, self.attention_gradients):
            # Assume attention maps and gradients are of shape: (batch, num_heads, N, N)
            # Compute element-wise product and clamp negative values (i.e. keep only positive contributions).
            combined = torch.clamp(att * grad, min=0)
            
            # Fuse across heads.
            if self.head_fusion == "mean":
                fused = combined.mean(dim=1)  # shape: (batch, N, N)
            elif self.head_fusion == "max":
                fused = combined.max(dim=1)[0]
            elif self.head_fusion == "min":
                fused = combined.min(dim=1)[0]
            else:
                raise ValueError("Unknown head fusion method: {}".format(self.head_fusion))
            
            # Add identity to incorporate skip connections (ensuring self-attention is preserved).
            I = torch.eye(fused.size(-1)).to(fused.device)
            A_bar = I + fused
            
            # Normalize each row (to mimic the conservation rule).
            # Adding a small epsilon to avoid division by zero.
            epsilon = 1e-6
            A_bar = A_bar / (A_bar.sum(dim=-1, keepdim=True) + epsilon)
            A_bars.append(A_bar)
        
        # Rollout: Multiply the adjusted attention matrices across layers.
        result = A_bars[0]
        for A_bar in A_bars[1:]:
            result = torch.matmul(result, A_bar)
        
        # Extract the explanation from the row corresponding to the [CLS] token (assumed to be at index 0)
        # and discard the [CLS] column.
        mask = result[:, 0, 1:]
        
        # Reshape the mask to a square grid.
        grid_size = int(np.sqrt(mask.size(-1)))
        mask = mask.reshape(mask.size(0), grid_size, grid_size)
        
        # Normalize the mask between 0 and 1.
        mask = mask / (mask.max() + epsilon)
        return mask.cpu().detach().numpy()[0]

# Example helper function to overlay a heatmap on an image.
def show_cam_on_image(img, mask):
    """
    Superimpose the mask (heatmap) on the image.
    
    Args:
        img: Original image in range [0,1] (H x W x 3).
        mask: Spatial mask (H x W) in range [0,1].
    
    Returns:
        The image with the heatmap overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
