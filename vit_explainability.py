# based on https://arxiv.org/pdf/2012.09838

import torch
import numpy as np

class VITTransformerExplainability:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean"):
        """
        Initialize the explainer.
        - model: the Vision Transformer model.
        - attention_layer_name: substring to match modules containing attention maps.
        - head_fusion: method for fusing heads (here we use mean).
        """
        self.model = model
        self.attention_layer_name = attention_layer_name
        self.head_fusion = head_fusion
        self.attentions = []
        self.attention_gradients = []
        
        # Register hooks on modules whose name contains the attention layer identifier.
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.forward_hook)
                # Note: backward hooks on modules are deprecated in favor of registering
                # hooks on the outputs. For simplicity, we use register_backward_hook here.
                module.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        # Store the attention maps.
        self.attentions.append(output.cpu())

    def backward_hook(self, module, grad_input, grad_output):
        # grad_output is a tuple; here we assume grad_output[0] is the gradient for the attention map.
        self.attention_gradients.append(grad_output[0].cpu())

    def __call__(self, input_tensor, target_class=None):
        """
        Perform a forward and backward pass to compute the explanation.
        - input_tensor: preprocessed input.
        - target_class: (optional) target class index for which to compute the explanation.
          If not provided, the predicted class is used.
        Returns a normalized spatial mask.
        """
        # Clear previously stored attention maps and gradients.
        self.attentions = []
        self.attention_gradients = []
        
        # Forward pass.
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients and compute backward for the target class.
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        # Ensure that we have collected attention maps and their gradients.
        if len(self.attentions) == 0 or len(self.attention_gradients) == 0:
            raise ValueError("No attention maps or gradients were captured. Check the hook registration.")
        
        # For each attention block, combine the attention with its gradient.
        A_bars = []
        for att, grad in zip(self.attentions, self.attention_gradients):
            # att and grad shape: (batch, num_heads, N, N)
            # Compute the element-wise product: grad ⊙ att.
            combined = att * grad
            # Fuse heads (using mean here).
            if self.head_fusion == "mean":
                fused = combined.mean(dim=1)  # shape: (batch, N, N)
            elif self.head_fusion == "max":
                fused = combined.max(dim=1)[0]
            elif self.head_fusion == "min":
                fused = combined.min(dim=1)[0]
            else:
                raise ValueError("Unknown head fusion method: {}".format(self.head_fusion))
            # Optionally, clip negative values if desired:
            # fused = torch.clamp(fused, min=0)
            # Add the identity matrix to preserve self-connections.
            I = torch.eye(fused.size(-1)).to(fused.device)
            A_bar = I + fused
            A_bars.append(A_bar)
        
        # Multiply the adjusted attention matrices across layers.
        result = A_bars[0]
        for A_bar in A_bars[1:]:
            result = torch.matmul(result, A_bar)
        
        # The result shape is (batch, N, N); select the row corresponding to [CLS] (assumed to be index 0)
        # and ignore the [CLS] column (i.e. take columns 1:).
        mask = result[:, 0, 1:]
        
        # For ViT, the remaining tokens correspond to image patches.
        # Reshape the mask to a square grid (e.g., 14x14 for 196 patches).
        grid_size = int(np.sqrt(mask.size(-1)))
        mask = mask.reshape(mask.size(0), grid_size, grid_size)
        
        # Normalize the mask.
        mask = mask / mask.max()
        return mask.cpu().detach().numpy()[0]
