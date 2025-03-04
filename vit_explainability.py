# based on https://arxiv.org/pdf/2012.09838

import torch
import numpy as np

class VITTransformerExplainability:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean"):
        """
        Implements the LRP-based relevance propagation for a Transformer model,
        following the paper’s equations.

        Args:
            model: The Vision Transformer model.
            attention_layer_name: Substring to match modules that output attention maps.
            head_fusion: Method to fuse relevance across heads ("mean", "max", or "min").
        """
        self.model = model
        self.attention_layer_name = attention_layer_name
        self.head_fusion = head_fusion
        self.epsilon = 1e-6  # small constant to avoid division by zero
        
        # Lists to store attention maps and their gradients across layers.
        self.attentions = []
        self.attention_gradients = []
        
        # Register forward and backward hooks on modules whose names contain attention_layer_name.
        # This step collects A(b) and ∇A(b) for each block (matching paper notation).
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.forward_hook)
                module.register_full_backward_hook(self.backward_hook)
    
    def forward_hook(self, module, input, output):
        # Equation: We are capturing the attention map A(b) for this layer.
        self.attentions.append(output.detach())
    
    def backward_hook(self, module, grad_input, grad_output):
        # Equation: Capture the gradient ∇A(b) for the attention map.
        # grad_output is a tuple; here we assume grad_output[0] contains ∇A(b).
        self.attention_gradients.append(grad_output[0].detach())
    
    def __call__(self, input_tensor, target_class=None):
        """
        Computes the LRP-based class-specific explanation.

        Returns a spatial mask (numpy array) that highlights the contribution
        of each input token (or image patch) to the target class.

        The implementation follows these steps:
          - Use a one-hot vector to initialize relevance (Eq. 7)
          - For each attention layer:
            * Compute raw relevance = max(0, A(b) ⊙ ∇A(b)) (Eqs. 4 & 6)
            * Normalize raw relevance across tokens (Eq. 2)
            * Fuse heads using mean (E_h operator)
            * Add identity to incorporate skip connections (Eq. 15)
            * Normalize rows to enforce the conservation rule (Eq. 3)
          - Roll out the adjusted matrices across layers (Eq. 16)
          - Extract the [CLS] row and reshape as final relevance map.
        """
        # Clear any stored data from previous calls.
        self.attentions = []
        self.attention_gradients = []
        epsilon = self.epsilon

        # Forward pass through the model.
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Equation (7): Initialize relevance R^(0) = 1_t using a one-hot vector.
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1

        # Backward pass computes gradients as per the chain rule (Eq. 1).
        output.backward(gradient=one_hot)

        A_bars = []  # List to store adjusted matrices for each layer.
        for att, grad in zip(self.attentions, self.attention_gradients):
            # att and grad shape: (batch, num_heads, N, N)

            # Equation (4) & (6): Compute raw relevance.
            # raw = max(0, A(b) ⊙ ∇A(b))
            raw = torch.clamp(att * grad, min=0)
            
            # Equation (2): Normalize over the token dimension.
            # For each head j, compute:
            # R = raw / (sum_{j'} raw_{j'} + epsilon)
            R = raw / (raw.sum(dim=-1, keepdim=True) + epsilon)
            
            # Fuse across heads using the chosen fusion method (this approximates E_h).
            if self.head_fusion == "mean":
                R_fused = R.mean(dim=1)  # shape: (batch, N, N)
            elif self.head_fusion == "max":
                R_fused = R.max(dim=1)[0]
            elif self.head_fusion == "min":
                R_fused = R.min(dim=1)[0]
            else:
                raise ValueError("Unknown head fusion method: {}".format(self.head_fusion))
            
            # Equation (15): Add the identity matrix I to the fused relevance.
            # This corresponds to: Ā(b) = I + E_h(R)
            batch_size, N, _ = R_fused.shape
            I = torch.eye(N, device=R_fused.device).unsqueeze(0).expand(batch_size, -1, -1)
            A_bar = I + R_fused

            # Equation (3): Normalize each row to enforce conservation: sum_j Ā(b)_j = 1.
            A_bar = A_bar / (A_bar.sum(dim=-1, keepdim=True) + epsilon)
            
            A_bars.append(A_bar)
        
        # Equation (16): Rollout – Multiply adjusted matrices across layers.
        result = A_bars[0]
        for A_bar in A_bars[1:]:
            result = torch.matmul(result, A_bar)
        
        # Extract the final relevance map:
        # We take the row corresponding to the [CLS] token (assumed at index 0)
        # and ignore the first column (which is the [CLS] token itself).
        mask = result[:, 0, 1:]
        
        # Reshape the mask into a square grid. For ViT, N-1 tokens are image patches.
        grid_size = int(np.sqrt(mask.size(-1)))
        mask = mask.reshape(mask.size(0), grid_size, grid_size)
        
        # Normalize the final mask between 0 and 1.
        mask = mask / (mask.max() + epsilon)
        return mask.cpu().detach().numpy()[0]