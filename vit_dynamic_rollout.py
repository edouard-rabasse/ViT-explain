import torch
import numpy as np

def dynamic_rollout(attentions, head_fusion, quantile=0.85):
    result = torch.eye(attentions[0].size(-1))
    
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise ValueError("Attention head fusion type not supported")
            
            # Calcul du seuil dynamique en fonction du quantile
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            threshold = torch.quantile(flat, quantile, dim=-1, keepdim=True)
            
            # Masquage des valeurs inférieures au seuil
            mask = flat >= threshold
            flat = flat * mask.float()
            
            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + I) / 2
            a = a / a.sum(dim=-1, keepdim=True)
            
            result = torch.matmul(a, result)
    
    # Extraction du masque final
    mask = result[0, 0, 1:]
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask

class VITAttentionRolloutDynamic:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean", quantile=0.85):
        self.model = model
        self.head_fusion = head_fusion
        self.quantile = quantile
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
        self.attentions = []
    
    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())
    
    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            _ = self.model(input_tensor)
        return dynamic_rollout(self.attentions, self.head_fusion, self.quantile)