import torch

class VITRawAttention:
    def __init__(self, model, attention_layer_name='attn_drop'):
        self.model = model
        self.attentions = []
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)
        return self.attentions