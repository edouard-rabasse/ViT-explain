# vit_explain.py
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout
from attention_flow import get_adjmat, get_attention_flow  # Importation des fonctions d'Attention Flow

def get_args():
    # (Le code pour récupérer les arguments reste inchangé)
    pass

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def main(args):
    model = torch.hub.load('facebookresearch/deit:main', 
        'deit_tiny_patch16_224', pretrained=True)
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(args.image_path)
    img = img.resize((224, 224))
    input_tensor = transform(img).unsqueeze(0)
    if args.use_cuda:
        input_tensor = input_tensor.cuda()

    # Génération des tokens d'entrée
    input_tokens = ['class'] + [f"patch_{i}" for i in range(196)]

    if args.category_index is None:
        print("Doing Attention Flow")
        attention_flow = VITAttentionFlow(model, discard_ratio=args.discard_ratio)
        mask = attention_flow(input_tensor, input_tokens)
        name = "attention_flow_{:.3f}.png".format(args.discard_ratio)
    else:
        print("Doing Gradient Attention Rollout")
        grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
        mask = grad_rollout(input_tensor, args.category_index)
        name = "grad_rollout_{}_{:.3f}.png".format(args.category_index,
            args.discard_ratio)

    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)
    cv2.imwrite("input.png", np_img)
    cv2.imwrite(name, mask)

if __name__ == "__main__":
    args = get_args()
    main(args)
