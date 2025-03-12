import argparse
import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import json 

from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout
from vit_explainability import VITTransformerExplainability
from vit_LRPmimic import VITTransformerLRPmimic
from vit_LRPexact import load_model_LRP, VITTransformerLRPexact


def get_default_attention_layer(model_name):
    # Define the default attention layer for each model.
    defaults = {
        "deit_tiny_patch16_224": "attn_drop",
        # Add more models and their defaults if needed:
        # "vit_base_patch16_224": "layer_name_for_vit_base",
    }
    return defaults.get(model_name, "attn_drop")  # Fallback to 'attn_drop' if model not found


def get_args():
    """
    Fonction qui permet de récupérer les arguments passés en ligne de commande
    TODO: Ajouter des arguments pour prendre par exemple un nouveau modèle. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    
    # Nouvel argument pour sélectionner la méthode
    parser.add_argument('--method', type=str, default='attention',
                        help='Méthode d\'explanation : "attention", "gradient", "transformer_attribution", "full_LRP", "partial_LRP"')
    
    parser.add_argument('--model_name', type=str, default='deit_tiny',
                        help='Nom du modèle à charger')
    
    # Vous pouvez aussi ajouter un argument pour les paramètres sous forme de chaîne JSON ou autres.
    parser.add_argument('--model_params', type=str, default='{"pretrained": true, "weights_path": "weights/deit_tiny_head_weights.pth"}',
                        help='Paramètres du modèle en JSON (ex: \'{"pretrained": true}\')')
    
    # Nouvel argument pour spécifier le nom de la couche d'attention
    parser.add_argument('--attention_layer_name', type=str, default='attn_drop',
                        help='Nom de la couche d\'attention à utiliser pour l\'explication')

    

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")
    return args


def show_mask_on_image(img, mask):
    """
    Fonction qui permet de superposer le masque sur l'image
    """
    img = np.float32(img) / 255
    print(mask.shape)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def load_model(model_name, parameters,):
    """
    TODO : si on veut pouvoir charger un modèle préentrainé autre que celui de Facebook.
    Par exemple, en fonction de model_name, on peut charger un modèle depuis torch.hub,
    ou depuis un autre repository.
    """
    print(model_name)
    if model_name == "deit_tiny":
        # Ici, parameters peut être un dictionnaire contenant des options comme pretrained=True
        # model = torch.hub.load('facebookresearch/deit:main', "deit_tiny_patch16_224", parameters["pretrained"])
        model = load_model_LRP(model_name, parameters)
    
    elif model_name == "deit_base":
        
        model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    
    elif model_name == "deit_tiny_finetuned":
        ## fine tuned model
        # from load_deit import load_deit
        # weights_path = parameters["weights_path"]
        # model = load_deit(weights_path)
        model = load_model_LRP(model_name, parameters)


    elif model_name == "autre_modele":
        # Vous pouvez ajouter d'autres conditions pour charger différents modèles.
        model = ...  # Code de chargement pour "autre_modele"
    else:
        raise ValueError("Modèle inconnu : {}".format(model_name))
    return model


def run_explanation(method, model, input_tensor, args):
    """
    Fonction qui permet de choisir la méthode d'explication à utiliser
    et de lancer l'explication
    """

    if method == 'attention':
        # print("Doing Attention Rollout")
        assert args.head_fusion in ["mean", "max", "min"], "Invalid head fusion type"
        assert 0 < args.discard_ratio < 1, "Invalid discard ratio"
        assert args.attention_layer_name is not None, "Attention layer name is required"
        explanation = VITAttentionRollout(model, head_fusion=args.head_fusion,
                                           discard_ratio=args.discard_ratio,
                                           attention_layer_name=args.attention_layer_name)
        mask = explanation(input_tensor)
        name = "out/attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)

    elif method == 'gradient':
        # print("Doing Gradient Attention Rollout")
        explanation = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
        mask = explanation(input_tensor, args.category_index)
        name = "out/grad_rollout_{}_{:.3f}_{}.png".format(args.category_index, args.discard_ratio, args.head_fusion)
    
    # elif method == 'explainability':
    #     print("Doing New Transformer Explainability Method")
    #     # Create an instance of the new explainer
    #     explanation = VITTransformerExplainability(model, attention_layer_name=args.attention_layer_name,
    #                                                head_fusion=args.head_fusion)
    #     # Pass the target class if needed (or leave as None to use the predicted class)
    #     mask = explanation(input_tensor, target_class=args.category_index)
    #     name = "out/ours_explanation.png"
    
    # elif method =="LRP_mimic":
    #     print("Doing LRP mimic Method")
    #     explanation = VITTransformerLRPmimic(model, attention_layer_name=args.attention_layer_name, head_fusion=args.head_fusion)
    #     # Pass the target class if needed (or leave as None to use the predicted class)
    #     mask = explanation(input_tensor, target_class=args.category_index)
    #     name = "out/LRP_mimic_explanation_{}_{}.png".format(args.category_index, args.head_fusion)


        # #print top 5 predictions
        # print("Top 5 predictions:", model(input_tensor).topk(5))
    
    elif method == 'transformer_attribution':
        assert args.use_cuda, "transformer attribution method requires cuda"
        # print("Doing transformer attribution Method")
        # ensure the model has a relprop method
        assert hasattr(model, 'relprop'), "Model does not have a relprop method"
        explanation = VITTransformerLRPexact(model)
        mask = explanation(input_tensor, category_index=args.category_index, method="transformer_attribution")
        name = "out/transformer_attribution_explanation.png"
    
    elif method == 'full_LRP':
        assert args.use_cuda, "full LRP method requires cuda"
        # print("Doing LRP exact Method")
        explanation = VITTransformerLRPexact(model)
        mask = explanation(input_tensor, category_index=args.category_index, method = "full")
        name = "out/LRP_exact_explanation.png"
    
    elif method == 'partial_LRP':
        assert args.use_cuda, "partial LRP method requires cuda"
        # print("Doing partial LRP")
        explanation = VITTransformerLRPexact(model)
        mask = explanation(input_tensor, category_index=args.category_index, method = "last_layer")
        name = "out/LRP_partial_explanation.png"


    # Vous pouvez ajouter ici d'autres méthodes :
    # Le mieux pour ça c'est de copier la structure des deux précédentes : nouveau fichier python avec une classe
    elif method == 'autre':
        print("Doing Other Explanation Method")
        mask = ...  # votre code ici
        name = "other_method.png"
    else:
        raise ValueError("Méthode inconnue : {}".format(method))
    return mask, name

def main(args, model=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(args.image_path)
    # img = img.resize((224, 224))
    input_tensor = transform(img).unsqueeze(0)
    if args.use_cuda:
        input_tensor = input_tensor.cuda()

    # Charger les paramètres du modèle depuis une chaîne JSON
    model_parameters = json.loads(args.model_params)
    
    # Charger le modèle via la fonction load_model
    if args.method in ["transformer_attribution", "full_LRP", "partial_LRP"]:
        print("loading model with LRP")
        model = load_model_LRP(args.model_name, model_parameters)
    
    else:
        model = load_model(args.model_name, model_parameters)


    if args.use_cuda:
        model = model.cuda()

    

    mask, name = run_explanation(args.method, model, input_tensor, args)


    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)
    # cv2.imshow("Input Image", np_img)
    # cv2.imshow(name, mask)
    cv2.imwrite("input.png", np_img)
    cv2.imwrite(name, mask)
    print("Explanation saved as", name)
    # cv2.waitKey(-1)

if __name__ == "__main__":
    print("Running main")
    args = get_args()
    main(args)