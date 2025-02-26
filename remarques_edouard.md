Quelques erreurs rencontrées : 
- si vous avez size incorrect sur  result = torch.eye(attentions[0].size(-1)), c'est parce que y'a une nouvelle version de timm qui fonctionne différemment. Le plus rapide c'est de faire un pip install timm==0.6.13 (source : https://github.com/jacobgil/vit-explain/issues/23)


J'ai commenté les lignes cv2.imshow() et cv2.waitKey() car elles ne fonctionnent pas en machine distante.


Le script principal (`vit_explain_modulable.py`) réalise les étapes suivantes :

1. **Récupération des arguments**  
   La fonction `get_args()` utilise `argparse` pour récupérer les arguments de la ligne de commande.  
   Les arguments disponibles sont :

   - `--use_cuda` : Utilise le GPU si disponible.
   - `--image_path` : Chemin vers l'image d'entrée (défaut : `./examples/both.png`).
   - `--head_fusion` : Méthode de fusion des têtes d'attention (`mean`, `max` ou `min`).
   - `--discard_ratio` : Proportion des chemins d'attention à ignorer.
   - `--category_index` : Indice de catégorie pour la méthode de Gradient Attention Rollout.
   - `--method` : Méthode d'explication à utiliser (`attention`, `gradient`, ou autres).
   - `--model_name` : Nom du modèle à charger (défaut : `deit_tiny_patch16_224`).
   - `--model_params` : Paramètres du modèle en JSON (défaut : `'{"pretrained": true}'`).

2. **Chargement du modèle**  
   La fonction `load_model(model_name, parameters)` permet de charger le modèle en fonction du nom et des paramètres fournis.  
   *Remarque :* Cette fonction est conçue pour être étendue afin de charger des modèles pré-entraînés autres que celui proposé par Facebook.

3. **Prétraitement de l'image**  
   Le script charge l'image, la redimensionne à 224×224 pixels, et applique une transformation (conversion en tenseur, normalisation, etc.).

4. **Exécution de l'explication**  
   La fonction `run_explanation(method, model, input_tensor, args)` sélectionne la méthode d'explication à utiliser en fonction de l'argument `--method` et exécute la méthode correspondante :
   - **Attention Rollout** : Utilise `VITAttentionRollout`.
   - **Gradient Attention Rollout** : Utilise `VITAttentionGradRollout`.
   - **Autres méthodes** : Possibilité d’ajouter d’autres cas.

5. **Superposition du masque et sauvegarde**  
   La fonction `show_mask_on_image()` permet de superposer le masque d'attention sur l'image d'origine, et le résultat est sauvegardé sur disque.

---

Pour run le script, dans votre terminal vous pouvez faire : 

``` python
python vit_explain.py --image_path examples/both.png --head_fusion max --discard_ratio 0.9 --use_cuda --method attention 
``` 

Sinon vous pouvez faire un truc comme dans le notebook test.ipynb

Ce qu'il faut faire : 
- fine-tune un modèle sur un dataset : adapter la fonction load_model pour charger le modèle préentrainé. 
- ajouter des méthodes d'explication : ajouter des cas dans la fonction run_explanation pour d'autres méthodes d'explication.
