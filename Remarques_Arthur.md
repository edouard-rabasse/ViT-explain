Pour run avec l'Attention Rollout:
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 vit_explain.py --image_path examples/both.png --head_fusion max --discard_ratio 0.9 --use_cuda

Pour run avec le Gradient Attention Rollout:
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 vit_explain.py --image_path examples/both.png --head_fusion max --discard_ratio 0.9 --use_cuda --category_index 243


############################################

1. Introduction

Contexte et motivation :
Le tutoriel commence par expliquer que, traditionnellement, les réseaux de neurones convolutifs (CNN) ont dominé la vision par ordinateur. Cependant, avec le succès des Transformers en traitement du langage naturel (NLP), l’idée est d’explorer leur application aux images. L’approche ViT repose sur l’utilisation de la self-attention pour modéliser les dépendances globales dans une image, sans utiliser de convolutions.

Concept de base :
Au lieu d’extraire des caractéristiques locales avec des filtres convolutifs, le modèle découpe l’image en petits patchs, les traite comme des tokens (similaires aux mots en NLP), et les envoie dans un Transformer standard.


2. Architecture du Vision Transformer

Découpage de l'image en patchs :
L’image est divisée en patchs de taille fixe (par exemple, 16×16 pixels). Chaque patch est ensuite aplati et projeté linéairement pour obtenir un vecteur d’une dimension fixe.
Encodage positionnel :
Comme le Transformer ne possède pas de notion d’ordre intrinsèque, des encodages positionnels sont ajoutés aux vecteurs de patchs pour préserver l’information spatiale.
Ajout du token de classification :
Un token spécial (souvent appelé [class] token) est préfixé à la séquence de patchs. La représentation finale associée à ce token, après passage dans le Transformer, sert de représentation globale de l’image pour la classification.
Transformer Encoder :
La séquence (composée du token de classification et des patchs) est traitée par une série de blocs d’encodeurs. Chaque bloc comprend :
Une couche de self-attention multi-tête qui permet au modèle d’apprendre des dépendances globales entre les patchs.
Un MLP (perceptron multicouche) pour enrichir la représentation.
Des mécanismes de normalisation (LayerNorm) et des connexions résiduelles pour stabiliser l’apprentissage.
Tête de classification :
Finalement, le vecteur issu du [class] token est passé dans une couche linéaire (ou MLP) pour produire les prédictions de classe.


3. Implémentation et Code
Prétraitement des données :
Le tutoriel décrit comment préparer les images (redimensionnement, normalisation, etc.) avant de les passer dans le modèle.
Définition du modèle ViT :
Le code montre comment construire ou charger un modèle Vision Transformer. Dans certains cas, le modèle est pré-entraîné sur ImageNet et peut ensuite être fine-tuné sur d’autres datasets.
Entraînement et évaluation :
Des exemples de boucle d’entraînement et d’évaluation sont présentés. Le tutoriel aborde également des aspects tels que la définition de la fonction de perte, l’optimisation et l’utilisation d’un GPU pour accélérer l’entraînement.
Visualisation et analyse :
Le tutoriel explique comment extraire et visualiser les cartes d’attention générées par le modèle. Ces visualisations permettent de mieux comprendre quelles parties de l’image le modèle utilise pour prendre ses décisions.


4. Expériences et Résultats
Performances du ViT :
Le tutoriel présente des résultats expérimentaux qui illustrent la compétitivité des Vision Transformers sur des tâches de classification par rapport aux approches CNN traditionnelles.
Comparaison avec d’autres approches :
Il est également question de certaines limites et des défis spécifiques aux Transformers (comme leur besoin en données massives pour atteindre des performances optimales) et comment ces défis peuvent être surmontés.


5. Conclusion et Perspectives
Bénéfices des Vision Transformers :
Le tutoriel conclut en soulignant que les ViT offrent une nouvelle manière de traiter les images en capturant des dépendances globales et en fournissant des visualisations d’attention intéressantes, ce qui est utile pour l’interprétabilité.
Perspectives de recherche :
Des pistes pour améliorer et étendre ces modèles sont discutées, notamment en combinant Transformers et CNN ou en adaptant les architectures aux contraintes spécifiques de certains domaines (comme l’imagerie médicale).


En résumé:

Différence fondamentale avec les CNN :
Plutôt que d'utiliser des convolutions pour extraire localement des caractéristiques, le Vision Transformer transforme l'image en une séquence de petits patchs (tokens) et applique un Transformer (basé sur la self-attention) pour apprendre des dépendances globales entre ces patchs.
Un Transformer standard :
C'est une architecture composée de blocs d'attention multi-tête et de MLPs, conçue initialement pour traiter des séquences en NLP. Il permet à chaque token de s'informer de tous les autres tokens, offrant ainsi une compréhension globale du contexte.
Cette approche permet au modèle d'exploiter les avantages de la self-attention pour capturer des interactions à longue distance dans l'image, ce qui peut être particulièrement utile pour certaines tâches de vision par ordinateur où les relations globales sont importantes.