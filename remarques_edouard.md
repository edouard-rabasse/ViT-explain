Quelques erreurs rencontrées : 
- si vous avez size incorrect sur  result = torch.eye(attentions[0].size(-1)), c'est parce que y'a une nouvelle version de timm qui fonctionne différemment. Le plus rapide c'est de faire un pip install timm==0.6.13 (source : https://github.com/jacobgil/vit-explain/issues/23)
