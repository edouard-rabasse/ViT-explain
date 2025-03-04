Pour run avec l'Attention Rollout:
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 vit_explain.py --image_path examples/both.png --head_fusion max --discard_ratio 0.9 --use_cuda

Pour run avec le Gradient Attention Rollout:
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 vit_explain.py --image_path examples/both.png --head_fusion max --discard_ratio 0.9 --use_cuda --category_index 243
