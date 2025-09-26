
from tensorflow.keras.models import load_model
from model import *
import torch
import torch.nn as nn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用GPU

# adult: 13
# compas: 6
dataset = 'german'
if dataset == 'adult':
    dim, num = 13, 13
    name = 'AC'
elif dataset == 'compas':
    dim, num = 6, 8
    name = 'compas'
elif dataset == 'bank':
    dim, num = 16, 9
    name = 'BM'
elif dataset == 'german':
    dim, num = 20, 6
    name = 'GC'
for i in range(1, num):

    target_model = f"{name}{i}"
    h5_model_path = f'/data/home/mjnn/majianan/fairness-repair/FairQuant-Artifact/models/{dataset}/{name}-{i}.h5'
    keras_model = load_model(h5_model_path)
    keras_model.summary()

    keras_weights = keras_model.get_weights()
    for layer in keras_model.layers:
        print(f"Layer: {layer.name}, Type: {layer.__class__.__name__}, Activation: {layer.activation.__name__ if hasattr(layer, 'activation') else 'None'}")

    pytorch_model = create_model(target_model, dim)

    keras_layer_idx = 0
    for l in keras_weights:
        print(l.shape)
    for layer in pytorch_model.layers:
        print(layer, isinstance(layer, nn.Linear))
        if isinstance(layer, nn.Linear):
            weight = keras_weights[keras_layer_idx]
            bias = keras_weights[keras_layer_idx + 1]
            layer.weight.data = torch.Tensor(weight.T)
            layer.bias.data = torch.Tensor(bias)
            keras_layer_idx += 2

    model_path = f"models_verify/{dataset}/{target_model}.pth"
    if not os.path.exists(f"models_verify/{dataset}"):
        os.makedirs(f"models_verify/{dataset}")
    torch.save(pytorch_model.state_dict(), model_path)
    print(pytorch_model)
    print(pytorch_model.layers[-1].bias)
