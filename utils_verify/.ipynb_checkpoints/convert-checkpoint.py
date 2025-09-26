
from tensorflow.keras.models import load_model
from model import AC1, AC2
import torch
import torch.nn as nn

# AC1/AC2/AC3/....
target_model = input("目标模型: ")
# adult: 13
dim = int(input("输入维度: "))

h5_model_path = f'FairQuant-Artifact/models/adult/{target_model[:-1]}-{target_model[-1]}.h5'
keras_model = load_model(h5_model_path)

keras_weights = keras_model.get_weights()
for w in keras_weights:
    print(w.shape)

for i in range(3, 13):
    keras_model = load_model(f'FairQuant-Artifact/models/adult/AC-{i}.h5')
    keras_weights = keras_model.get_weights()
    for w in keras_weights:
        print(w.shape)
    print(f'--------{i}---------')
exit()

pytorch_model = {'AC1': AC1, 
                 'AC2': AC2}[target_model](dim)

keras_layer_idx = 0
for layer in pytorch_model.children():
    if isinstance(layer, nn.Linear):
        weight = keras_weights[keras_layer_idx]
        bias = keras_weights[keras_layer_idx + 1]
        layer.weight.data = torch.Tensor(weight.T)
        layer.bias.data = torch.Tensor(bias)
        keras_layer_idx += 2
model_path = f"models-verify/{target_model}.pth"
torch.save(pytorch_model.state_dict(), model_path)
print(pytorch_model)