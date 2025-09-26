"""
This python file provides essential functions for individual discrimination generation.
(PyTorch version, minimal changes)
"""

import numpy as np
import torch
from sklearn import cluster
import itertools
import time


def clustering(data, c_num):
    # standard KMeans algorithm
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    kmeans = cluster.KMeans(n_clusters=c_num)
    y_pred = kmeans.fit_predict(data)
    return [data[y_pred == n] for n in range(c_num)]


# def clip(instance, constraint):
#     # clip the generated instance to satisfy the constraint
#     return np.minimum(constraint[:, 1].cpu(), np.maximum(constraint[:, 0].cpu(), instance))

def clip(instance, constraint):
    # Clip the generated instance to satisfy the constraint.
    instance = torch.as_tensor(instance, dtype=torch.float32)
    lower, upper = constraint[:, 0], constraint[:, 1]
    return torch.clamp(instance, min=lower, max=upper)

# def random_pick(probability):
#     # randomly pick an element from a probability distribution
#     random_number = np.random.rand()
#     current_proba = 0
#     for i in range(len(probability)):
#         current_proba += probability[i]
#         if current_proba > random_number:
#             return i

def random_pick(probability: torch.Tensor) -> int:
    # randomly pick an element from a probability distribution
    probability = torch.tensor(probability, dtype=torch.float32)
    idx = torch.multinomial(probability, 1)
    return idx.item()

def get_seed(clustered_data, X_len, c_num, cluster_i, fashion='RoundRobin'):
    # get a seed from the specified cluster
    if fashion == 'RoundRobin':
        index = np.random.randint(0, len(clustered_data[cluster_i]))
        return clustered_data[cluster_i][index]
    elif fashion == 'Distribution':
        pick_probability = [len(clustered_data[i]) / X_len for i in range(c_num)]
        x = clustered_data[random_pick(pick_probability)]
        index = np.random.randint(0, len(x))
        return x[index]


def similar_set(x, num_attribs, protected_attribs, constraint, device):
    # find all similar inputs corresponding to different combinations of protected attributes with non-protected attributes unchanged
    # x is a torch tensor now

    similar_x = torch.empty((0, num_attribs), device=device)
    protected_domain = []
    for i in protected_attribs:
        protected_domain.append(list(range(int(constraint[i][0]), int(constraint[i][1]) + 1)))

    all_combs = list(itertools.product(*protected_domain))
    for comb in all_combs:
        x_new = x.clone()
        for a, c in zip(protected_attribs, comb):
            x_new[a] = c
        similar_x = torch.vstack([similar_x, x_new.unsqueeze(0)])
    return similar_x


# def is_discriminatory(x, similar_x, model):
#     # identify whether the instance is discriminatory w.r.t. the model
#     device = next(model.parameters()).device
#     x = torch.tensor(x, dtype=torch.float32).to(device)
#     y_pred = (model(x) > 0.5).item()

#     for x_new in similar_x:
#         y_new = (model(x_new) > 0.5).item()
#         if y_new != y_pred:
#             return True
#     return False

def is_discriminatory(x, similar_x, model):
    device = next(model.parameters()).device
    
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_pred = (model(x_tensor) > 0.5).item()
    
    similar_x = torch.tensor(similar_x, dtype=torch.float32, device=device)
    y_similar = (model(similar_x) > 0.5).view(-1).tolist()
    
    return any(y != y_pred for y in y_similar)


# def max_diff(x, similar_x, model):
#     # select a similar instance such that the DNN outputs on them are maximally different
#     device = next(model.parameters()).device
#     x = torch.tensor(x, dtype=torch.float32).to(device)
#     y_pred_proba = model(x).detach().cpu().numpy()

#     def distance(x_new):
#         return np.sum(np.square(y_pred_proba - model(x_new).detach().cpu().numpy()))

#     max_dist = 0.0
#     x_potential_pair = x.detach().clone()
#     for x_new in similar_x:
#         dist = distance(x_new)
#         if dist > max_dist:
#             max_dist = dist
#             x_potential_pair = x_new.detach().clone()
#     return x_potential_pair


def max_diff(x, similar_x, model):
    """
    Select a similar instance such that the DNN outputs on them are maximally different.
    """
    device = next(model.parameters()).device
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.to(device)

    y_pred_proba = model(x)              # shape: [1, c] or [1]
    if similar_x.dim() == 1:
        similar_x = similar_x.unsqueeze(0)
    similar_x = similar_x.to(device)

    y_similar = model(similar_x)         # shape: [N, c] or [N]
    diffs = torch.sum((y_pred_proba - y_similar) ** 2, dim=1)
    max_idx = torch.argmax(diffs)
    return similar_x[max_idx].detach().clone()


def find_pair(x, similar_x, model):
    """
    Find a discriminatory pair given an individual discriminatory instance.
    """
    device = next(model.parameters()).device
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    y_pred = (model(x_tensor.unsqueeze(0)) > 0.5).item()  # batch维度

    pairs = []

    for x_pair in similar_x:
        x_pair = x_pair.to(device)
        y_pair = (model(x_pair.unsqueeze(0)) > 0.5).item()
        if y_pair != y_pred:
            pairs.append(x_pair)

    if len(pairs) == 0:
        pairs = [similar_x[0].to(device)]
    
    # 随机选一个
    selected_idx = random_pick([1.0 / len(pairs)] * len(pairs))
    return pairs[selected_idx]



# def normalization(grad1, grad2, protected_attribs, epsilon):
#     # gradient normalization during local search
#     gradient = np.zeros_like(grad1)
#     grad1 = np.abs(grad1)
#     grad2 = np.abs(grad2)
#     for i in range(len(gradient)):
#         saliency = grad1[i] + grad2[i]
#         gradient[i] = 1.0 / (saliency + epsilon)
#         if i in protected_attribs:
#             gradient[i] = 0.0
#     gradient_sum = np.sum(gradient)
#     probability = gradient / gradient_sum
#     return probability

def normalization(grad1, grad2, protected_attribs, epsilon):
    # gradient normalization during local search
    grad1 = grad1.abs()
    grad2 = grad2.abs()
    saliency = grad1 + grad2

    gradient = 1.0 / (saliency + epsilon)
    if protected_attribs: 
        gradient[protected_attribs] = 0.0
    gradient_sum = gradient.sum()
    probability = gradient / (gradient_sum)
    return probability

def purely_random(num_attribs, protected_attribs, constraint, model, gen_num):
    # generate instances randomly
    gen_id = np.empty(shape=(0, num_attribs))
    for i in range(gen_num):
        x_picked = [0] * num_attribs
        for a in range(num_attribs):
            x_picked[a] = np.random.randint(constraint[a][0], constraint[a][1] + 1)
        if is_discriminatory(x_picked, similar_set(x_picked, num_attribs, protected_attribs, constraint), model):
            gen_id = np.append(gen_id, [x_picked], axis=0)
    return gen_id
