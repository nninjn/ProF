"""
PyTorch reimplementation of ADF (Automatic Discrimination Finder) generation loop.
"""
from __future__ import annotations
import sys
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# keep search path behavior
sys.path.append("backup/evaluation")
from . import generation_utilities

def compute_grad(x: np.ndarray, model: torch.nn.Module) -> np.ndarray:

    device = next(model.parameters()).device
    x = x.unsqueeze(0).requires_grad_(True)  # [1, d]
    prob = model(x)
    if prob.dim() > 1 and prob.size(-1) == 1:
        prob = prob.squeeze(-1)
    
    with torch.no_grad():
        y_pred = (prob > 0.5).float()
    
    loss = F.binary_cross_entropy(prob, y_pred)
    (grad,) = torch.autograd.grad(loss, x)
    return grad.squeeze(0).detach()


# -------------------------------------
#  ADF global/local generation phases
# -------------------------------------

def global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, max_iter, s_g, device):
    g_id = []
    all_gen_g = []
    try_times = 0
    g_num = len(seeds)
    for i in range(g_num):
        x1 = torch.tensor(seeds[i], dtype=torch.float32, device=device).clone()
        for _ in range(max_iter):
            try_times += 1
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint, device)
            if generation_utilities.is_discriminatory(x1, similar_x1, model):
                g_id.append(x1.clone())
                break
            x2 = generation_utilities.max_diff(x1, similar_x1, model)
            grad1 = compute_grad(x1, model)
            grad2 = compute_grad(x2, model)
            direction = torch.zeros_like(x1)
            sign_grad1 = torch.sign(grad1)
            sign_grad2 = torch.sign(grad2)
            for attrib in range(num_attribs):
                if attrib not in protected_attribs and sign_grad1[attrib] == sign_grad2[attrib]:
                    direction[attrib] = sign_grad1[attrib]
            x1 = x1 + s_g * direction
            x1 = generation_utilities.clip(x1, constraint)
            all_gen_g.append(x1.clone())


    g_id = torch.stack(g_id).unique(dim=0) if g_id else torch.empty((0, num_attribs), device=device)
    all_gen_g = torch.stack(all_gen_g) if all_gen_g else torch.empty((0, num_attribs), device=device)

    print('global', g_id.shape, all_gen_g.shape, try_times)
    return g_id, all_gen_g, try_times


def local_generation(num_attribs, l_num, g_id, protected_attribs, constraint, model, s_l, epsilon, device):
    direction = [-1, 1]
    l_id = []
    all_gen_l = []
    try_times = 0
    for x1 in g_id:
        x0 = x1.detach().clone()
        for _ in range(l_num):
            try_times += 1
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint, device)
            x2 = generation_utilities.find_pair(x1, similar_x1, model)
            grad1 = compute_grad(x1, model)
            grad2 = compute_grad(x2, model)
            p = generation_utilities.normalization(grad1, grad2, protected_attribs, epsilon)
            a = generation_utilities.random_pick(p)
            s = generation_utilities.random_pick([0.5, 0.5])
            x1[a] = x1[a] + direction[s] * s_l
            x1 = generation_utilities.clip(x1, constraint)
            all_gen_l.append(x1.clone())
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint, device)
            if generation_utilities.is_discriminatory(x1, similar_x1, model):
                l_id.append(x1.clone())
            else:
                x1 = x0.clone()
    # 去重
    l_id = torch.stack(l_id).unique(dim=0) if l_id else torch.empty((0, num_attribs), device=device)
    all_gen_l = torch.stack(all_gen_l) if all_gen_l else torch.empty((0, num_attribs), device=device)

        
    print('local', l_id.shape, all_gen_l.shape, try_times)
    return l_id, all_gen_l, try_times


def individual_discrimination_generation(X, seeds, protected_attribs, constraint, model,
                                         l_num, max_iter=10, s_g=1.0, s_l=1.0, epsilon=1e-6, device='cuda:0'):
    num_attribs = len(X[0])
    t1 = time.time()
    g_id, gen_g, g_gen_num = global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, max_iter, s_g, device)
    t2 = time.time()
    l_id, gen_l, l_gen_num = local_generation(num_attribs, l_num, g_id, protected_attribs, constraint, model, s_l, epsilon, device)

    all_id = torch.cat([g_id, l_id], dim=0)
    all_gen = torch.cat([gen_g, gen_l], dim=0)
    all_id_nondup = torch.unique(all_id, dim=0)
    all_gen_nondup = torch.unique(all_gen, dim=0)

    mask = torch.ones(X[0].shape[0], dtype=torch.bool, device=X[0].device)
    mask[protected_attribs] = False
    if all_id.numel() > 0:
        all_id_nondup = [x[mask] for x in all_id]
        all_id_nondup = torch.stack(all_id_nondup).unique(dim=0)
    else:
        all_id_nondup = torch.empty((0, num_attribs - len(protected_attribs)), device=device)

    if all_gen.numel() > 0:
        all_gen_nondup = [x[mask] for x in all_gen]
        all_gen_nondup = torch.stack(all_gen_nondup).unique(dim=0)
    else:
        all_gen_nondup = torch.empty((0, num_attribs - len(protected_attribs)), device=device)

    print(all_id_nondup.shape, all_gen_nondup.shape)
    return all_id_nondup, all_gen_nondup, g_gen_num + l_gen_num, g_id, g_gen_num, (t2 - t1)


def generate_seeds(X, c_num=4, num_seeds=100, fashion='Distribution'):
    num_attribs = len(X[0])
    clustered_data = generation_utilities.clustering(X, c_num)
    id_seeds = np.empty(shape=(0, num_attribs))
    for i in range(100000000):
        x_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i % c_num, fashion=fashion)
        id_seeds = np.append(id_seeds, [x_seed], axis=0)
        if len(id_seeds) >= num_seeds:
            break
    return id_seeds


def ADF(model, protected_attribs, constraint, X_train, device='cuda:0'):

    # ---------------- run ADF ----------------
    l_num = 1000
    for ROUND in range(1, 2):
        id_seeds = generate_seeds(X_train, num_seeds=1000)

        t1 = time.time()
        ids, _, total_iter, g_id, g_iter, global_time = individual_discrimination_generation(
            X_train, id_seeds, protected_attribs, constraint, model,
            l_num, max_iter=10, s_g=1.0, s_l=1.0, epsilon=1e-6, device=device
        )
        t2 = time.time()
        # np.save(f'discriminatory_data/{m_name}_{benchmark}_ids_ADF.npy', ids)
        num_ids = len(ids)
        global_num_ids = len(g_id)
        print("ADF", protected_attribs,  num_ids, t2 - t1, global_num_ids, global_time, ROUND)
        # with open(os.path.join(results_dir, filename), 'a+', encoding='utf-8') as save_file:
        #     print("ADF", benchmark, model_path, num_ids, t2 - t1, global_num_ids, global_time, ROUND, file=save_file)
    return num_ids