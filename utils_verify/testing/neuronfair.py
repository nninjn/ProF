# nf_pytorch.py
import os
import copy
import argparse
import numpy as np
from typing import Tuple, List
from sklearn.cluster import KMeans
from scipy.optimize import basinhopping
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

@torch.no_grad()
def model_predict(model, x_np, device, return_label: bool = False) :
    model.eval()
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    prob = model(x).view(-1)  # after sigmoid
    if return_label:
        return (prob > 0.5).int().cpu().numpy()
    return prob.cpu().numpy()

def clip(instance, constraint):
    lower, upper = constraint[:, 0], constraint[:, 1]
    return torch.clamp(instance, min=lower, max=upper)


def seed_test_input(clusters: List[np.ndarray], limit: int) -> np.ndarray:
    """按集群轮转选取种子样本索引。clusters 是每个簇的索引数组列表。"""
    i = 0
    rows = []
    max_size = max(len(c) for c in clusters)
    while i < max_size and len(rows) < limit:
        for c in clusters:
            if i < len(c):
                rows.append(c[i])
                if len(rows) == limit:
                    break
        i += 1
    return torch.tensor(rows)

def check_for_error_condition(bounds, model, t, sens, combos, device: torch.device) -> bool:
    prob = model(t)
    label = int(prob > 0.5)

    t_batch = t.repeat(len(combos), 1)   # [N, D]
    t_batch[:, sens] = combos            # 批量替换敏感属性

    probs = model(t_batch)  
    labels_new = (probs > 0.5).int().view(-1)  


    return (labels_new != label).any().item()
    # t_batch = t.unsqueeze(0).repeat(len(values), 1)
    # t_batch[:, sens] = values.unsqueeze(1)

    # probs = model(t_batch)
    # labels_new = (probs > 0.5).int().view(-1)

    # return (labels_new != label).any().item()


class LocalPerturbation:
    def __init__(self, model, n_value, sens_param, input_dim, bounds, device):
        self.model = model
        self.n_value = n_value
        self.sens_param = sens_param
        self.input_dim = input_dim
        self.bounds = bounds
        self.device = device

    def __call__(self, x):
        perturbation_size = 1
        s = perturbation_size * (2 * torch.randint(0, 2, (1,), device=self.device).float() - 1.0).item()

        x = torch.tensor(x, device=self.device)
        nx = x.clone()
        nx[self.sens_param] = torch.tensor(
            self.n_value, dtype=nx.dtype, device=nx.device
        )

        def input_grad(w: torch.Tensor) -> torch.Tensor:
            t = w.unsqueeze(0).float().to(self.device).requires_grad_(True)
            out = self.model(t)
            p = out.view(-1)[0] if out.ndim==1 or out.shape[-1]==1 else F.softmax(out, dim=-1)[0, out.argmax(dim=-1)]
            (g,) = torch.autograd.grad(p, t)
            return g.detach().squeeze(0)  # 返回 [D] tensor

        ind_grad = input_grad(x)
        n_ind_grad = input_grad(nx)

        if torch.allclose(ind_grad, torch.zeros_like(ind_grad)) and torch.allclose(n_ind_grad, torch.zeros_like(n_ind_grad)):
            probs = torch.ones(self.input_dim, device=self.device) / (self.input_dim - 1)
            probs[self.sens_param] = 0.0
        else:
            grad_sum = 1.0 / (ind_grad.abs() + n_ind_grad.abs() + 1e-12)
            grad_sum[self.sens_param] = 0.0
            ssum = grad_sum.sum()
            probs = grad_sum / (ssum if ssum > 0 else 1.0)

        index = torch.multinomial(probs, num_samples=1).item()
        x_new = x.clone()
        x_new[index] += s
        x_new = clip(x_new, self.bounds)
        return x_new.detach().cpu().numpy()

# ----------------------------
# 主流程（全局 + 局部）
# ----------------------------
def neuronfair(sensitive_param, model, X, bounds, input_shape, device: str = "cuda:1"):
    cluster_num = 4
    max_global = 1000
    max_local = 1000
    max_iter = 40
    perturbation_size = 1
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(X.cpu().numpy())
    clusters = [np.where(kmeans.labels_ == i)[0] for i in range(cluster_num)]
    inputs_idx = seed_test_input(clusters, min(max_global, len(X)))

    tot_inputs = set()
    global_disc_inputs = set()
    global_disc_inputs_list = []
    local_disc_inputs = set()
    local_disc_inputs_list = []
    value_list = []
    suc_idx = []

    # 取各敏感属性的上下界
    sens_bounds = [range(int(bounds[s, 0].item()), int(bounds[s, 1].item())+1) for s in sensitive_param]
    combos = torch.tensor(list(product(*sens_bounds)), device=device, dtype=X[0].dtype)  # [N, len(sens)]

    sd = 0

    def evaluate_local(inp: np.ndarray) -> bool:
        inp = torch.tensor(inp, dtype=torch.float32, device=device)
        result = check_for_error_condition(bounds, model, inp, sensitive_param, combos=combos, device=device)
        temp = inp.to(torch.int).tolist()
        temp_wo = [v for idx, v in enumerate(temp) if idx not in sensitive_param]
        tot_inputs.add(tuple(temp_wo))
        if result and (tuple(temp_wo) not in global_disc_inputs) and (tuple(temp_wo) not in local_disc_inputs):
            local_disc_inputs.add(tuple(temp_wo))
            local_disc_inputs_list.append(temp_wo)
        # return (not result)
        return 0.0 if result else 1.0 
    



    for num, index in enumerate(inputs_idx):
        sample = X[index:index+1].clone()
        memory1 = torch.zeros_like(sample)
        memory2 = torch.ones_like(sample)

        memory3 = -torch.ones_like(sample)

        for it in range(max_iter + 1):
            probs = model(sample)  # [C]
            label = int(probs > 0.5)
            prob = float(probs)

            s_idx = sensitive_param
            # lo, hi = int(bounds[s_idx, 0].item()), int(bounds[s_idx, 1].item())
            max_diff = -1.0

            n_label = label
            orig_values = [int(sample[0, s].item()) for s in sensitive_param]
            n_value = orig_values[:]  # 当前最优值
            # for val in range(lo, hi + 1):
            #     if val == sample[0, s_idx]:
            #         continue
            #     n_sample = sample.clone()
            #     n_sample[0, s_idx] = val
            #     n_probs = model(n_sample)
            #     nlab = int(n_probs > 0.5)
            #     if nlab != label:
            #         n_value = val
            #         n_label = nlab
            #         break  # 直接发现翻转
            #     diff = abs(prob - n_probs)
            #     if diff > max_diff:
            #         max_diff = diff
            #         n_value = val
            #         n_label = nlab

            # 枚举所有组合
            for vals in product(*sens_bounds):
                if list(vals) == orig_values:
                    continue

                n_sample = sample.clone()
                for s, v in zip(s_idx, vals):
                    n_sample[0, s] = v

                n_probs = model(n_sample)
                nlab = int(n_probs > 0.5)

                if nlab != label:
                    n_value = vals
                    n_label = nlab
                    break  # 直接发现翻转
                diff = abs(prob - n_probs)
                if diff > max_diff:
                    max_diff = diff
                    n_value = vals
                    n_label = nlab

            # 生成去掉敏感特征后的 key
            temp = copy.deepcopy(sample[0].cpu().int().tolist())  # list
            temp = [v for idx, v in enumerate(temp) if idx not in sensitive_param]

            # 判重并记录
            if label != n_label and (tuple(temp) not in global_disc_inputs) and (tuple(temp) not in local_disc_inputs):
                global_disc_inputs_list.append(temp)         # list 用于存储
                global_disc_inputs.add(tuple(temp))          # tuple 用于 set 判重
                orig_vals = [int(sample[0, s].item()) for s in sensitive_param]
                value_list.append([orig_vals, n_value])  # 记录敏感属性的原始值和当前值     
                suc_idx.append(index)

                # 局部扰动
                local_step = LocalPerturbation(model=model, n_value=n_value, sens_param=sensitive_param,
                    input_dim=input_shape, bounds=bounds, device=device)
                minimizer = {"method": "L-BFGS-B"}
                basinhopping(evaluate_local, sample[0].detach().cpu().numpy(), stepsize=1.0, take_step=local_step,
                                minimizer_kwargs=minimizer,
                                niter=max_local)
                print(len(tot_inputs), num, len(local_disc_inputs),
                      f"Percentage discriminatory inputs of local search- { (len(local_disc_inputs) / (len(tot_inputs)+1)) * 100:.2f}")
                break  # 跳出全局迭代，对下一个种子样本继续

            sample.requires_grad_(True)
            out = model(sample)
            if out.ndim == 1 or out.shape[-1] == 1:
                s_grad = torch.sign(torch.autograd.grad(out, sample, retain_graph=False)[0])
            else:
                probs_t = F.softmax(out, dim=-1)
                s_grad = torch.sign(torch.autograd.grad(probs_t[0, label], sample, retain_graph=False)[0])

            n_sample = sample.clone()
            n_sample[0, s_idx] = torch.tensor(
            n_value, dtype=n_sample.dtype, device=n_sample.device
        )
            n_sample.requires_grad_(True)
            out_n = model(n_sample)
            if out_n.ndim == 1 or out_n.shape[-1] == 1:
                n_grad = torch.sign(torch.autograd.grad(out_n, n_sample, retain_graph=False)[0])
            else:
                probs_n = F.softmax(out_n, dim=-1)
                n_grad = torch.sign(torch.autograd.grad(probs_n[0, n_label], n_sample, retain_graph=False)[0])

            sn_grad = torch.sign(s_grad + n_grad)

            # 计算 g_diff
            g_diff = torch.where(s_grad == n_grad, torch.ones_like(s_grad), torch.zeros_like(s_grad))
            g_diff[0, s_idx] = 0
            if torch.all(g_diff == 0):
                g_diff = sn_grad
                g_diff[0, s_idx] = 0

            # 全零梯度随机扰动
            if torch.all(s_grad == 0) or torch.all(memory1 == memory3):
                torch.manual_seed(2020 + sd)
                sd += 1
                delta = perturbation_size
                s_grad[0] = torch.randint(-int(delta), int(delta)+1, s_grad.shape[1:], device=device)

            g_diff = torch.ones_like(sample)
            g_diff[0, s_idx] = 0
            cal_grad = s_grad * g_diff

            memory1, memory2, memory3 = memory2, memory3, cal_grad

            # 更新样本
            sample = clip(sample.clone() + perturbation_size * cal_grad, bounds)

            if it == max_iter:
                break

    print("Total Inputs:", len(tot_inputs))
    print("Global discriminatory inputs:", len(global_disc_inputs))
    print("Local discriminatory inputs:", len(local_disc_inputs))
    print(len(global_disc_inputs_list), len(local_disc_inputs_list))
    return len(global_disc_inputs_list) + len(local_disc_inputs_list)

