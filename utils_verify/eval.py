import random
import numpy as np
import torch

from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference
from sklearn.metrics import average_precision_score

from .milp_cert import certify

from itertools import product
from .testing.ADF_torch import ADF
from .testing.EIDIG import EIDIG

def evaluate_dp_new(model, X_test, y_test, A_test):
    model.eval()

    # calculate average precision
    X_test_cuda = torch.tensor(X_test).cuda().float()
    output, _, _, _ = model(X_test_cuda)
    pred = np.int64(output.cpu().detach().numpy() > 0.5)
    dp = demographic_parity_difference(y_test, pred, sensitive_features=A_test)
    y_scores = output[:, 0].data.cpu().numpy()
    ap = average_precision_score(y_test, y_scores)

    return ap, dp


def evaluate_eo_new(model, X_test, y_test, A_test, testing=False, optimizer=None):
    model.eval()

    # calculate average precision
    X_test_cuda = torch.tensor(X_test).cuda().float()
    output, _, _, _ = model(X_test_cuda)
    pred = np.int64(output.cpu().detach().numpy() > 0.5)
    eo = equalized_odds_difference(y_test, pred, sensitive_features=A_test)
    y_scores = output[:, 0].data.cpu().numpy()
    ap = average_precision_score(y_test, y_scores)

    return ap, eo


def my_dp(model, X_test, y_test, A_test):
    model.eval()
    pred = np.int64(model(X_test).cpu().detach().numpy() > 0.5)
    new_dp = demographic_parity_difference(y_test, pred, sensitive_features=A_test)
    return new_dp


def my_eo(model, X_test, y_test, A_test):
    model.eval()
    output = model(X_test)
    pred = np.int64(output.cpu().detach().numpy() > 0.5)
    eo = equalized_odds_difference(y_test, pred, sensitive_features=A_test)
    return eo


def test_acc(model, X_test, y_test):
    y = model(X_test, all=False)
    pred = torch.where(y > 0.5, 1, 0).squeeze().cpu().numpy()
    assert pred.shape == y_test.shape, "error in shape"
    correct = (pred == y_test).sum()
    acc = correct / len(y_test)
    return acc


from .data_config import *
def my_idi_test(sample_round, num_gen, dataset, model, sen, eps_prop,
                device, mode="sample", testset=None, store=False, store_num=0):
    model.if_sig = True
    model.only_feature = False
    model.eval()
    assert (mode == "sample" or testset is not None), "need testset!"

    bounds = dataset.input_bounds
    sensitive_indices = [dataset.sensitive_feature[s] for s in sen]
    if eps_prop != "":
        eps_attr = dataset.eps_fair_property[eps_prop][0]
        epsilon = dataset.eps_fair_property[eps_prop][1]

    value_ranges = [list(range(bounds[idx][0], bounds[idx][1] + 1)) for idx in sensitive_indices]
    value_combinations = torch.tensor(list(product(*value_ranges)), dtype=torch.float, device=device)

    not_fair = 0
    unfair_pairs = []

    if mode != 'sample':
        sample_round = 1
        num_gen = len(testset)

    for i in range(sample_round):
        if mode == 'sample':
            X = torch.zeros((num_gen, len(bounds)), dtype=torch.float, device=device)
            for i, (lower, upper) in enumerate(bounds):
                X[:, i] = torch.randint(low=lower, high=upper+1, size=(num_gen,), device=device).float()
        else:
            X = torch.tensor(testset, dtype=torch.float, device=device)

        # Step 2: build similar samples by varying sensitive features
        num_comb = len(value_combinations)
        X_similar = X.unsqueeze(1).repeat(1, num_comb, 1)  # shape: [N, C, d]
        # Expand combinations to match (num_gen, num_comb, len(sensitive_indices))
        comb_tensor = value_combinations.unsqueeze(0).repeat(num_gen, 1, 1)  # (num_gen, num_comb, k)

        # Assign values in a vectorized way
        for i, attr_idx in enumerate(sensitive_indices):
            X_similar[:, :, attr_idx] = comb_tensor[:, :, i]

        if eps_prop != "":
            perturb_range = torch.arange(-epsilon, epsilon + 1, device=device)  # shape: [2*eps+1]
            num_perturb = len(perturb_range)

            N, C, d = X_similar.shape  # N=batch, C=Combine, d=input dim

            # Expand X_similar to shape [N, C, 2*eps+1, d]
            X_expanded = X_similar.unsqueeze(2).repeat(1, 1, num_perturb, 1)  # shape: [N, C, R, d]

            # Create delta tensor for the eps_attr
            delta = torch.zeros((1, 1, num_perturb, d), device=device)
            delta[..., eps_attr] = perturb_range
            perturbed_all = X_expanded + delta
            perturbed_all = torch.clamp(perturbed_all, min=torch.tensor([b[0] for b in bounds], device=device),
                                                    max=torch.tensor([b[1] for b in bounds], device=device))
            X_similar = perturbed_all.view(N, C * num_perturb, d)

        X_flat = X_similar.view(-1, X_similar.shape[-1])  # [N*C', d]
        with torch.no_grad():
            y = model(X_flat.to(device), all=False).view(X_similar.shape[:2])  # [N, C']
        pred = (y > 0.5).int()  # shape: [N, C']

        first_preds = pred[:, 0].unsqueeze(1).expand_as(pred)
        row_equal = (pred == first_preds).all(dim=1)  # shape: [N], bool

        not_fair += (~row_equal).sum().item()
        if store and store_num > 0:
            unfair_indices = (~row_equal).nonzero(as_tuple=True)[0]
            for idx in unfair_indices[:store_num - len(unfair_pairs)]:
                pred_row = pred[idx]
                if not torch.all(pred_row == pred_row[0]):
                    diff_idx = (pred_row != pred_row[0]).nonzero()[0].item()
                    unfair_pair = torch.stack([X_similar[idx, 0], X_similar[idx, diff_idx]])
                    unfair_pairs.append(unfair_pair)

 
    unfairness_rate = not_fair / (sample_round * num_gen)
    # print(f"{mode} {not_fair} {sample_round} {num_gen}")
    if store:
        return unfairness_rate, torch.stack(unfair_pairs[:store_num])
    else:
        return unfairness_rate



def discrete_sens_cfr(model, dataset, sen, X_spaces, device):
    model.if_sig = True
    model.only_feature = False
    model.eval()
    bounds = dataset.input_bounds
    sensitive_indices = [dataset.sensitive_feature[s] for s in sen]

    value_ranges = [list(range(bounds[idx][0], bounds[idx][1] + 1)) for idx in sensitive_indices]
    value_combinations = torch.tensor(list(product(*value_ranges)), dtype=torch.float, device=device)
    num_comb = value_combinations.shape[0]
    fair = 0
    for i in range(len(X_spaces)):
        X = X_spaces[i, ..., 0]
        X_similar = X.unsqueeze(0).repeat(num_comb, 1)

        for j, attr_idx in enumerate(sensitive_indices):
            X_similar[:, attr_idx] = value_combinations[:, j]

        y = model(X_similar.to(device), all=False)
        pred = torch.where(y > 0.5, 1, 0)
        fair += torch.all(pred == pred[0])

    cfr = fair / len(X_spaces)
    print(f'discrete cfr: {fair} / {len(X_spaces)} = {cfr}')
    return cfr

def evaluation(model, dataset, X_test, y_test, A_test, full_set, 
                sen, X_spaces, eps_prop, device, 
                constraints, verbose=True, logger=None):
    model.if_sig = True
    model.only_feature = False
    model.eval()
    dataset = {'adult': adult, 'census': adult, 'compas': compas, 'bank': bank, 'german': german}[dataset]
    sensitive_indices = [dataset.sensitive_feature[s] for s in sen]

    idi_sample_per = my_idi_test(10, 10000, dataset, model, sen, eps_prop, device)
    idi_test_per = my_idi_test(10, 10000, dataset, model, sen, eps_prop, device, mode='testset', testset=full_set)

    X_test = torch.from_numpy(X_test).to(device).float()
    y_hat = model(X_test)
    y_scores = y_hat[:, 0].data.cpu().numpy()
    ap = average_precision_score(y_test, y_scores)
    acc = test_acc(model, X_test, y_test)
    eo = my_eo(model, X_test, y_test, A_test)
    dp = my_dp(model, X_test, y_test, A_test)

    full_set = torch.tensor(full_set, dtype=torch.float32).to(device)
    # adf_idis = ADF(model, sensitive_indices, constraints, full_set)
    # eidig_idis = EIDIG(model, sensitive_indices, constraints, full_set)

    if sen == 'age' or 'age' in sen or eps_prop != "":
        cert_fair_rate, results, outputs, _ = certify(model, X_spaces=X_spaces)
    else:
        cert_fair_rate = discrete_sens_cfr(model, dataset, sen, X_spaces=X_spaces, device=device).item()
    if verbose and logger is None:
        print(f'average precision : {ap}')
        print(f'average  accuracy : {acc}')
        print('EO:', eo)
        print('DP:', dp)
        print('Percentage of IDI from full data set:', idi_test_per)
        print('Percentage of IDI from full space sampling:', idi_sample_per)
        print('Certified fair rate:', cert_fair_rate)
        # print(f'Num of idis generated by adf testing: {adf_idis}')
        # print(f'Num of idis generated by eidig testing: {eidig_idis}')
    if logger is not None:
        logger.info(f'average precision : {ap}')
        logger.info(f'average  accuracy : {acc}')
        # logger.info('EO:', eo)
        # logger.info('DP:', dp)
        logger.info(f'Certified fair rate: {cert_fair_rate}')
        logger.info(f'Percentage of IDI from full data set: {idi_test_per}')
        logger.info(f'Percentage of IDI from full space sampling: {idi_sample_per}')
        # logger.info(f'Num of idis generated by testing: {adf_idis}')
    return ap, acc, eo, dp, idi_sample_per, cert_fair_rate