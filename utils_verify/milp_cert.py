# Source: https://github.com/eth-sri/lcifr/blob/master/code/experiments/certify.py

import argparse
import random
import time
from os import path, makedirs

import numpy as np
import torch.nn as nn
import torch
from gurobipy import GRB, LinExpr, Model

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

EPS = 1e-4


def add_relu_constraints(model, in_lb, in_ub, in_neuron, out_neuron, is_binary):
    if in_ub <= 0:
        out_neuron.lb = 0
        out_neuron.ub = 0

    elif in_lb >= 0:
        model.addConstr(in_neuron, GRB.EQUAL, out_neuron)

    else:
        model.addConstr(out_neuron >= 0)
        model.addConstr(out_neuron >= in_neuron)

        if is_binary:
            relu_ind = model.addVar(vtype=GRB.BINARY)
            model.addConstr(out_neuron <= in_ub * relu_ind)
            model.addConstr(out_neuron <= in_neuron - in_lb * (1 - relu_ind))
            model.addGenConstrIndicator(
                relu_ind, True, in_neuron, GRB.GREATER_EQUAL, 0.0
            )
            model.addGenConstrIndicator(
                relu_ind, False, in_neuron, GRB.LESS_EQUAL, 0.0
            )

        else:
            model.addConstr(
                -in_ub * in_neuron + (in_ub - in_lb) * out_neuron,
                GRB.LESS_EQUAL, -in_lb * in_ub
            )


def propagate(x, layers, grb_model, complete):
    n_outs = len(x[-1])

    for layer_idx, layer in enumerate(layers):
        x[layer_idx] = []

        if isinstance(layer, nn.Linear):
            for i in range(layer.out_features):
                expr = LinExpr()
                expr += layer.bias[i]
                expr += LinExpr(
                    layer.weight[i].detach().cpu().numpy().tolist(),
                    x[layer_idx - 1]
                )
                grb_model.update()

                grb_model.setObjective(expr, GRB.MINIMIZE)
                grb_model.optimize()
                lb = grb_model.objVal - EPS

                grb_model.setObjective(expr, GRB.MAXIMIZE)
                grb_model.optimize()
                ub = grb_model.objVal + EPS

                x[layer_idx] += [
                    grb_model.addVar(
                        lb=lb, ub=ub, vtype=GRB.CONTINUOUS,
                        name='x_{}_{}'.format(layer_idx, i)
                    )
                ]
                grb_model.addConstr(expr, GRB.EQUAL, x[layer_idx][i])
                n_outs = layer.out_features

        elif isinstance(layer, nn.ReLU):
            for i in range(n_outs):
                in_lb, in_ub = x[layer_idx - 1][i].lb, x[layer_idx - 1][i].ub
                x[layer_idx] += [
                    grb_model.addVar(in_lb, in_ub, vtype=GRB.CONTINUOUS)
                ]
                add_relu_constraints(
                    grb_model, in_lb, in_ub, x[layer_idx - 1][i],
                    x[layer_idx][i], complete
                )

        else:
            assert False

        grb_model.update()

    return x, n_outs


def certify(model, X_spaces, verbose=False):
            
    for param in model.parameters():
        param.data = param.double()

    encoder_layers = [model.layers[0]]
    for layer in model.layers[1:]:
        encoder_layers += [nn.ReLU()]
        encoder_layers += [layer]
    last_layer_ind = len(encoder_layers) - 1

    model.eval()
    model.if_sig = True
    model.only_feature = False

    all_times = []
    results = {}
    bounds = {}
    ver = 0

    for idx, x in enumerate(X_spaces):
        time_start = time.time()

        x = x.double()
        y = model(x[:, 0])
        y_pred = torch.where(y > 0.5, 1, 0).squeeze().cpu().numpy()

        grb_model = Model('milp')
        grb_model.setParam('OutputFlag', False)
        grb_model.setParam('MIPGap', 0)
        grb_model.setParam('NumericFocus', 2)
        grb_model.setParam('FeasibilityTol', 1e-9)
        grb_model.setParam('IntFeasTol', 1e-9)

        x_inp = []
        for i in range(x.shape[0]):
            x_inp.append(grb_model.addVar(x[i][0], x[i][1], name='x_-1_{}'.format(i)))
        x_data = {-1: x_inp}

        grb_model.update()

        y_outs, _ = propagate(x_data, encoder_layers, grb_model, complete=True)
        grb_model.update()

        grb_model.setObjective(y_outs[last_layer_ind][0], GRB.MINIMIZE)
        grb_model.optimize()
        min_logit = grb_model.objVal

        # if idx + 1 == 126:
        #     for var in grb_model.getVars():
        #         print(f"{var.varName}: {var.x}")

        grb_model.setObjective(y_outs[last_layer_ind][0], GRB.MAXIMIZE)
        grb_model.optimize()
        max_logit = grb_model.objVal

        # if idx + 1 == 126:
        #     for var in grb_model.getVars():
        #         print(f"{var.varName}: {var.x}")

        out_class = -1
        if min_logit >= 0:
            out_class = 1
        elif max_logit < 0:
            out_class = 0

        if out_class == y_pred:
            ver += 1
        cert = (out_class == y_pred)

        time_end = time.time()
        all_times += [time_end - time_start]

        if verbose:
            print(f"[n={(idx + 1):d}] {min_logit=}, {max_logit=}, certified%: {ver / (idx + 1):.4f}")
        results[idx + 1] = cert
        bounds[idx + 1] = [min_logit, max_logit]


    for param in model.parameters():
        param.data = param.data.float()
    # print(bounds)

    return ver / (idx + 1), results, bounds, all_times
            

