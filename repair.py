import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from gurobipy import *
from utils_verify.model import slice_model_at_penultimate_layer


class FairRepair():
    def __init__(self, model, input_space, X_repair, ns_dim, device) -> None:

        self.device = device  
        self.X_repair = X_repair
        self.dim = X_repair.shape[-1]
        self.ns_dim = ns_dim
        self.full_space = input_space
        self.s_dim = [i for i in range(self.dim) if i not in self.ns_dim]
        self.model = model
        
        with torch.no_grad():
            # X_spaces: (N, dim, 2) the set of spaces that contain all similarily individual for each x in X
            self.X_spaces = torch.stack([X_repair, X_repair], dim=-1)
            self.X_spaces[:, self.s_dim, :] = self.full_space[self.s_dim, :]
        
        self.phase1_time = 0
        self.phase2_time = 0

        self.L_fair_set = {}
        self.L_bce_set = {}

    
    def symbolic_bounds(self, model, space):
        # Note: The bounds computed by auto_LiRPA seem contain numerical errors (~1e-6 to 1e-5).
        input_lb = space[..., 0].detach().clone()
        input_ub = space[..., 1].detach().clone()
        symbolic_net = BoundedModule(model, torch.empty_like(input_lb), device=self.device)
        ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
        true_input = BoundedTensor(torch.empty_like(input_lb), ptb)

        required_A = defaultdict(set)
        required_A[symbolic_net.output_name[0]].add(symbolic_net.input_name[0])
        lb, ub, A_dict = symbolic_net.compute_bounds(x=(true_input,), method="backward", return_A=True, needed_A_dict=required_A)

        lower_A = A_dict[symbolic_net.output_name[0]][symbolic_net.input_name[0]]['lA']
        upper_A = A_dict[symbolic_net.output_name[0]][symbolic_net.input_name[0]]['uA']
        lower_bias = A_dict[symbolic_net.output_name[0]][symbolic_net.input_name[0]]['lbias']
        upper_bias = A_dict[symbolic_net.output_name[0]][symbolic_net.input_name[0]]['ubias']

        return lb, ub, lower_A, upper_A, lower_bias, upper_bias



    def get_Hc(self, model, space):
        # isolate the last layer and sigmoid
        model.only_feature = True
        slice_model = model
        input_lb = space[..., 0].detach().clone()
        input_ub = space[..., 1].detach().clone()
        symbolic_slice_net = BoundedModule(slice_model, torch.empty_like(input_lb), device=self.device)
        ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
        true_input = BoundedTensor(torch.empty_like(input_lb), ptb)
        lb, ub = symbolic_slice_net.compute_bounds(x=(true_input,), method="backward")
        Hc = torch.stack([lb, ub], dim=-1)
        model.only_feature = False
        return Hc


    def minimize_Hc(self, X_normal, y_normal, epochs):
        # Similar set of X_idi: self.X_spaces
        start = time.time()
        with torch.no_grad():
            original_H = self.get_Hc(self.model, space=self.X_spaces)
            ori_H_diff = original_H[..., 1] - original_H[..., 0]
            # Exclude the spaces where the H_difference is zero
            valid_mask = (torch.sum(ori_H_diff, dim=-1) > 0)
        # print(valid_mask)
        self.model.train()
        self.model.if_sig = True
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        y = torch.tensor(y_normal, dtype=torch.float32, device=X_normal.device).unsqueeze(-1)

        
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Symbolic Bound Synthesis 
            H = self.get_Hc(self.model, space=self.X_spaces.detach().clone())
            H_diff = H[..., 1] - H[..., 0]        
            H_loss = (torch.sum(H_diff[valid_mask], dim=-1) / torch.sum(ori_H_diff[valid_mask], dim=-1)).mean() 

            outputs = self.model(X_normal)
            train_loss = criterion(outputs, y)

            loss = train_loss + H_loss
            loss.backward()
            optimizer.step()
            self.L_fair_set[epoch] = H_loss
            self.L_bce_set[epoch] = train_loss
            # print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | H loss {H_loss:.2f}% | ")
        self.phase1_time = time.time() - start


    def symbolic_constraint_solve(self, mode='NoLabel'):
        start = time.time()

        W = self.model.layers[-1].weight.double().detach().cpu().numpy()
        b = self.model.layers[-1].bias.double().detach().cpu().numpy()
        k = len(self.X_spaces)  # number of property
        m = self.dim  # dim for x
        d = W.shape[1]  # dim for h
        M = 1e4  # BigM
        self.model.only_feature = True
        self.model.if_sig = False

        L, U, lalpha, ualpha, lbeta, ubeta = self.symbolic_bounds(self.model, self.X_spaces)
        lalpha, ualpha, lbeta, ubeta = lalpha.cpu().numpy(), ualpha.cpu().numpy(), lbeta.cpu().numpy(), ubeta.cpu().numpy()
        print(lalpha.shape)
        lX = self.X_spaces[..., 0].detach().cpu().numpy()
        uX = self.X_spaces[..., 1].detach().cpu().numpy()

        MILP_model = Model("MILP_Model")
        # MILP_model.Params.NumericFocus = 3
        MILP_model.Params.IntFeasTol = 1e-9
        MILP_model.Params.FeasibilityTol = 1e-9
        # MILP_model.setParam('OutputFlag', 0)

        Delta_W = MILP_model.addVars(d, lb=-3, ub=3, name="Delta_W")
        Delta_b = MILP_model.addVar(lb=-3, ub=3, name="Delta_b")
        abs_delta_w = MILP_model.addVars(d, lb=0, ub=3, name="abs_delta_w")
        abs_delta_b = MILP_model.addVar(lb=0, ub=3, name="abs_delta_b")
        Z = MILP_model.addVars(k, vtype=GRB.BINARY, name="Z")

        A_rows = 2 * d + 2 * m
        lamda = MILP_model.addVars(k, A_rows, lb=0, ub=GRB.INFINITY, name="lamda")
        eta = MILP_model.addVars(k, A_rows, lb=0, ub=GRB.INFINITY, name="eta")
        MILP_model.update()

        # ABS constraint
        for j in range(d):
            MILP_model.addGenConstrAbs(abs_delta_w[j], Delta_W[j], name=f"abs_constr_w_{j}")
        MILP_model.addGenConstrAbs(abs_delta_b, Delta_b, name="abs_constr_b")

        A = np.zeros((k, 2 * (d + m), d + m))
        D = np.zeros((k, 2 * (d + m)))  
        for i in range(k):
            # Construct matrix A, D and C
            A[i] = np.vstack([np.hstack([-np.eye(d), lalpha[i]]),
                             np.hstack([np.eye(d), -ualpha[i]]),
                             np.hstack([np.zeros((m, d)), -np.eye(m)]),
                             np.hstack([np.zeros((m, d)), np.eye(m)])])
            # print(-lbeta[i], ubeta[i], -lX[i], uX[i])
            D[i] = np.concatenate([-lbeta[i], ubeta[i], -lX[i], uX[i]])
            C_expr = [W[0][j] + Delta_W[j] for j in range(d)] + [0.0 for j in range(m)]

            # dual cons: A_i^T * lambda_i = -C, A_i^T * eta_i = -C
            for j in range(d + m):
                MILP_model.addConstr(quicksum(A[i][l, j] * lamda[i, l] for l in range(A_rows)) == -C_expr[j], name=f"dual_condition_lambda_{i}_{j}")
            for j in range(d + m):
                MILP_model.addConstr(quicksum(A[i][l, j] * eta[i, l] for l in range(A_rows)) == C_expr[j], name=f"dual_condition_eta_{i}_{j}")

            # constraint for lower bound and upper bound
            lb_i = b[0] + Delta_b - quicksum(lamda[i, l] * D[i][l] for l in range(A_rows))
            ub_i = b[0] + Delta_b + quicksum(eta[i, l] * D[i][l] for l in range(A_rows))
            # MILP_model.addConstr(lb_i >= - M * (1 - Z[i]), name=f"LB_constraint_{i}")
            # MILP_model.addConstr(ub_i <= M * Z[i] - 1e-2, name=f"UB_constraint_{i}")
            MILP_model.addGenConstrIndicator(Z[i], False, lb_i >= 1e-6)  # Z[i]=0 ⇒ lb_i ≥ 0
            MILP_model.addGenConstrIndicator(Z[i], True,  ub_i <= -1e-6)  # Z[i]=1 ⇒ ub_i ≤ -1e-6

                
        MILP_model.setObjective(quicksum(abs_delta_w[j] for j in range(d)) + abs_delta_b, GRB.MINIMIZE)        
        MILP_model.optimize()

        if MILP_model.status == GRB.OPTIMAL:
            print(f"Optimal solution found, Objective value: {MILP_model.objVal}")
            for j in range(d):
                print(f"Delta_W_{j} = {Delta_W[j].X}")
            print(f"Delta_b = {Delta_b.X}")
        
        self.solve_result = {}
        for v in MILP_model.getVars():
            if v.varName not in self.solve_result:
                self.solve_result[v.varName] = v.X
        # print(self.solve_result)
        print(self.model.layers[-1].weight, self.model.layers[-1].bias)
        for col in range(W.shape[1]):
            self.model.layers[-1].weight[0][col].data += self.solve_result[f'Delta_W[{col}]']
        self.model.layers[-1].bias[0].data += self.solve_result[f'Delta_b']
        print(self.model.layers[-1].weight, self.model.layers[-1].bias)

        self.phase2_time = time.time() - start
        # self.output()
        return MILP_model.objVal

    