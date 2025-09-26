import os
import time
import random
import logging
import argparse
from datetime import datetime
import numpy as np
import torch

from utils_verify.model import *
from utils_verify.data_config import *
from utils_verify.eval import evaluation, my_idi_test

from repair import FairRepair

parser = argparse.ArgumentParser(description='Fairness Repair')
parser.add_argument('--dataset', default='adult', type=str)
parser.add_argument('--model', default='AC1', type=str)
parser.add_argument('--model_path', default=None, type=str)
parser.add_argument('--SA', default='sex', type=str)
parser.add_argument('--device', default='cuda:1', type=str)
parser.add_argument('--data_num', default=500, type=int)
parser.add_argument('--idi_num', default=100, type=int)
parser.add_argument('--eps_prop', default="", help='whether using epslion-fairness, "" means no')
parser.add_argument('--seed', default=2025, type=int)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


log = logging.getLogger('my_unique_logger')
log.setLevel(logging.DEBUG)
log_path = f"results/ProF/"
os.makedirs(log_path, exist_ok=True)
log_path = f"{log_path}/{args.dataset}_{args.SA}"
log_path += '.log' if args.eps_prop == "" else f"_{args.eps_prop}.log"
file_handler = logging.FileHandler(log_path, "a")
file_handler.setLevel(logging.DEBUG)
log.addHandler(file_handler)
now = datetime.now()
now_time = now.strftime("%Y-%m-%d %H:%M:%S")
log.info(f"Time: {now_time}")
log.info(f"Repair Model {args.model}")
log.info(f"Repair Dataset {args.dataset}")
log.info(f"Repair SA {args.SA}")
log.info(f"Repair eps_prop {args.eps_prop}")
log.info(f"Repair num: {args.data_num=}, {args.idi_num=}")
log.info("Before Repair:")

device = args.device
dataset = globals()[args.dataset]
sens_feat = args.SA.split('_')
S_dim = [dataset.sensitive_feature[s] for s in sens_feat]
NS_dim = [dim for dim in range(dataset.params) if dim not in S_dim]

#Here we only pass in the first sensitive attribute, which only affects the returned A (and is not used later).
dataframe, X_train, y_train, A_train, X_test, y_test, A_test = dataset.get_data(sens_feat[0])
full_set = np.concatenate([X_train, X_test], axis=0)
X_normal, y_normal, A_normal = prepare_repair_data(args.data_num, X_train, y_train, A_train, device)
print(torch.sum(X_normal), X_normal.shape)
net = create_model(args.model, X_train.shape[1]).to(device)
model_path = f"models_verify/{args.dataset}/{args.model}.pth" if args.model_path is None else args.model_path
net.load_state_dict(torch.load(model_path, map_location=device))
net.if_sig = False

# net.layers[0].weight[:, S_dim[0]].data *= 0

bounds = torch.tensor(dataset.input_bounds, dtype=torch.float).to(device)
# net = ProductNN(net=net, ns_num=NS_num, s_ind=S_dim)
# input_lb = torch.cat([bounds[:, 0], bounds[S_dim, 0].detach().clone()])
# input_ub = torch.cat([bounds[:, 1], bounds[S_dim, 1].detach().clone()])
# NS_num = dataset.params - 1
input_lb = bounds[:, 0].detach().clone()
input_ub = bounds[:, 1].detach().clone()
space = torch.stack([input_lb, input_ub], dim=-1)
full_space = space.detach().clone()


X_repair, y_idi, _ = prepare_repair_data(args.idi_num, X_train, y_train, A_train, device)
print(X_repair.shape, torch.sum(X_repair))

FR = FairRepair(model=net, input_space=space, X_repair=X_repair, ns_dim=NS_dim, device=device)

if args.eps_prop != "":
    FR.X_spaces = dataset.get_eps_fair_property(FR.X_spaces, args.eps_prop)


epoch_num = 200 if args.eps_prop == "" else 200

evaluation(model=net, dataset=args.dataset, X_test=X_test, y_test=y_test, A_test=A_test, full_set=full_set, 
            eps_prop=args.eps_prop, X_spaces=FR.X_spaces, sen=sens_feat, constraints=bounds, device=device, logger=log)

FR.minimize_Hc(X_normal=X_normal, y_normal=y_normal, epochs=epoch_num)
# evaluation(model=net, dataset=args.dataset, X_test=X_test, y_test=y_test, A_test=A_test, full_set=full_set,
#             eps_prop=args.eps_prop, X_spaces=FR.X_spaces, sen=sens_feat, device=device)
FR.symbolic_constraint_solve()

log.info(f"Step1 epochs {epoch_num}")
log.info("-" * 50 + 'After Repair' + "-" * 50)
evaluation(model=net, dataset=args.dataset, X_test=X_test, y_test=y_test, A_test=A_test, full_set=full_set, 
            eps_prop=args.eps_prop, X_spaces=FR.X_spaces, sen=sens_feat, constraints=bounds, device=device, logger=log)
cost = FR.phase1_time + FR.phase2_time
log.info(f"Repair time {cost:.2f}s")
log.info(f"Step1 time {FR.phase1_time:.2f}s")
log.info(f"Step2 time {FR.phase2_time:.2f}s")
log.info('#' * 100)

if args.eps_prop == "":
    save_path = f"models_repair/ProF/{args.dataset}_{args.SA}.pth"
else:
    save_path = f"models_repair/ProF/{args.dataset}_{args.SA}_{args.eps_prop}.pth"

os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(net.state_dict(), save_path)
