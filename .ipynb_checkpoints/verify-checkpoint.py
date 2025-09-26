import copy
import random
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import sys
sys.path.append('/root/xinglin-data/auto_LiRPA')
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


class ProductNN(nn.Module):
    def __init__(self, net, ns_num, s_ind):
        super(ProductNN, self).__init__()
        self.net = net
        if hasattr(self.net, 'if_sig'):
            self.net.if_sig = False
        else:
            raise AttributeError("Error: The net is missing the `if_sig` attribute.") 
        self.NS_NUM = ns_num
        self.S_NUM = len(s_ind)
        self.dim = self.NS_NUM + self.S_NUM

        # define the indices of input for net1 and net2
        self.x1_indices = list(range(self.dim))
        self.x2_indices = [s_ind.index(i) + self.dim if i in s_ind else i for i in range(self.dim)]
        print(self.x1_indices)
        print(self.x2_indices)
        self.relu = nn.ReLU()

    def forward(self, x):
        assert x.shape[1] == self.NS_NUM + self.S_NUM * 2, \
            f"Error in shape: got {x.shape}, expected dim = {self.S_NUM + self.NS_NUM} + {self.S_NUM}"
        x1 = x[:, self.x1_indices]
        x2 = x[:, self.x2_indices]
        out1 = self.net(x1)
        out2 = self.net(x2)
        output = torch.cat([out1, out2], dim=1)
        # output = out1 * out2
        # output = torch.abs(out1) + torch.abs(out2) - torch.abs(out1 + out2)
        # output = self.relu(out1) + self.relu(-out1) + self.relu(out2) + self.relu(-out2) - self.relu(out1 + out2) - self.relu(-out1 - out2)
        return output


class Fair():
    def __init__(self, approximate_method, input_space, ns_dim, device, verify_batch=1000000) -> None:

        self.device = device  
        self.verify_method = approximate_method
        self.full_space = input_space
        self.dim = input_space.shape[0]
        self.ns_dim = ns_dim
        self.s_dim = [i for i in range(self.dim) if i not in self.ns_dim]
        self.verify_batch = verify_batch

        self.partitions = {
                        'spaces': input_space.unsqueeze(0).to(device),
                        # satisfy = 0 & violate = 0 ==> unknown
                        'satisfy': torch.zeros(1, dtype=torch.bool).to(device),
                        # TODO how to determine violate?
                        'violate': torch.zeros(1, dtype=torch.bool).to(device),
                        # shape of impact: (num of spaces) * (dim of spaces)
                        'impact': torch.zeros(1, self.dim, dtype=torch.float).to(device),
                        'bounds_diff': torch.zeros(1, dtype=torch.float).to(device),
                        'can_split': torch.ones(1, dtype=torch.bool).to(device),
                        }


    def multi_properties_verify(self, model, verbose=False):
        """ Verify multiple properties in a given property set
            And update set['satisfy'] """
        sat = 0
        print(f"Verify {len(self.partitions['spaces'])} properties")
        for i in range(0, len(self.partitions['spaces']), self.verify_batch):
            input_lb = self.partitions['spaces'][i: i + self.verify_batch, ..., 0].detach().clone()
            input_ub = self.partitions['spaces'][i: i + self.verify_batch, ..., 1].detach().clone()
            # print(i, i + self.verify_batch)
            symbolic_net = BoundedModule(model, torch.empty_like(input_lb), device=self.device)
            ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
            true_input = BoundedTensor(torch.empty_like(input_lb), ptb)

            required_A = defaultdict(set)
            required_A[symbolic_net.output_name[0]].add(symbolic_net.input_name[0])
            
            if 'Optimized' in self.verify_method:
                symbolic_net.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.01, }})
            lb, ub, A_dict = symbolic_net.compute_bounds(x=(true_input,), method=self.verify_method, return_A=True, needed_A_dict=required_A)

            lower_A = A_dict[symbolic_net.output_name[0]][symbolic_net.input_name[0]]['lA']
            upper_A = A_dict[symbolic_net.output_name[0]][symbolic_net.input_name[0]]['uA']

            satisfy_batch = ((ub[:, 0] < 0) | (lb[:, 0] >= 0))
            sat += satisfy_batch.sum().item()

            self.partitions['satisfy'][i:i + self.verify_batch] = satisfy_batch
            self.partitions['impact'][i:i + self.verify_batch] = torch.abs(lower_A[:, 0]) + torch.abs(upper_A[:, 0])
            self.partitions['bounds_diff'][i:i + self.verify_batch] = (ub - lb).squeeze(-1)
        return sat


    @torch.no_grad()
    def property_refine(self, method='refine_score'):
        # filter out the partitions that are already satisfied or cannot be split
        mask = (~self.partitions['satisfy'] & self.partitions['can_split'])
        # calculate score for refinement, shape: space_num * space_dim
        if method == 'mag':
            score = self.partitions['spaces'][mask][..., 1] - self.partitions['spaces'][mask][..., 0]
        elif method == 'refine_score':
            input_range = self.partitions['spaces'][mask][..., 1] - self.partitions['spaces'][mask][..., 0]
            score = torch.abs(input_range * self.partitions['impact'][mask])
        # senstive dims cannot be split
        senstive_dims = torch.tensor(self.s_dim, dtype=torch.long, device=score.device)
        score[..., senstive_dims] = -float('inf')

        spaces_to_split = self.partitions['spaces'][mask]
        split_num = mask.sum().item()
        p1_spaces = spaces_to_split.clone()
        p2_spaces = spaces_to_split.clone()

        candidate_dims = torch.argmax(score, dim=-1)

        split_values = (spaces_to_split[torch.arange(spaces_to_split.shape[0]), candidate_dims, 0] +
                        spaces_to_split[torch.arange(spaces_to_split.shape[0]), candidate_dims, 1]) // 2
        # print(candidate_dims, split_values)

        p1_spaces[torch.arange(spaces_to_split.shape[0]), candidate_dims, 1] = split_values
        p2_spaces[torch.arange(spaces_to_split.shape[0]), candidate_dims, 0] = split_values + 1

        self.partitions['spaces']      = torch.cat([self.partitions['spaces'][~mask], p1_spaces, p2_spaces], dim=0)
        self.partitions['satisfy']     = torch.cat([self.partitions['satisfy'][~mask], 
                                                    torch.zeros(2 * split_num, dtype=torch.bool).to(self.device)], dim=0)
        self.partitions['impact']      = torch.cat([self.partitions['impact'][~mask], 
                                                    torch.zeros(2 * split_num, self.dim, dtype=torch.float).to(self.device)], dim=0)
        self.partitions['bounds_diff'] = torch.cat([self.partitions['bounds_diff'][~mask], 
                                                    torch.zeros(2 * split_num, dtype=torch.float).to(self.device)], dim=0)
        self.partitions['can_split']   = torch.cat([self.partitions['can_split'][~mask], 
                                                    torch.zeros(2 * split_num, dtype=torch.bool).to(self.device)], dim=0)

    def symbolic_bounds(self, model, space):
        input_lb = space[..., 0].detach().clone()
        input_ub = space[..., 1].detach().clone()

        symbolic_net = BoundedModule(model, torch.empty_like(input_lb), device=self.device)
        ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
        true_input = BoundedTensor(torch.empty_like(input_lb), ptb)

        required_A = defaultdict(set)
        required_A[symbolic_net.output_name[0]].add(symbolic_net.input_name[0])
        
        if 'Optimized' in self.verify_method:
            symbolic_net.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.01, }})
        lb, ub, A_dict = symbolic_net.compute_bounds(x=(true_input,), method=self.verify_method, return_A=True, needed_A_dict=required_A)

        lower_A = A_dict[symbolic_net.output_name[0]][symbolic_net.input_name[0]]['lA']
        upper_A = A_dict[symbolic_net.output_name[0]][symbolic_net.input_name[0]]['uA']

        return lb, ub, lower_A, upper_A

    def certify(self, model, total_spaces, repair=False, opt=None):
        if hasattr(model, 'if_sig'):
            # ignore the sigmoid to reduce errors in verification
            model.if_sig = False
        sat_num = self.multi_properties_verify(model)
        p_num = len(self.partitions['satisfy'])
        while sat_num != p_num and (p_num <= total_spaces):
            print(f"{sat_num=}, {p_num=}")
            if sat_num < p_num:
                self.property_refine(method='refine_score')
                p_num = len(self.partitions['satisfy'])
            print(f"{sat_num=}, {p_num=}")
            sat_num = self.multi_properties_verify(model)
            if repair:
                self.tighten_bounds(model, optimizer=opt)
                sat_num = self.multi_properties_verify(model)
            cert_rate = self.quantify()
            # All potentially unsatisfiable spaces cannot be split.
            if not repair and torch.sum(self.partitions['can_split'] & ~self.partitions['satisfy']) == 0:
                break
        return cert_rate

    def repair(self, model, epoch_num, optimizer):
        # TODO Is setting the input range of the sensitive attribute (SA) to 0 equivalent to removing that dimension?
        for i in range(epoch_num):
            optimizer.zero_grad()
            not_satisfy = ~self.partitions['satisfy']
            space = self.partitions['spaces'][not_satisfy]
            lb, ub, lA, uA = self.symbolic_bounds(model=model, space=space)
            diff1 = (ub - lb).mean()
            diff1.backward()

            non_SA_space = self.partitions['spaces'][not_satisfy].detach().clone()
            non_SA_space[:, self.s_dim, :] = non_SA_space[:, self.s_dim, :] * 0.
            nlb, nub, nlA, nuA = self.symbolic_bounds(model=model, space=non_SA_space)


            # area1: ub + (ub - uA * I) - nub - (nub - nuA * I) = 2 * (ub - nub) - I * (uA - nuA)
            # area2: nlb + (nlb + nlA * I) - lb - (lb + lA * I) = 2 * (nlb - lb) - I * (nlA - lA)
            
            diff2 = (- nub + nlb).mean()
            diff2.backward()
            optimizer.step()
            sat_num = self.multi_properties_verify(model)
            cert_rate = self.quantify()
            print(f"Repair Epoch {i}, Cert: {cert_rate}")


    def quantify(self):
        ori_space = (self.full_space[..., 1] - self.full_space[..., 0] + 1).cpu().numpy().astype(np.float64)
        ori_vol = np.prod(ori_space[self.ns_dim].reshape(-1))
        satisfy = self.partitions['satisfy']
        sat_vol = np.sum(self.calculate_volume(self.partitions['spaces'][satisfy]))
        other_vol = np.sum(self.calculate_volume(self.partitions['spaces'][~satisfy]))
        total_vol = self.calculate_volume(self.partitions['spaces'])
        self.partitions['can_split'] = (torch.tensor(total_vol, dtype=torch.float32, device=self.device) > 1.)
        total_vol = np.sum(total_vol)
        print(f"Vol before split: {ori_vol} Total vol: {total_vol}, sat: {sat_vol}, other: {other_vol}, Cert: {100 * sat_vol / total_vol:.2f}%")
        assert ori_vol == total_vol, f"Error in refinement? Original full vol: {ori_vol}, after split: {total_vol}"
        return 100 * sat_vol / total_vol

    def calculate_volume(self, spaces):
        space_lengths = (spaces[..., 1] - spaces[..., 0] + 1).cpu().numpy().astype(np.float64)
        vol = [np.prod(space_lengths[i][self.ns_dim]) for i in range(len(space_lengths))]
        return vol
    
if __name__ == '__main__':
    from utils_verify.model import *
    from utils_verify.data_config import *
    net = compas4(6)
    net.load_state_dict(torch.load(f"models_verify/compas/compas4.pth"))
    net.if_sig = False
    space = torch.tensor([[[ 0.,  1.],
                            [ 0., 38.],
                            [ 0.,  1.],
                            [ 0.,  1.],
                            [ 0.,  1.],
                            [ 0.,  1.]]], device='cuda:0')
    input_lb = space[:, ..., 0]
    input_ub = space[:, ..., 1]
    symbolic_net = BoundedModule(net, torch.empty_like(input_lb), device='cuda:0')
    ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
    true_input = BoundedTensor(torch.empty_like(input_lb), ptb)
    lb, ub = symbolic_net.compute_bounds(x=(true_input,), method='backward')
    print(lb, ub)

    space = torch.tensor([[[ 0.,  1.],
                            [ 0., 19.],
                            [ 0.,  1.],
                            [ 0.,  1.],
                            [ 0.,  1.],
                            [ 0.,  1.]]], device='cuda:0')
    input_lb = space[:, ..., 0]
    input_ub = space[:, ..., 1]
    ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
    true_input = BoundedTensor(torch.empty_like(input_lb), ptb)
    lb, ub = symbolic_net.compute_bounds(x=(true_input,), method='backward')
    print(lb, ub)

    space = torch.tensor([[[ 0.,  1.],
                            [ 20., 38.],
                            [ 0.,  1.],
                            [ 0.,  1.],
                            [ 0.,  1.],
                            [ 0.,  1.]]], device='cuda:0')
    input_lb = space[:, ..., 0]
    input_ub = space[:, ..., 1]
    ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
    true_input = BoundedTensor(torch.empty_like(input_lb), ptb)
    lb, ub = symbolic_net.compute_bounds(x=(true_input,), method='backward')
    print(lb, ub)

    space = torch.tensor([[[ 0.,  1.],
                            [ 0., 19.],
                            [ 0.,  1.],
                            [ 0.,  0.],
                            [ 0.,  1.],
                            [ 0.,  1.]]], device='cuda:0')
    input_lb = space[:, ..., 0]
    input_ub = space[:, ..., 1]
    ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
    true_input = BoundedTensor(torch.empty_like(input_lb), ptb)
    lb, ub = symbolic_net.compute_bounds(x=(true_input,), method='backward')
    print(lb, ub)

    space = torch.tensor([[[ 0.,  1.],
                            [ 20., 38.],
                            [ 0.,  1.],
                            [ 0.,  0.],
                            [ 0.,  1.],
                            [ 0.,  1.]]], device='cuda:0')
    input_lb = space[:, ..., 0]
    input_ub = space[:, ..., 1]
    ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
    true_input = BoundedTensor(torch.empty_like(input_lb), ptb)
    lb, ub = symbolic_net.compute_bounds(x=(true_input,), method='backward')
    print(lb, ub)
