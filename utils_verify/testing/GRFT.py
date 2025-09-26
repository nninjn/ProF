"""
This python file calls functions from experiments.py to reproduce the main experiments of our paper.
"""
import os
import numpy as np
from . import generation_utilities
import time
import random
import itertools
import copy
import torch
import torch.nn.functional as F

import sys
sys.path.append('.')

def create_image_indvs(img, num):
    indivs = []
    indivs.append(img)
    for i in range(num-1):
        indivs.append(img+1)
    return np.array(indivs)

class Population():

    def __init__(self, individuals, mutation_function, fitness_compute_function, mutate_func2,num_attribs,
                 subtotal, first_attack, seed, max_iteration, tour_size=3, cross_rate=0.5, mutate_rate=0.5, max_trials = 50, max_time=30):
        self.ground_truth = 0
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.seed =seed
        self.individuals = individuals # a list of individuals, current is numpy
        self.tournament_size = tour_size
        self.fitness = None   # a list of fitness values
        self.pop_size = len(self.individuals)
        self.subtotal = subtotal

        self.firstattack = first_attack


        self.muation_func2 = mutate_func2
        self.mutation_func = mutation_function
        self.fitness_fuc = fitness_compute_function
        self.order = []
        self.best_fitness = -1000
        self.success = 0
        self.discriminatory=np.empty(shape=(0, num_attribs))
        self.round = 0

        self.first_time_used = max_time
        self.first_iteration_used = max_iteration
        self.first_ids = 100
        # for i in range(max_trials):
        start_time = time.time()
        i = 0
        while True:

            if i >= max_iteration:
                self.round=i
                break
            if time.time()-start_time > max_time:
                self.round=i
                break
            i += 1
            results = self.evolvePopulation()
            if results is None:
                xx=""
                # print("   Total generation: %d, best fitness:%.9f"%(i, self.best_fitness))
            else:
                self.discriminatory=np.append(self.discriminatory,results[-2], axis=0)
                # print(self.discriminatory)
                self.round=i
                self.success = 1
                self.first_ids = i
                # results1[new_index_result], results2[new_index_result], array[new_index_result], r_indexes

                if self.first_time_used == max_time:
                    self.first_time_used = time.time()-start_time
                    self.first_iteration_used = i
                if self.firstattack == 1:
                    break
                else:
                    self.individuals=np.array(self.individuals)
                    self.individuals = self.muation_func2(self.individuals )
            

    def crossover(self, ind1, ind2):
        shape = ind1.shape
        ind1 = ind1.flatten()
        ind2 = ind2.flatten()
        new_ind = np.copy(ind1)

        for i in range(len(ind1)):
            if random.uniform(0, 1) < self.cross_rate:
                new_ind[i] = ind1[i]
            else:
                new_ind[i] = ind2[i]
        return np.reshape(new_ind,shape)

    def evolvePopulation(self):

        results = self.fitness_fuc(self.individuals, self.ground_truth)

        objects = results[-2]  # new_index_result
        self.fitness = results[-1] # fitness

        if len(objects) > 0:
            return results[:-1]

        """
            sorted_fitness_indexes: the ordered indexes based on fitness value
            sorted_fitness_indexes[0] is the index of individual with the best fitness
        """
        sorted_fitness_indexes = sorted(range(0, len(self.fitness)), key=lambda k: self.fitness[k], reverse=True)

        # sorted_fitness_indexes1 = sorted(range(0,self.subtotal), key=lambda k: self.fitness[k], reverse=True)
        # sorted_fitness_indexes2 = sorted(range(self.subtotal, self.subtotal*2), key=lambda k: self.fitness[k], reverse=True)
        # sorted_fitness_indexes3 = sorted(range(self.subtotal*2, len(self.fitness)), key=lambda k: self.fitness[k], reverse=True)
        """
            tournaments: randomly select a tournament from the individuals and get the indv with best fittness
            Instead of select from individuals , we select from the sorted indexes (i.e., sorted_fitness_indexes) randomly.
            sorted_fitness_indexes[order_seq1[0]] is the index of indivitual with best fitness in the selected tournament.
        """

        new_indvs = []
        sorted_fitnesses = [sorted_fitness_indexes]
        # ranges = [(0,self.pop_size)]
        ranges = [(0,self.pop_size)]
        tour_ranges = [(0, self.subtotal)]

        for j in range(len(sorted_fitnesses)):
            sorted_fitness_indexes = sorted_fitnesses[j]
            best_index = sorted_fitness_indexes[0]
            (start,end) = ranges[j]
            (tour_start,tour_end) = tour_ranges[j]
            for i in range(start,end):
                item = self.individuals[i]
                if i == best_index:  # keep best
                    new_indvs.append(item)
                else:
                    # print(tour_start,tour_end,'-------')
                    order_seq1 = np.sort(np.random.choice(np.arange(tour_start,tour_end), self.tournament_size, replace=False))
                    order_seq2 = np.sort(np.random.choice(np.arange(tour_start,tour_end), self.tournament_size, replace=False))
                    first_individual = self.individuals[sorted_fitness_indexes[order_seq1[0]]]
                    second_individual = self.individuals[
                        sorted_fitness_indexes[order_seq2[0] if order_seq2[0] != order_seq1[0] else order_seq2[1]]]
                    # Cross over
                    ind = self.crossover(first_individual, second_individual)
                    if random.uniform(0, 1) < self.mutate_rate:
                        ind = self.mutation_func(ind)
                    new_indvs.append(ind)

        self.individuals = new_indvs
        self.best_fitness = self.fitness[sorted_fitness_indexes[0]]
        return None


def similar_set(X, num_attribs, protected_attribs, constraint):
    # find all similar inputs corresponding to different combinations of protected attributes with non-protected attributes unchanged
    similar_X = []
    protected_domain = []
    for i in protected_attribs:
        protected_domain = protected_domain + [list(range(int(constraint[i][0]), int(constraint[i][1])+1))]
    all_combs = np.array(list(itertools.product(*protected_domain)))
    for i, comb in enumerate(all_combs):
        X_new = copy.deepcopy(X)
        for a, c in zip(protected_attribs, comb):
            X_new[:, a] = c
        similar_X.append(X_new)
    return torch.tensor(similar_X, dtype=torch.float32, device='cpu')

def similar_set_(X, num_attribs, protected_attribs, constraint):
    # find all similar inputs corresponding to different combinations of protected attributes with non-protected attributes unchanged
    similar_X = np.empty(shape=(0, num_attribs))
    X=np.array(X)
    protected_domain = []
    for i in protected_attribs:
        protected_domain = protected_domain + [list(range(int(constraint[i][0]), int(constraint[i][1])+1))]
    all_combs = np.array(list(itertools.product(*protected_domain)))
    for i, comb in enumerate(all_combs):
        X_new = copy.deepcopy(X)
        for a, c in zip(protected_attribs, comb):
            X_new[:, a] = c
        similar_X=np.append(similar_X,X_new,axis=0)
    return torch.tensor(similar_X, dtype=torch.float32, device='cpu')

def local_generation_random(num_attribs, l_num, g_id, protected_attribs, constraint, model, s_l, epsilon):
    # local generation phase of EIDIG
    non_protected_attribs=[]
    for attrib in range(num_attribs):
        if attrib not in protected_attribs:
            non_protected_attribs.append(attrib)
    direction = [-1, 1]
    l_id = np.empty(shape=(0, num_attribs))
    all_gen_l = np.empty(shape=(0, num_attribs))
    try_times = 0
    p=[]
    for i in range(num_attribs):
        if(i not in protected_attribs):
            p.append(i)
    gid=np.array(g_id)
    suc_iter = 0
    if(len(gid)!=0):
        for _ in range(l_num):
            g0=np.copy(gid)
            try_times += 1
            suc_iter += 1
            a = random.choice(non_protected_attribs)
            s = generation_utilities.random_pick([0.5, 0.5])
            # print(g0)
            g0[:,a] = g0[:,a] + direction[s] * s_l
            g1 = generation_utilities.clip(g0, constraint)
            similar_x1 = similar_set_(g1, num_attribs, protected_attribs, constraint)
            y_pred=(model(similar_x1)>0.5).numpy().astype('int').flatten()
            y_pred=y_pred.reshape(-1,len(g1))
            unique_columns = [len(np.unique(y_pred[:, col])) > 1 for col in range(y_pred.shape[1])]
            index = np.where(unique_columns)[0]
            l_id = np.append(l_id,g1[index],axis=0)
            gid[index] = g0[index]
        l_id = np.array(list(set([tuple(id) for id in l_id])))
    
    return l_id, all_gen_l, try_times


def get_rand_num():
    return random.randint(-5, 5)

def create_image_indvs(seed, num,num_attribs,protected_attribs,constraint):
    non_protected_attribs=[]
    indivs=[]
    indivs.append(seed)
    for attrib in range(num_attribs):
        if attrib not in protected_attribs:
            non_protected_attribs.append(attrib)
    index = np.random.choice(non_protected_attribs,2,replace=False)
    unproattr=len(non_protected_attribs)
    for i in range(num-1):
        temp=seed.copy()
        temp[index[0]]=temp[index[0]]+get_rand_num()
        temp[index[1]]=temp[index[1]]+get_rand_num()
        indivs.append(temp)
    indivs = torch.tensor(indivs, dtype=torch.float32, device='cpu')
    indivs = generation_utilities.clip(indivs, constraint)
    return np.array(indivs)


def untarget_object_func(model, num_attribs, protected_attribs, constraint, target_ratio=0.9):
    def func(indvs, ground_truth):
         # define target
        x_array = np.array(indvs)
        x_array = torch.tensor(x_array, dtype=torch.float32, device='cpu')
        index_result=[]
        fitness = []
        similar_x = similar_set_(x_array,num_attribs,protected_attribs,constraint)
        # print(similar_x)
        # 计算原始输入和变异输入的预测差异
        y_pred = model((x_array)).detach().numpy()
        # y_pred_label = y_pred>0.5).astype("int").flatten()
        pred_similar_x = model(similar_x).detach().numpy()
        pred_similar_x_label=(pred_similar_x>0.5).astype("int").flatten()
        pred_similar_x_label=pred_similar_x_label.reshape(-1,len(indvs))
        unique_columns = [len(np.unique(pred_similar_x_label[:, col])) > 1 for col in range(pred_similar_x_label.shape[1])]
        new_index_result = np.where(unique_columns)[0]
        y_pred = np.repeat(y_pred, pred_similar_x_label.shape[0], axis=0)
        # print(y_pred.shape)
        # print(pred_similar_x.shape)
        fitness = np.sum(abs(y_pred-pred_similar_x).reshape(-1,len(indvs)), axis=0)
        # print(fitness)
        return x_array[new_index_result], new_index_result, fitness
    return func


def build_mutate_func(num_attribs,protected_attribs,constraint):
    def func(indv):
        non_protected_attribs=[]
        for attrib in range(num_attribs):
            if attrib not in protected_attribs:
                non_protected_attribs.append(attrib)
        index = np.random.choice(non_protected_attribs,4,replace=False)
        mutate_indv=indv.copy()
        mutate_indv[index[0]]=mutate_indv[index[0]]+random.randint(-3,3)
        mutate_indv[index[1]] = mutate_indv[index[1]] + random.randint(-3,3)
        mutate_indv[index[2]] = mutate_indv[index[2]] + random.randint(-3,3)
        mutate_indv = torch.tensor(mutate_indv, dtype=torch.float32, device='cpu')
        mutate_indv = generation_utilities.clip(mutate_indv, constraint)
        return mutate_indv
    return func


def build_mutate_func2(num_attribs,protected_attribs,constraint):
    def func(indv):
        non_protected_attribs=[]
        for attrib in range(num_attribs):
            if attrib not in protected_attribs:
                non_protected_attribs.append(attrib)
        index = np.random.choice(non_protected_attribs,1,replace=False)
        mutate_indv=np.array(indv.copy())
        # print(mutate_indv)
        if random.random() < 0.5:
            mutate_indv[:, index] += 1
        else:
            mutate_indv[:, index] -= 1
        mutate_indv = generation_utilities.clip(mutate_indv, constraint)
        return mutate_indv
    return func



def global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, max_iter, s_g,pop_num=100):
    # global generation phase of GRFT
    g_id = np.empty(shape=(0, num_attribs))
    all_gen_g = np.empty(shape=(0, num_attribs))
    try_times = 0
    g_num = len(seeds)
    index = 0
    for i in range(g_num):
        x1 = seeds[i].copy()
        inds = create_image_indvs(x1, pop_num,num_attribs,protected_attribs,constraint)
        mutation_function = build_mutate_func(num_attribs,protected_attribs, constraint)
        build_mutate_func2_function = build_mutate_func2(num_attribs,protected_attribs, constraint)
        fitness_compute_function = untarget_object_func(model,num_attribs, protected_attribs, constraint, target_ratio=0)
        pop = Population(inds,mutation_function,fitness_compute_function,build_mutate_func2_function,num_attribs,first_attack=1,subtotal=10, max_time=1000000000, seed=x1, max_iteration=max_iter)
        try_times=try_times+pop.round
        index += pop.success
        for di in pop.discriminatory:
            g_id = np.append(g_id, [di], axis=0)
    g_id = np.array(list(set([tuple(id) for id in g_id])))
    return g_id, all_gen_g, try_times



# num_attribs, l_num, g_id, protected_attribs, constraint, model, update_interval, s_l, epsilon
def individual_discrimination_generation(X, seeds, protected_attribs, constraint, model, l_num, max_iter=10, s_g=1.0, s_l=1.0, epsilon=1e-6):
    # complete implementation of GRFT
    # return non-duplicated individual discriminatory instances generated, non-duplicate instances generated and total number of search iterations
    # benchmark=""
    num_attribs = len(X[0])
    t1=time.time()
    g_id, gen_g, g_gen_num = global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, max_iter, s_g)
    t2=time.time()
    l_id, gen_l, l_gen_num = local_generation_random(num_attribs, l_num, g_id, protected_attribs, constraint, model, s_l, epsilon)

    if(len(g_id)!=0):
        all_id = np.append(g_id, l_id, axis=0)
        g_id = torch.tensor(g_id, dtype=torch.float32, device=X[0].device)
        l_id = torch.tensor(l_id, dtype=torch.float32, device=X[0].device)
        all_id = torch.cat([g_id, l_id], dim=0)

    else:
        all_id = g_id
    all_id_nondup = np.array(list(set([tuple(id) for id in all_id])))
    all_gen_nondup = np.empty(shape=(0, num_attribs))

    mask = torch.ones(X[0].shape[0], dtype=torch.bool, device=X[0].device)
    mask[protected_attribs] = False
    if len(all_id) > 0:
        all_id_nondup = [x[mask] for x in all_id]
        all_id_nondup = torch.stack(all_id_nondup).unique(dim=0)
    else:
        all_id_nondup = torch.empty((0, num_attribs - len(protected_attribs)))

    return all_id_nondup, all_gen_nondup, g_gen_num + l_gen_num,g_id,g_gen_num,t2-t1


def generate_seeds(X, c_num=4, num_seeds = 100, fashion='Distribution'):
    num_attribs = len(X[0])
    clustered_data = generation_utilities.clustering(X, c_num)
    id_seeds = np.empty(shape=(0, num_attribs))
    for i in range(100000000):
        x_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i % c_num, fashion=fashion)
        id_seeds = np.append(id_seeds, [x_seed], axis=0)
        if len(id_seeds) >= num_seeds:
            break
    return id_seeds


def GRFT(model, protected_attribs, constraint, X_train, device='cuda:0'):

    # ---------------- run ADF ----------------
    l_num = 1000
    for ROUND in range(1, 2):
        id_seeds = generate_seeds(X_train, num_seeds=1000)

        t1 = time.time()
        ids, _, total_iter, g_id, g_iter, global_time = individual_discrimination_generation(
            X_train, id_seeds, protected_attribs, constraint, model,
            l_num, max_iter=10, s_g=1.0, s_l=1.0, epsilon=1e-6
        )
        t2 = time.time()
        num_ids = len(ids)
        global_num_ids = len(g_id)
        print("GRFT", protected_attribs,  num_ids, t2 - t1, global_num_ids, global_time, ROUND)

    return num_ids

if __name__ == "__main__":
    import os
    import time
    import random
    from datetime import datetime
    import numpy as np
    import torch
    from utils_verify.model import *
    from utils_verify.data_config import *

    data = 'compas'
    data = 'adult'
    SA = 'age'
    SA = 'sex'
    model = 'compas1'
    model = 'AC1'
    dataset = globals()[data]
    device = 'cpu'
    sens_feat = SA.split('_')
    S_dim = [dataset.sensitive_feature[s] for s in sens_feat]
    NS_dim = [dim for dim in range(dataset.params) if dim not in S_dim]

    #Here we only pass in the first sensitive attribute, which only affects the returned A (and is not used later).
    dataframe, X_train, y_train, A_train, X_test, y_test, A_test = dataset.get_data(sens_feat[0])
    full_set = np.concatenate([X_train, X_test], axis=0)
    full_set = torch.tensor(full_set, dtype=torch.float32).to(device)
    bounds = torch.tensor(dataset.input_bounds, dtype=torch.float).to(device)
    net = create_model(model, X_train.shape[1]).to(device)

    model_path = f"models_verify/{data}/{model}.pth"
    net.load_state_dict(torch.load(model_path))
    net.if_sig = True
    GRFT(net, S_dim, bounds, full_set)