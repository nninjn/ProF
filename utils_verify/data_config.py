import numpy as np
import torch
from .dataset import load_adult_ac1, load_compas, load_bank, load_german, preprocess_bank_data

class adult:
    """
    Configuration of dataset Adult Census Income
    """
    # the size of total features
    params = 13

    # the valid religion of each feature
    input_bounds = []

    range_dict = {}
    range_dict['age'] = [10, 100]
    range_dict['workclass'] = [0, 6]
    range_dict['education'] = [0, 15]
    range_dict['education-num'] = [1, 16]
    range_dict['marital-status'] = [0, 6]
    range_dict['occupation'] = [0, 13]
    range_dict['relationship'] = [0, 5]
    range_dict['race'] = [0, 4]
    range_dict['sex'] = [0, 1]
    range_dict['capital-gain'] = [0, 19]
    range_dict['capital-loss'] = [0, 19]
    range_dict['hours-per-week'] = [1, 100]
    range_dict['native-country'] = [0, 40]

    for feature in range_dict.keys():
        input_bounds.append(range_dict[feature])

    # sensitive feature 
    sensitive_feature = {"age": 0, "race": 7, "sex": 8, "gender": 8}
    eps_fair_property = {
        'A1': (11, 1), # attribute 11, eps = 1
    }

    def get_data(sens_feat):
        return load_adult_ac1(sen=sens_feat)
    
    @classmethod
    def get_eps_fair_property(cls, space, prop):
        attr = cls.eps_fair_property[prop][0]
        eps = cls.eps_fair_property[prop][1]
        space[:, attr, 0] = torch.clamp(space[:, attr, 0] - eps, min=cls.input_bounds[attr][0], max=cls.input_bounds[attr][1])
        space[:, attr, 1] = torch.clamp(space[:, attr, 1] + eps, min=cls.input_bounds[attr][0], max=cls.input_bounds[attr][1])
        return space


class compas:
    """
    Configuration of dataset compas
    """
    # the size of total features
    params = 6

    # the valid religion of each feature
    input_bounds = []

    range_dict = {}
    range_dict['Two_yr_Recidivism'] = [0, 1]
    range_dict['Number_of_Priors'] = [0, 38]
    range_dict['Age'] = [0, 1]
    range_dict['Race'] = [0, 1]
    range_dict['Female'] = [0, 1]
    range_dict['Misdemeanor'] = [0, 1]

    for feature in range_dict.keys():
        input_bounds.append(range_dict[feature])

    # sensitive feature 
    sensitive_feature = {"age": 2, "race": 3, "sex": 4, "gender": 4}


    def get_data(sens_feat):
        return load_compas(sen=sens_feat)
   


# class bank:
#     """
#     Configuration of dataset bank
#     """
#     # the size of total features
#     params = 16

#     # the valid religion of each feature
#     input_bounds = []

#     range_dict = {}
#     range_dict['age'] = [0, 1]
#     range_dict['job'] = [0, 10]
#     range_dict['marital'] = [0, 2]
#     range_dict['education'] = [0, 6]
#     range_dict['default'] = [0, 1]
#     range_dict['housing'] = [0, 1]
#     range_dict['loan'] = [0, 1]
#     range_dict['contact'] = [0, 1]
#     range_dict['month'] = [0, 11] #0, 9?
#     range_dict['day_of_week'] = [0, 6]
#     range_dict['duration'] = [0, 5000] #0, 4918?
#     range_dict['emp.var.rate'] = [-3, 1]
#     range_dict['campaign'] = [1, 50] #1, 43?
#     range_dict['pdays'] = [0, 999]
#     range_dict['previous'] = [0, 7]
#     range_dict['poutcome'] = [0, 2]


#     for feature in range_dict.keys():
#         input_bounds.append(range_dict[feature])

#     # sensitive feature 
#     sensitive_feature = {"age": 0}


#     def get_data(sens_feat):
#         return load_bank(sen=sens_feat)


# use preprocess from RULER
class bank:
    """
    Configuration of dataset bank
    """
    # the size of total features
    params = 16

    # the valid religion of each feature
    input_bounds = []

    range_dict = {
        'age': [1, 4],
        'job': [0, 10],
        'marital': [0, 2],
        'education': [0, 2],
        'default': [0, 1],
        'balance': [1, 4],
        'housing': [0, 1],
        'loan': [0, 1],
        'contact': [0, 1],
        'day': [1, 3],
        'month': [1, 4],
        'duration': [1, 4],
        'campaign': [1, 4],
        'pdays': [1, 4],
        'previous': [1, 4],
        'poutcome': [0, 2]
    }


    for feature in range_dict.keys():
        input_bounds.append(range_dict[feature])

    # sensitive feature 
    sensitive_feature = {"age": 0}
    eps_fair_property = {
        'B1': (11, 1), # attribute 11, eps = 1, duration
    }

    def get_data(sens_feat):
        return preprocess_bank_data()
    
    @classmethod
    def get_eps_fair_property(cls, space, prop):
        attr = cls.eps_fair_property[prop][0]
        eps = cls.eps_fair_property[prop][1]
        space[:, attr, 0] = torch.clamp(space[:, attr, 0] - eps, min=cls.input_bounds[attr][0], max=cls.input_bounds[attr][1])
        space[:, attr, 1] = torch.clamp(space[:, attr, 1] + eps, min=cls.input_bounds[attr][0], max=cls.input_bounds[attr][1])
        return space


class german:
    """
    Configuration of dataset german
    """
    # the size of total features
    params = 20

    # the valid religion of each feature
    input_bounds = []

    range_dict = {}
    range_dict['status'] = [0, 2]
    range_dict['month'] = [0, 80] # 4, 72?
    range_dict['credit_history'] = [0, 2]
    range_dict['purpose'] = [0, 9]
    range_dict['credit_amount'] = [0, 20000] # 250, 18424?
    range_dict['savings'] = [0, 2]
    range_dict['employment'] = [0, 2]
    range_dict['investment_as_income_percentage'] = [1, 4]
    range_dict['other_debtors'] = [0, 2]
    range_dict['residence_since'] = [1, 4]
    range_dict['property'] = [0, 2] # 0, 3?
    range_dict['age'] = [0, 1]
    range_dict['installment_plans'] = [0, 2]
    range_dict['housing'] = [0, 2]
    range_dict['number_of_credits'] = [1, 4]
    range_dict['skill_level'] = [0, 3]
    range_dict['people_liable_for'] = [1, 2]
    range_dict['telephone'] = [0, 1]
    range_dict['foreign_worker'] = [0, 1]
    range_dict['sex'] = [0, 1]


    for feature in range_dict.keys():
        input_bounds.append(range_dict[feature])

    # sensitive feature 
    sensitive_feature = {"age": 11, "sex": 19}
    eps_fair_property = {
        'G1': (4, 100), # attribute 4, eps = 100
        'G2': (18, 1), # attribute 18, eps = 1
    }


    def get_data(sens_feat):
        return load_german(sen=sens_feat)

    
    @classmethod
    def get_eps_fair_property(cls, space, prop):
        attr = cls.eps_fair_property[prop][0]
        eps = cls.eps_fair_property[prop][1]
        space[:, attr, 0] = torch.clamp(space[:, attr, 0] - eps, min=cls.input_bounds[attr][0], max=cls.input_bounds[attr][1])
        space[:, attr, 1] = torch.clamp(space[:, attr, 1] + eps, min=cls.input_bounds[attr][0], max=cls.input_bounds[attr][1])
        return space


def prepare_repair_data(num, X, y, A, device):
    num = min(num, len(X))
    indices = np.random.choice(len(X), num, replace=False)
    X_repair = torch.from_numpy(X[indices]).to(device).float()
    y_repair = y[indices]
    A_repair = torch.from_numpy(A[indices]).to(device).float()
    return X_repair, y_repair, A_repair