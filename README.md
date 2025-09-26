# ProF: Provable Fairness Repair for Deep Neural Networks

This repository contains the code and scripts for the paper "Provable Fairness Repair for Deep Neural Networks", accepted in ASE 2025.

## Project Structure
```
.
├── data               # datasets used in experiments
├── models_verify      # models used in experiments
├── models_repair      # repaired models
├── results            # logs and outputs from experiments
├── script             # scripts to run our method
└── utils_verify       # utility functions
```

## Requirements

To run the code, please ensure the following dependencies are installed:

- Python 3.9.19
- PyTorch 2.3.1
- auto_LiRPA: You can install auto_LiRPA from [here](https://github.com/Verified-Intelligence/auto_LiRPA).
- Gurobi (gurobipy 11.0.2, Reproducing experiments requires a Gurobi academic license.)

## Reproduce the Experiments

### Repairing with a Single Sensitive Attribute

You can run the following command:
```    
bash script/main.sh <dataset>
```
- `<dataset>`: specifies the dataset.

### Repairing with Multiple Sensitive Attributes

You can run the following command:
```
bash script/multi.sh <dataset>
```
### Repairing with Relaxed Fairness Constraints

You can run the following command:
```
bash script/eps.sh <dataset>
```
### Example

For example, you can run:
```
bash script/main.sh adult
```
to repair a biased model on the "Adult" dataset. The resulting logs will be saved in:
- `results/adult_sex.log`
- `results/adult_race.log`
- `results/adult_age.log`

### Results of Fairness Testing

We provide the implementations of the four fairness testing tools used in our experiments in `utils_verify/testing`.

