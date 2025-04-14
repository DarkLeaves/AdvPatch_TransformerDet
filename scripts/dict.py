from utils.solver import *
from attack.methods import LinfBIMAttack, LinfMIMAttack, LinfPGDAttack, OptimAttacker, CFAOptimAttacker, TransOptimAttacker
from utils.solver.loss import *

scheduler_factory = {
    'plateau': PlateauLR,
    'cosine': CosineLR,
    'ALRS': ALRS, # This is used for most of the experiments.
    'warmupALRS': warmupALRS, # This is not used.
    'ALRS_LowerTV': ALRS_LowerTV # This is just for observational scheduler comparison to the baseline.
}

optim_factory = {
    'optim': lambda p_obj, lr: torch.optim.Adam([p_obj], lr=lr, amsgrad=True),  # default
    'cfa-optim': lambda p_obj, lr: torch.optim.Adam([p_obj], lr=lr, amsgrad=True),
    'trans-optim': lambda p_obj, lr: torch.optim.Adam([p_obj], lr=lr, amsgrad=True),
    'optim-adam': lambda p_obj, lr: torch.optim.Adam([p_obj], lr=lr, amsgrad=True),
    'optim-sgd': lambda p_obj, lr: torch.optim.SGD([p_obj], lr=lr * 100),
}

attack_method_dict = {
    "": None,
    "bim": LinfBIMAttack,
    "mim": LinfMIMAttack,
    "pgd": LinfPGDAttack,
    "optim": OptimAttacker,
    "cfa-optim": CFAOptimAttacker,
    "trans-optim": TransOptimAttacker
}

loss_dict = {
    '': None,
    "ascend-mse": ascend_mse_loss,  # for gradient sign-based method
    "descend-mse": descend_mse_loss,  # for gradient sign-based method
    "obj-tv": obj_tv_loss,  # for optim(MSE as well)
    "obj-tv-fi": obj_tv_fi_loss,
    "trans_loss": obj_tv_loss
}


def get_attack_method(attack_method: str):
    if attack_method == 'cfa-optim':
        return attack_method_dict['cfa-optim']
    if attack_method == 'trans-optim':
        return attack_method_dict['trans-optim']
    if 'optim' in attack_method:
        return attack_method_dict['optim']
    return attack_method_dict[attack_method]


MAP_PATHS = {'attack-img': 'imgs',          # visualization of the detections on adversarial samples.
             'det-lab': 'det-labels',       # Detections on clean samples.
             'attack-lab': 'attack-labels', # Detections on adversarial samples.
             'det-res': 'det-res',          # statistics
             'ground-truth': 'ground-truth'
             }
