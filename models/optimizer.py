import torch
import torch_optimizer as optim

class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    # print(p.shape)
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def get_optimizer(optimizer: str, parameters, optimizer_args):
    if optimizer == "sgd":
        return torch.optim.SGD(parameters, **optimizer_args)
    elif optimizer == "lars":
        return LARS(parameters, **optimizer_args)
    elif optimizer == "adam":
        return torch.optim.Adam(parameters, **optimizer_args)
    elif optimizer == "yogi":
        return optim.Yogi(parameters, **optimizer_args)
    elif optimizer == "shampoo":
        return optim.Shampoo(parameters, **optimizer_args)
    elif optimizer == "swats":
        return optim.SWATS(parameters, **optimizer_args)
    elif optimizer == "sgdw":
        return optim.SGDW(parameters, **optimizer_args)
    elif optimizer == "sgdp":
        return optim.SGDP(parameters, **optimizer_args)
    elif optimizer == "rangerva":
        return optim.RangerVA(parameters, **optimizer_args)
    elif optimizer == "rangerqh":
        return optim.RangerQH(parameters, **optimizer_args)
    elif optimizer == "ranger":
        return optim.Ranger(parameters, **optimizer_args)
    elif optimizer == "radam":
        return optim.RAdam(parameters, **optimizer_args)
    elif optimizer == "qhm":
        return optim.QHM(parameters, **optimizer_args)
    elif optimizer == "qhadam":
        return optim.QHAdam(parameters, **optimizer_args)
    elif optimizer == "pid":
        return optim.PID(parameters, **optimizer_args)
    elif optimizer == "novograd":
        return optim.NovoGrad(parameters, **optimizer_args)
    elif optimizer == "lamb":
        return optim.Lamb(parameters, **optimizer_args)
    elif optimizer == "diffgrad":
        return optim.DiffGrad(parameters, **optimizer_args)
    elif optimizer == "apollo":
        return optim.Apollo(parameters, **optimizer_args)
    elif optimizer == "aggmo":
        return optim.AggMo(parameters, **optimizer_args)
    elif optimizer == "adamp":
        return optim.AdamP(parameters, **optimizer_args)
    elif optimizer == "adafactor":
        return optim.Adafactor(parameters, **optimizer_args)
    elif optimizer == "adamod":
        return optim.AdaMod(parameters, **optimizer_args)
    elif optimizer == "adabound":
        return optim.AdaBound(parameters, **optimizer_args)
    elif optimizer == "adabelief":
        return optim.AdaBelief(parameters, **optimizer_args)
    elif optimizer == "accsgd":
        return optim.AccSGD(parameters, **optimizer_args)
    elif optimizer == "a2graduni":
        return optim.A2GradUni(parameters, **optimizer_args)
    elif optimizer == "a2gradinc":
        return optim.A2GradInc(parameters, **optimizer_args)
    elif optimizer == "a2gradexp":
        return optim.A2GradExp(parameters, **optimizer_args)
    else:
        raise Exception(f"Optimizer '{optimizer}' does not exist!")


def get_scheduler(scheduler: str, optimizer, scheduler_args):
    if scheduler == "cosine_decay":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          **scheduler_args)
    else:
        raise Exception(f"Scheduler '{scheduler}' does not exist!")
