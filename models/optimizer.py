import torch
import torch_optimizer as optim


def get_optimizer(optimizer: str, model, optimizer_args):

    if type(model) == list:
        parameters = [param  for m in model for param in m.parameters()]
    else:
        parameters = model.parameters()
    if optimizer == "sgd":
        return torch.optim.SGD(parameters, **optimizer_args)
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
