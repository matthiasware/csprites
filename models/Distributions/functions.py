import random
#
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Beta


def dir_sample_simple(alpha):
    gammas = [random.gammavariate(a, 1) for a in alpha]
    norm = sum(gammas)
    return np.array([g / norm for g in gammas])


def dir_sample(alpha, n=1):
    return np.array([dir_sample_simple(alpha) for _ in range(n)])


# def beta_params(X):
#     mu = X.mean()
#     var = X.var()
#     #
#     a = ((mu * (1 - mu)) / var - 1) * mu
#     b = ((mu * (1 - mu)) / var - 1) * (1 - mu)
#     return a, b


# def beta_params2(X):
#     mu = X.mean()
#     var = X.var()
#     #
#     a = ((1 - mu) / var - (1 / mu)) * mu**2
#     b = a * (1 / mu - 1)
#     return a, b

def beta_params(X):
    mu = X.mean(axis=0)
    var = X.var(axis=0)
    #
    a = ((mu * (1 - mu)) / var - 1) * mu
    b = ((mu * (1 - mu)) / var - 1) * (1 - mu)
    return a, b


def kl_beta_beta(ab_aprx, ab_true, forward=True):
    """
    Calculates either:
        Forward KL: D_kl(P||Q)
        Reverse KL: D_kl(Q||P)
    where:
        P ... True distribution
        Q ... Approximation
    Forward:
        - Mean seeking
        - Where pdf(P) is high, pdf(Q) must be high
    Reverse:
        - Mode seeking
        - where pdf(Q) is high, pdf(P) must be high
    """
    if forward:
        p_a, p_b = ab_aprx
        q_a, q_b = ab_true
    else:
        p_a, p_b = ab_true
        q_a, q_b = ab_aprx
    #
    sum_pab = p_a + p_b
    sum_qab = q_a + q_b
    #
    t1 = q_b.lgamma() + q_a.lgamma() + (sum_pab).lgamma()
    t2 = p_b.lgamma() + p_a.lgamma() + (sum_qab).lgamma()
    t3 = (p_b - q_b) * torch.digamma(p_b)
    t4 = (p_a - q_a) * torch.digamma(p_a)
    t5 = (sum_qab - sum_pab) * torch.digamma(sum_pab)
    return t1 - t2 + t3 + t4 + t5


def kl_beta_beta_pt(p, q):
    sum_params_p = p.concentration1 + p.concentration0
    sum_params_q = q.concentration1 + q.concentration0
    t1 = q.concentration1.lgamma() + q.concentration0.lgamma() + \
        (sum_params_p).lgamma()
    t2 = p.concentration1.lgamma() + p.concentration0.lgamma() + \
        (sum_params_q).lgamma()
    t3 = (p.concentration1 - q.concentration1) * \
        torch.digamma(p.concentration1)
    t4 = (p.concentration0 - q.concentration0) * \
        torch.digamma(p.concentration0)
    t5 = (sum_params_q - sum_params_p) * torch.digamma(sum_params_p)
    return t1 - t2 + t3 + t4 + t5


# def kl_beta_beta(ab_aprx, ab_true, forward=True):
#     """
#     Calculates either:
#         Forward KL: D_kl(P||Q)
#         Reverse KL: D_kl(Q||P)
#     where:
#         P ... True distribution
#         Q ... Approximation
#     Forward:
#         - Mean seeking
#         - Where pdf(P) is high, pdf(Q) must be high
#     Reverse:
#         - Mode seeking
#         - where pdf(Q) is high, pdf(P) must be high
#     """
#     if forward:
#         p_a, p_b = ab_aprx
#         q_a, q_b = ab_true
#     else:
#         p_a, p_b = ab_true
#         q_a, q_b = ab_aprx
#     #
#     sum_pab = p_a + p_b
#     sum_qab = q_a + q_b
#     #
#     t1 = q_b.lgamma() + q_a.lgamma() + (sum_pab).lgamma()
#     t2 = p_b.lgamma() + p_a.lgamma() + (sum_qab).lgamma()
#     t3 = (p_b - q_b) * torch.digamma(p_b)
#     t4 = (p_a - q_a) * torch.digamma(p_a)
#     t5 = (sum_qab - sum_pab) * torch.digamma(sum_pab)
#     return t1 - t2 + t3 + t4 + t5


def plot_beta_pdf(dist=None, ab=None, title=None, p_file=None):
    xx = torch.linspace(0, 1, 200)[1:-1]
    if dist is None:
        dist = Beta(ab[0], ab[1])
    plt.plot(xx, torch.exp(dist.log_prob(xx)))
    a, b = float(dist.concentration0), float(dist.concentration1)
    if title is not None:
        plt.title("{} \n a={:.3f}, beta={:.3f}".format(
            title, a, b))
    else:
        plt.title("a={:.3f}, beta={:.3f}".format(a, b))
    if p_file is not None:
        plt.savefig(p_file)
    plt.show()

def beta_normalize(x, beta_mean, beta_std):
    return (x - beta_mean) / beta_std