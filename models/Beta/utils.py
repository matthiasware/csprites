# import numpy as np
# import matplotlib.pyplot as plt
# import torch


# def off_diagonal(x):
#     # return a flattened view of the off-diagonal elements of a square matrix
#     n, m = x.shape
#     assert n == m
#     return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# def calc_lambda(d):
#     return 1 / ((d - 1) * 0.0244)


# def beta_plot1d(x, figsize=(10, 1)):
#     x = np.array(x).squeeze()
#     assert len(x.shape) == 1
#     plt.figure(figsize=figsize)
#     plt.scatter(x, np.ones(len(x)))
#     plt.plot([0, 1], [1, 1])
#     plt.xlim([-0.2, 1.2])
#     plt.show()


# def plot_beta_pdf(dist, title=None, p_file=None):
#     xx = torch.linspace(0, 1, 200)[1:-1]
#     plt.plot(xx, torch.exp(dist.log_prob(xx)))
#     a, b = float(dist.concentration0), float(dist.concentration1)
#     if title is not None:
#         plt.title("{} \n a={:.3f}, beta={:.3f}".format(
#             title, a, b))
#     else:
#         plt.title("a={:.3f}, beta={:.3f}".format(a, b))
#     if p_file is not None:
#         plt.savefig(p_file)
#         plt.close()
#     else:
#         plt.show()


# def beta_params(X):
#     mu = X.mean(axis=0)
#     var = X.var(axis=0)
#     #
#     a = ((mu * (1 - mu)) / var - 1) * mu
#     b = ((mu * (1 - mu)) / var - 1) * (1 - mu)
#     return a, b


# def kl_beta_beta_pt(p, q):
#     sum_params_p = p.concentration1 + p.concentration0
#     sum_params_q = q.concentration1 + q.concentration0
#     t1 = q.concentration1.lgamma() + q.concentration0.lgamma() + \
#         (sum_params_p).lgamma()
#     t2 = p.concentration1.lgamma() + p.concentration0.lgamma() + \
#         (sum_params_q).lgamma()
#     t3 = (p.concentration1 - q.concentration1) * \
#         torch.digamma(p.concentration1)
#     t4 = (p.concentration0 - q.concentration0) * \
#         torch.digamma(p.concentration0)
#     t5 = (sum_params_q - sum_params_p) * torch.digamma(sum_params_p)
#     return t1 - t2 + t3 + t4 + t5


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
