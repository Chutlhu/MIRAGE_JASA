import numpy as np
import torch

from statsmodels.distributions.empirical_distribution import ECDF
from torch.autograd import Variable, Function


def softargmax(x, beta = 1, n_out = 1, support = None):
    if support is None:
        support = Variable(torch.from_numpy(np.arange(1,x.shape[-1]+1))).float()
    return torch.sum(support*torch.exp(beta*x)/torch.sum(torch.exp(beta*x),-1)[:,:,None],-1)

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

class WassersteinLossStab(Function):
    def __init__(self,cost, lam = 1e-3, sinkhorn_iter = 50):
        super(WassersteinLossStab,self).__init__()

        # cost = matrix M = distance matrix
        # lam = lambda of type float > 0
        # sinkhorn_iter > 0
        # diagonal cost should be 0
        self.cost = cost
        self.lam = lam
        self.sinkhorn_iter = sinkhorn_iter
        self.na = cost.size(0)
        self.nb = cost.size(1)
        self.K = torch.exp(-self.cost/self.lam)
        self.KM = self.cost*self.K
        self.stored_grad = None

    def forward(self, pred, target):
        """pred:   Batch * K: K = # mass points
           target: Batch * L: L = # mass points"""

        assert pred.size(1)==self.na
        assert target.size(1)==self.nb

        batch_size = pred.size(0)

        log_a, log_b = torch.log(pred), torch.log(target)
        log_u = self.cost.new(batch_size, self.na).fill_(-np.log(self.na)) # B x K
        log_v = self.cost.new(batch_size, self.nb).fill_(-np.log(self.nb)) # B x L

        for i in range(self.sinkhorn_iter):
            log_u_max = torch.max(log_u, dim=1)[0]       # B x 1
            u_stab = torch.exp(log_u-log_u_max[:,None])  # B x K
            log_v = log_b - torch.log(torch.mm(self.K.t(),u_stab.t()).t()) - log_u_max[:,None]
            log_v_max = torch.max(log_v, dim=1)[0]
            v_stab = torch.exp(log_v-log_v_max[:,None])
            log_u = log_a - torch.log(torch.mm(self.K, v_stab.t()).t()) - log_v_max[:,None]

        log_v_max = torch.max(log_v, dim=1)[0]
        v_stab = torch.exp(log_v-log_v_max[:,None])
        logcostpart1 = torch.log(torch.mm(self.KM,v_stab.t()).t())+log_v_max[:,None]
        wnorm = torch.exp(log_u+logcostpart1).mean(0).sum() # sum(1) for per item pair loss...
        grad = log_u*self.lam
        grad = grad-torch.mean(grad,dim=1)[:,None]
        grad = grad-torch.mean(grad,dim=1)[:,None] # does this help over only once?
        grad = grad/batch_size

        self.stored_grad = grad

        return self.cost.new((wnorm,))

    def backward(self, grad_output):
        #print (grad_output.size(), self.stored_grad.size())
        #print (self.stored_grad, grad_output)
        res = grad_output.new()
        res.resize_as_(self.stored_grad).copy_(self.stored_grad)
        if grad_output[0] != 1:
            res.mul_(grad_output[0])
        return res,None
