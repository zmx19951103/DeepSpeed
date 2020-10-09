'''
Copyright 2019 The Microsoft DeepSpeed Team test
Copyright NVIDIA/apex
This file is adapted from NVIDIA/apex/optimizer/fused_adam and implements the LAMB optimizer
'''
import types
import torch
import importlib
import numpy as np
import time
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
# from deepspeed.pt.log_utils import logger
import torch.distributed as dist


class OnebitLambSimulate(torch.optim.Optimizer):
    """Implements LAMB algorithm. Currently GPU-only.  Requires DeepSpeed adapted Apex to be installed via
    ``python setup.py install --cuda_ext --cpp_ext``.
    For usage example please see, TODO DeepSpeed Tutorial
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes.
    https://arxiv.org/abs/1904.00962
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        max_coeff(float, optional): maximum value of the lamb coefficient (default: 10.0)
        min_coeff(float, optional): minimum value of the lamb coefficient (default: 0.01)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    def __init__(self,
                 params,
                 deepspeed=None,
                 lr=1e-3,
                 freeze_step=10000000,
                 bias_correction=True,
                 betas=(0.9,
                        0.999),
                 eps=1e-8,
                 eps_inside_sqrt=False,
                 weight_decay=0.,
                 max_grad_norm=0.,
                 max_coeff=10.0,
                 min_coeff=0.01,
                 amsgrad=False,
                 threshold=0.001,
                 update_window=1):

        if amsgrad:
            raise RuntimeError('FusedLamb does not support the AMSGrad variant.')
        defaults = dict(lr=lr,
                        bias_correction=bias_correction,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm,
                        max_coeff=max_coeff,
                        min_coeff=min_coeff)
        super(OnebitLambSimulate, self).__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1
        self.freeze_step = int(freeze_step)
        self.update_window = int(update_window)

        self.comm_time = 0.0
        self.step_time = 0.0
        self.ave_step = 1
        self.bk_time = 0.0
        self.deepspeed = deepspeed
        self.adam_freeze_key = False
        self.threshold = threshold
        self.initialize = False
        # self.freeze_step = freeze_step
        self.lamb_coeffs = []
        self.last_local_lamb_coeffs = {}
        self.last_lamb_coeffs = {}
        self.lazy_lamb_coeffs = {}
        self.accu_lamb_coeffs = {}

    def tenary_compress(self, buffer_m, error):
        buffer_m.add_(error)
        scale = torch.norm(buffer_m) / np.sqrt(buffer_m.numel())
        error.set_(buffer_m - scale * buffer_m.sign())
        buffer_m.sign_()
        buffer_m.mul_(scale)

    def step(self,
             closure=None,
             grads=None,
             output_params=None,
             scale=1.,
             grad_norms=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output params (list of tensors, optional): A reduced precision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients. (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        if grads is None:
            grads_group = [None] * len(self.param_groups)
        # backward compatibility
        # assuming a list/generator of parameter means single group
        elif isinstance(grads, types.GeneratorType):
            grads_group = [grads]
        elif type(grads[0]) != list:
            grads_group = [grads]
        else:
            grads_group = grads

        if grad_norms is None:
            grad_norms = [None] * len(self.param_groups)

        #remove the previous coeffs
        del self.lamb_coeffs[:]
        p_idx = 0

        for group, grads_this_group, grad_norm_group in zip(self.param_groups, grads_group, grad_norms):
            if grads_this_group is None:
                grads_this_group = [None] * len(group['params'])

            if grad_norm_group is None:
                grad_norm_group = [None] * len(group['params'])
            elif not isinstance(grad_norm_group, list):
                grad_norm_group = [grad_norm_group]

            bias_correction = 1 if group['bias_correction'] else 0

            for p, grad, grad_norm in zip(group['params'], grads_this_group, grad_norm_group):
                # compute combined scale factor for this group
                combined_scale = scale
                if group['max_grad_norm'] > 0:
                    # norm is in fact norm*scale
                    clip = ((grad_norm / scale) + 1e-6) / group['max_grad_norm']
                    if clip > 1:
                        combined_scale = clip * scale

                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'FusedAdam does not support sparse gradients, please consider SparseAdam instead'
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['worker_error'] = torch.zeros_like(p.data)
                    state['server_error'] = torch.zeros_like(p.data)
                    state['last_worker_error'] = torch.zeros_like(p.data)
                    state['last_server_error'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                max_coeff = group['max_coeff']
                min_coeff = group['min_coeff']

                grad = grad / combined_scale

                state['step'] += 1
                weight_norm = p.data.pow(2).sum().sqrt()
                # weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                # logger.info('I am Here')
                if self.adam_freeze_key is False:
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    # v_diff = -beta2 * exp_avg_sq + beta2 * grad * grad
                    # v_diff_buffer += v_diff.norm() / exp_avg_sq.norm() / state['tensor_size']
                    # exp_avg_sq.add_(v_diff).addcmul_(1 - beta2, grad, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    grad = None
                    # v_diff = None

                else:
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)

                    # local_update = exp_avg / (exp_avg_sq.sqrt() + group['eps'])
                    # if group['weight_decay'] > 0.0:
                    #     local_update += group['weight_decay'] * p.data
                    # local_update_norm = local_update.pow(2).sum().sqrt()
                    # local_lamb_coeff = 1.0
                    # if weight_norm != 0 and local_update_norm != 0:
                    #     local_lamb_coeff = (weight_norm / local_update_norm).item()
                    #     if local_lamb_coeff > max_coeff:
                    #         local_lamb_coeff = max_coeff
                    #     if local_lamb_coeff < min_coeff:
                    #         local_lamb_coeff = min_coeff
                    # if p_idx in self.last_local_lamb_coeffs:
                    #     state['worker_error'] = state['worker_error'] * self.last_local_lamb_coeffs[p_idx] / local_lamb_coeff
                    #     state['server_error'] = state['server_error'] * self.last_local_lamb_coeffs[p_idx] / local_lamb_coeff
                    # self.last_local_lamb_coeffs[p_idx] = local_lamb_coeff

                    # if state['step'] % self.update_window == 0 and p_idx in self.accu_lamb_coeffs:
                    #     new_lamb_coeff = 1.0
                    #     if weight_norm != 0 and self.accu_lamb_coeffs[p_idx] != 0:
                    #         new_lamb_coeff = (weight_norm / self.accu_lamb_coeffs[p_idx]).item() * self.update_window
                    #     if new_lamb_coeff > max_coeff:
                    #         new_lamb_coeff = max_coeff
                    #     if new_lamb_coeff < min_coeff:
                    #         new_lamb_coeff = min_coeff
                    #     if p_idx in self.lazy_lamb_coeffs:
                    #         state['worker_error'] = state['worker_error'] * self.lazy_lamb_coeffs[p_idx] / new_lamb_coeff
                    #         state['server_error'] = state['server_error'] * self.lazy_lamb_coeffs[p_idx] / new_lamb_coeff
                    #     self.lazy_lamb_coeffs[p_idx] = new_lamb_coeff
                    #     self.accu_lamb_coeffs[p_idx] = 0.0

                    # state['last_worker_error'].data = state['worker_error'].clone()
                    # state['last_server_error'].data = state['server_error'].clone()

                    worker_error = state['worker_error']
                    server_error = state['server_error']
                    grad = None
                    self.tenary_compress(exp_avg, worker_error)
                    dist.all_reduce(exp_avg)
                    exp_avg.mul_(1 / dist.get_world_size())
                    self.tenary_compress(exp_avg, server_error)

                update = exp_avg / (exp_avg_sq.sqrt() + group['eps'])

                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                update_norm = update.pow(2).sum().sqrt()

                # if self.adam_freeze_key:
                #     if p_idx not in self.accu_lamb_coeffs:
                #         self.accu_lamb_coeffs[p_idx] = update_norm
                #     else:
                #         self.accu_lamb_coeffs[p_idx] += update_norm
                # if self.adam_freeze_key is False or p_idx not in self.lazy_lamb_coeffs:
                #     lamb_coeff = 1.0
                #     if weight_norm != 0 and update_norm != 0:
                #         lamb_coeff = (weight_norm / update_norm).item()
                #         if lamb_coeff > max_coeff:
                #             lamb_coeff = max_coeff
                #         if lamb_coeff < min_coeff:
                #             lamb_coeff = min_coeff
                # else:
                #     lamb_coeff = self.lazy_lamb_coeffs[p_idx]

                lamb_coeff = 1.0
                if weight_norm != 0 and update_norm != 0:
                    lamb_coeff = (weight_norm / update_norm).item()
                    if lamb_coeff > max_coeff:
                        lamb_coeff = max_coeff
                    if lamb_coeff < min_coeff:
                        lamb_coeff = min_coeff

                self.lamb_coeffs.append(lamb_coeff)
                with torch.no_grad():
                    p.add_(-group['lr'] * lamb_coeff * update)
                # if self.adam_freeze_key and p_idx in self.last_lamb_coeffs:
                #     state['worker_error'] += (self.last_lamb_coeffs[p_idx]/lamb_coeff - 1) * state['last_worker_error']
                #     state['server_error'] += (self.last_lamb_coeffs[p_idx]/lamb_coeff - 1) * state['last_server_error']
                # self.last_lamb_coeffs[p_idx] = lamb_coeff
                p_idx += 1

        if self.adam_freeze_key is False:
            if state['step'] >= self.freeze_step:
                # if v_diff_buffer >= self.threshold:
                self.adam_freeze_key = True
                self.deepspeed.enable_backward_allreduce = False
        return loss

    def get_lamb_coeffs(self):
        # lamb_coeffs = [lamb_coeff.item() for lamb_coeff in self.lamb_coeffs]
        # return lamb_coeffs
        return self.lamb_coeffs
