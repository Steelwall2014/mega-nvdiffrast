import torch
from torch.optim.optimizer import _get_value, _dispatch_sqrt
from distribute import get_local_rank, get_num_gpus, log_dist
import renderutils as ru

@torch.no_grad()
def single_tensor_adam(param: torch.Tensor,
                        grad: torch.Tensor,
                        exp_avg: torch.Tensor,
                        exp_avg_sq: torch.Tensor,
                        step_t: torch.Tensor,
                        beta1: float,
                        beta2: float,
                        lr: float,
                        eps: float):
    # update step
    step_t += 1

    # Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

    step = _get_value(step_t)

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    step_size = lr / bias_correction1

    bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

    param.addcdiv_(exp_avg, denom, value=-step_size)

@torch.no_grad()
def cuda_step(
        params: list[torch.Tensor], grads: list[torch.Tensor], 
        exp_avgs: list[torch.Tensor], exp_avg_sqs: list[torch.Tensor], 
        steps: list[torch.Tensor], beta1, beta2, lr, eps, async_copy=False):
    # 假定所有params、exp_avgs、exp_avg_sqs、steps都在cpu上，而grads都在cuda上
    num_params = len(params)
    cuda_params = []
    cuda_exp_avgs = []
    cuda_exp_avg_sqs = []
    for i in range(num_params):
        param = params[i]
        step = steps[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        cuda_param = param.cuda()
        cuda_exp_avg = exp_avg.cuda()
        cuda_exp_avg_sq = exp_avg_sq.cuda()
        single_tensor_adam(cuda_param, grads[i], cuda_exp_avg, cuda_exp_avg_sq, step, beta1, beta2, lr, eps)
        if async_copy:
            cuda_params.append(cuda_param)
            cuda_exp_avgs.append(cuda_exp_avg)
            cuda_exp_avg_sqs.append(cuda_exp_avg_sq)
        else:
            param.copy_(cuda_param, non_blocking=True)
            exp_avg.copy_(cuda_exp_avg, non_blocking=True)
            exp_avg_sq.copy_(cuda_exp_avg_sq, non_blocking=True)
    if async_copy:
        cuda_tensors = cuda_params + cuda_exp_avgs + cuda_exp_avg_sqs
        cpu_tensors = params + exp_avgs + exp_avg_sqs
        return ru.async_copy_(cuda_tensors, cpu_tensors)
    return None


def cpu_step(
        params: list[torch.Tensor], grads: list[torch.Tensor], 
        exp_avgs: list[torch.Tensor], exp_avg_sqs: list[torch.Tensor], 
        steps: list[torch.Tensor], beta1, beta2, lr, eps):
    thread = ru.async_multi_tensor_adam(params, grads, exp_avgs, exp_avg_sqs, steps, beta1, beta2, lr, eps)
    return thread

class TwinFlowAdam(torch.optim.Optimizer):
    # 可以让cpu和cuda同时进行计算。
    # 尽管理论上应该有加速，但是实际上没什么效果，可能是线程切换的开销比较大。
    # 还需要进一步优化
    def __init__(self, params: list[torch.Tensor], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, share_memory=True) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        assert isinstance(params, list), "only support list of parameters"
        self.share_memory = share_memory
        for p in params:
            assert p.is_cpu, "only support cpu parameters"
            state = self.state[p]
            state['step'] = torch.zeros(1)
            state['exp_avg'] = torch.zeros_like(p)
            state['exp_avg_sq'] = torch.zeros_like(p)
            if p.is_shared() and share_memory:
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def state_pin_memory_(self):
        cudart = torch.cuda.cudart()
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                torch.cuda.check_error(cudart.cudaHostRegister(state['exp_avg'].data_ptr(), state['exp_avg'].numel() * state['exp_avg'].element_size(), 0))
                torch.cuda.check_error(cudart.cudaHostRegister(state['exp_avg_sq'].data_ptr(), state['exp_avg_sq'].numel() * state['exp_avg_sq'].element_size(), 0))
                torch.cuda.check_error(cudart.cudaHostRegister(state['step'].data_ptr(), state['step'].numel() * state['step'].element_size(), 0))
        return self

    def step(self, grads: list[torch.Tensor], async_copy=False, twin_flow_ratio=0.0):
        if len(grads) == 0:
            return
        local_rank = get_local_rank()
        num_gpus = get_num_gpus()
        params = self.param_groups[0]['params']
        local_params = []
        local_grads = []
        local_exp_avgs = []
        local_exp_avg_sqs = []
        local_steps = []
        for i, param in enumerate(params):
            if i % num_gpus == local_rank or not param.is_shared():
                state = self.state[param]
                local_params.append(param)
                local_grads.append(grads[i])
                local_exp_avgs.append(state['exp_avg'])
                local_exp_avg_sqs.append(state['exp_avg_sq'])
                local_steps.append(state['step'])

        num_local_params = len(local_params)
        num_local_params_cpu_flow = int(twin_flow_ratio * num_local_params)
        beta1, beta2 = self.param_groups[0]["betas"]
        eps = self.param_groups[0]["eps"]
        lr = self.param_groups[0]['lr']

        if num_local_params_cpu_flow != 0:
            cpu_step_thread = cpu_step(
                local_params[:num_local_params_cpu_flow],
                local_grads[:num_local_params_cpu_flow],
                local_exp_avgs[:num_local_params_cpu_flow],
                local_exp_avg_sqs[:num_local_params_cpu_flow],
                local_steps[:num_local_params_cpu_flow],
                beta1, beta2, 
                lr, 
                eps)

        cuda_step(
            local_params[num_local_params_cpu_flow:],
            local_grads[num_local_params_cpu_flow:],
            local_exp_avgs[num_local_params_cpu_flow:],
            local_exp_avg_sqs[num_local_params_cpu_flow:],
            local_steps[num_local_params_cpu_flow:],
            beta1, beta2, 
            lr, 
            eps, async_copy)
        
        if num_local_params_cpu_flow != 0:
            cpu_step_thread.join()
