import torch
import math  # if you're using it

# Custom AdamWTrustRegion optimizer
class AdamWTR(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                 alpha=0.01, gamma_clip=(0.1, 10.0), decay_rate=0.99, decay_steps=100):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        alpha=alpha, gamma_clip=gamma_clip, decay_rate=decay_rate, decay_steps=decay_steps)
        super(AdamWTR, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                step = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = group["lr"] * (group["decay_rate"] ** (step / group["decay_steps"]))
                step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

                rho = (grad * grad).sum() / (grad.norm() ** 2 + 1e-8)
                gamma_t = 1 / (1 + group["alpha"] * rho)
                #gamma_t = torch.clamp(torch.tensor(gamma_t), group["gamma_clip"][0], group["gamma_clip"][1])
                gamma_t = torch.clamp(gamma_t.clone().detach().requires_grad_(True), group["gamma_clip"][0], group["gamma_clip"][1])
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size * gamma_t.item())

                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-step_size * group["weight_decay"])

        return loss