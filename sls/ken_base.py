import torch
import contextlib
import numpy as np
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@contextlib.contextmanager
def random_seed_torch(seed, device=0):
    cpu_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        gpu_rng_state = torch.cuda.get_rng_state(0)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(gpu_rng_state, device)

#seems pretty slow like 0.1secs per iteration
def compute_grad_norm(grad_list):
    grad_norm = torch.zeros(1, device=device)# 0.
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
   # print(grad_norm)
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm

def get_grad_list(params):
    return [p.grad for p in params]


def try_sgd_update(params, step_size, params_current, grad_current):
    zipped = zip(params, params_current, grad_current)

    for p_next, p_current, g_current in zipped:
       # p_next.data[:] = p_current.data
        p_next.data = p_current - step_size * g_current

class StochLineSearchBase(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=0.1,
                 beta_b=0.9,
                 gamma=2.0,
                 reset_option=0,
                 line_search_fn="armijo"):
        params = list(params)
        super().__init__(params, {})

        self.params = params
        self.c = c
        self.beta_b = beta_b
        self.gamma = gamma
        self.init_step_size = init_step_size
        self.n_batches_per_epoch = n_batches_per_epoch
        self.line_search_fn = line_search_fn
        self.state['step'] = 0
        self.state['n_forwards'] = 0
       # self.state['n_backwards'] = []
        self.state['n_backtr'] = []
        self.budget = 0
        self.reset_option = reset_option
        self.g_norm_momentum = 0.0
        self.loss_decrease_momentum = 0.0
        self.new_epoch()


    def step(self, closure):
        # deterministic closure
        raise RuntimeError("This function should not be called")

    def update_step(self, step_size, params_current, grad_current, precond=False):
        if precond:
            self.try_sgd_precond_update(self.params, step_size, params_current, grad_current, momentum=self.momentum)
        else:
            try_sgd_update(self.params, step_size, params_current, grad_current)
        self.state["forward_passes"] += 1

    def basic_line_search(self,step_size,   params_current, grad_current, closure_deterministic, precond=False ):
        with torch.no_grad():
            losses =[]
            step_sizes = []
            for e in range(200):
                self.update_step( step_size, params_current, grad_current, precond=precond)
                loss_next = closure_deterministic()
                losses.append(loss_next)
                step_sizes.append(step_size)
                step_size = step_size * self.beta_b
                if step_size < 5e-7:
                    break

        return losses, step_sizes

    def line_search(self, step_size, params_current, grad_current,g_norm, loss, closure_deterministic, precond=False):
        self.state["forward_passes"] = 0
        with torch.no_grad():
         #   beta_large = 0.5
            beta_small = 0.999
            beta_momentum = 0.99
         #   armijo = True
            small_step_size = 5e-9
            step_size = step_size /beta_small

            #estimate g_norm for optimizers with momentum
            self.update_step( small_step_size, params_current, grad_current, precond=precond)
            loss_next = closure_deterministic()
            loss_decrease = (loss-loss_next)

         #   print("gradient according to pp_norm", g_norm)

            g_norm = loss_decrease / small_step_size
            while g_norm == 0.0:#abs(loss_decrease)  < 2e-7:
                print("g_norm is  zero, this indicates a numerical error, most likely training should be stopped, increasing min step size")
             #   print("loss decrease too small, increasing min step size")
                self.state['numerical_error'] += 1
                small_step_size = small_step_size * 10
                self.update_step( small_step_size, params_current, grad_current, precond=precond)
                loss_next = closure_deterministic()
                loss_decrease = (loss-loss_next)
                g_norm = loss_decrease / small_step_size
           # print("gradient according to estimate", g_norm)
            # if g_norm < 0.0:
            #     g_norm = 0.0
            self.g_norm_momentum = g_norm * (1-beta_momentum) + self.g_norm_momentum * beta_momentum

            self.update_step( step_size, params_current, grad_current, precond=precond)
            loss_next = closure_deterministic()
            loss_decrease = (loss-loss_next)
            self.loss_decrease_momentum_temp = loss_decrease * (1-beta_momentum) + self.loss_decrease_momentum * beta_momentum
            c = self.c if self.g_norm_momentum > 0.0 else 1.0/self.c
            #print("g_norm_momentum is smaller 0, indicates something going horribly wrong")
            for e in range(100):#
             #   print(step_size)
                if step_size < small_step_size:
                    step_size = step_size * (1.0 /  self.beta_b)
                   # print("step size too small", step_size)
                    break
                if self.loss_decrease_momentum_temp < self.g_norm_momentum * c * step_size: #armijo condition not fullfilled -> decrease step size
                    step_size = step_size * self.beta_b
                    self.update_step( step_size, params_current, grad_current, precond=precond)
                    loss_next = closure_deterministic()
                    loss_decrease = (loss-loss_next)
                    self.loss_decrease_momentum_temp = loss_decrease * (1-beta_momentum) + self.loss_decrease_momentum * beta_momentum
               #     print("decreasing because armijo not satisfied")
                else:
                    break


            self.loss_decrease_momentum = loss_decrease * (1-beta_momentum) + self.loss_decrease_momentum * beta_momentum

            self.state['loss_decrease'] = loss -loss_next
            self.state['gradient_norm'] = g_norm.item() if isinstance(g_norm, torch.Tensor) else g_norm
            self.state['c'] = loss_decrease/(g_norm*step_size)
            self.state["l_dec_momentum"] = self.loss_decrease_momentum
            self.state['average c'] = self.loss_decrease_momentum/(self.g_norm_momentum*step_size)
            self.state['loss_decrease_momentum'] = self.loss_decrease_momentum
            self.state['g_norm_momentum'] = self.g_norm_momentum
               
          #  print("found step size:", step_size)
            return step_size, loss_next



    def check_armijo_conditions(self, step_size, decrease, suff_dec, c, beta_b):
        found = 0
        sufficient_decrease = step_size * c * suff_dec
        if (decrease >= sufficient_decrease):
            found = 1
        else:
            step_size = step_size * beta_b

        return found, step_size


    def save_state(self, step_size, loss, loss_next):
       # if isinstance(step_sizes[0], torch.Tensor):
      #  step_sizes = [step_size.item() if isinstance(step_size, torch.Tensor) else step_size for step_size in step_sizes]
            #step_size = step_size.item()
     #   self.state['step_size'] = step_size
        self.state['step_size'] = step_size
        self.state['step'] += 1
        self.state['all_step_size'].append(step_size)
        self.state['all_losses'].append(loss.item())
        self.state['all_new_losses'].append(loss_next.item())
      #  self.state['grad_norm_avg']  = [a.item() for a in self.avg_gradient_norm]
     #   self.state['loss_dec_avg'] = [a.item()  if isinstance(a, torch.Tensor) else a for a in self.avg_decrease]
        self.state['n_batches'] += 1
     #   self.state['avg_step'] += step_sizes
        # if isinstance(grad_norm, torch.Tensor):
        #     grad_norm = grad_norm.item()
        # self.state['grad_norm'].append(grad_norm)

    def new_epoch(self):
        self.state['avg_step'] = 0
        self.state['semi_last_step_size'] = 0
        self.state['all_step_size'] = []
        self.state['all_losses'] = []
        self.state['grad_norm'] = []
        self.state['grad_norm_avg'] = []
        self.state['loss_dec_avg'] = []
        self.state['all_new_losses'] = []
        self.state['f_eval'] = []
        self.state['backtracks'] = 0
        self.state['n_batches'] = 0
        self.state['zero_steps'] = 0
        self.state['numerical_error'] = 0

    def gather_flat_grad(self, params):
        views = []
        for p in params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def flatten_vect(self, vect):
        views = []
        for p in vect:
            if p is None:
                view = p.new(p.numel()).zero_()
            elif p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def scale_vector(self,vector, alpha, step, eps=1e-8):
        scale = (1-alpha**(max(1, step)))
        return vector / scale
