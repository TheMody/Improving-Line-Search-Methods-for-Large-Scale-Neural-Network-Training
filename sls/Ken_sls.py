import copy
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import wandb
from .ken_base import StochLineSearchBase, get_grad_list, compute_grad_norm, random_seed_torch, try_sgd_update
from PIL import Image
from utils_plot import *

#gets a nested list of parameters as input
class KenSLS(StochLineSearchBase):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=0.1,
                 c=0.1,
                 gamma=2.0,
                 beta=0.999,
                 momentum=0.9,
                 gv_option='per_param',
                 base_opt='adam',
                 pp_norm_method='pp_armijo',
                 mom_type='standard',
                 clip_grad=False,
                 beta_b=0.9,
                 beta_s = 0.99,
                 reset_option=1,
                 timescale = 0.05,
                 line_search_fn="armijo",
                 smooth = True,
                 smooth_after = 0,
                 only_decrease = False, log = False):
        params = list(params)
        super().__init__(params,
                         n_batches_per_epoch=n_batches_per_epoch,
                         init_step_size=init_step_size,
                         c=c,
                         beta_b=beta_b,
                         gamma=gamma,
                         reset_option=reset_option,
                         line_search_fn=line_search_fn)
        self.mom_type = mom_type
        self.pp_norm_method = pp_norm_method

        self.init_step_size = init_step_size
        self.step_size = init_step_size 
        self.axlist = []
        self.smooth = smooth
        # sls stuff
        self.beta_b = beta_b
        self.beta_s = beta_s
        self.smooth_after = smooth_after
        self.only_decrease = only_decrease
        if not self.smooth_after == 0:
            self.smooth = False

        # others
        self.nextcycle = 0
        self.params = params
        if self.mom_type == 'heavy_ball':
            self.params_prev = copy.deepcopy(params) 


        self.momentum = momentum
        self.beta = beta
        self.first_step = True
        self.timescale = timescale
        self.log = log
        # self.state['step_size'] = init_step_size

      #  self.avg_decrease = torch.zeros(len(params))#(0.0 for i in range(len(params))]
       # self.avg_step_size = torch.zeros(len(params))#[0.0 for i in range(len(params))]
      #  self.avg_gradient_norm_scaled = torch.zeros(len(params))#[0.0 for i in range(len(params))]

        self.clip_grad = clip_grad
        self.gv_option = gv_option
        self.base_opt = base_opt
        # self.step_size_method = step_size_method
        self.tslls = 100
        self.avg_step_size_slow = 1e-8
        self.avg_step_size_fast = 1e-8
        self.avg_step_size  = 1e-8
        # gv options
        self.gv_option = gv_option
        if self.gv_option in ['scalar']:
            self.state['gv'] = [[0.]]

        elif self.gv_option == 'per_param':
            self.state['gv'] = [torch.zeros(p.shape).to(p.device) for p in self.params]
            self.state['mv'] = [torch.zeros(p.shape).to(p.device) for p in self.params] 
        self.state["ema_cosine_similarity"] = 0.0
        

    def step(self, closure, closure_with_backward = None):
        # deterministic closure
        seed = time.time()
        start = time.time()
        def closure_deterministic():
            with random_seed_torch(int(seed)):
                return closure()

        if closure_with_backward is not None:
            loss = closure_with_backward()
        else:
            loss = closure_deterministic()
            loss.backward()
            
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.params, 0.25)
        # increment # forward-backward calls
        self.state['n_forwards'] += 1 
        # save the current parameters:
        params_current = copy.deepcopy(self.params)
        grad_current = get_grad_list(self.params) 

       # grad_norm = [compute_grad_norm(grad) for grad in grad_current]
        # print("time for grad norm:", time.time()-start)
        # start = time.time()
        #  Gv options
        # =============
        if self.gv_option == 'per_param':
            # update gv
            for a, g in enumerate(grad_current):
                    if isinstance(g, torch.Tensor) and isinstance(self.state['gv'][a], torch.Tensor):
                        if g.device != self.state['gv'][a].device:
                            self.state['gv'][a] = self.state['gv'][a].to(g.device)
                    if isinstance(g, torch.Tensor) and isinstance(self.state['mv'][a], torch.Tensor):
                        if g.device != self.state['mv'][a].device:
                            self.state['mv'][a] = self.state['mv'][a].to(g.device)
                    if self.base_opt == 'adam':
                        self.state['gv'][a] = (1-self.beta)*(g**2) + (self.beta) * self.state['gv'][a]
                        self.state['mv'][a] = (1-self.momentum)*g + (self.momentum) * self.state['mv'][a]
                    if self.base_opt == 'lion':
                       # self.state['gv'][a][i] = (1-self.beta)*(g**2) + (self.beta) * self.state['gv'][a][i]
                        self.state['mv'][a] = (1-self.beta)*g + (self.beta) * self.state['mv'][a]


        step_size = self.step_size

        # compute step size and execute step
        # =================
        if self.smooth == False and not self.smooth_after == 0:
            if self.state['step'] > self.smooth_after:
                self.smooth = True


        # if self.state['step'] < 10:
        #     step_size , loss_next = self.line_search(step_size, params_current, grad_current, loss, closure_deterministic, precond=True)
        #     self.avg_step_size_slow = self.avg_step_size_slow * 0.99 + (step_size) *(1-0.99)
        #     self.avg_step_size_fast = self.avg_step_size_fast * 0.9 + (step_size) *(1-0.9)
        # else:
      #  print("avg step size slow:", (self.avg_step_size_slow))#/((1-0.99)**(self.state['step']+1))))
     #   print("avg step size fast:", (self.avg_step_size_fast)) # /((1-0.9)**(self.state['step']+1))))

     #efficient training
      #  if not (self.state['step'] % 10 == 0 and self.log):
        # rate_of_change = self.avg_step_size_fast /self.avg_step_size_slow
        # if rate_of_change < 1:
        #     rate_of_change = 1/rate_of_change
        # neededchecks = min(10,1/((rate_of_change-1)+ 1e-8))
    #  print("needed checks:", neededchecks)
        
       # if self.tslls > neededchecks:
        g_norm =self.get_pp_norm(grad_current)
        #  explore_step_size = step_size if self.first_step else self.avg_step_size_fast/(1-0.9**(self.state['step']+1))
        #   print(explore_step_size)
        step_size , loss_next = self.line_search(step_size, params_current, grad_current,g_norm, loss, closure_deterministic, precond=True)

        
        # if self.state['step'] % 20 == 0 or (loss- loss_next).item() < 0:
        #     with torch.no_grad():
        #         #loss_next  = torch.Tensor([0])
        #         losses, step_sizes = self.basic_line_search(5e-3, params_current, grad_current, closure_deterministic, precond=True)
                
        #         lossesdec = [(loss - l).cpu().numpy() for l in losses]
        #         losses = [ l.cpu().numpy() for l in losses]
            
        #     dict = {"step_sizes": step_sizes, "lossesdec": lossesdec, 
        #         "armijo": [self.c*self.state['gradient_norm'] * s for s in step_sizes], 
        #         "current_step": (step_size, (loss-loss_next).item()), 
        #         "optimum_step": (step_sizes[np.argmax(lossesdec)], lossesdec[np.argmax(lossesdec)]) }
        #     self.axlist.append(dict)
        #     if len(self.axlist) > 100:
        #         self.axlist.pop(0)
        #     loss_surface = np.asarray([self.axlist[i]["lossesdec"] for i in range(len(self.axlist))])

        #     #normal plot of current loss landscape
        #     plt.plot(step_sizes, lossesdec)#,'-gD', markevery = [next_pos])
        #     plt.plot(step_sizes, dict["armijo"], c = "k")
        #     plt.scatter(dict["current_step"][0], dict["current_step"][1], c = 'r')
        #     plt.scatter(dict["optimum_step"][0], dict["optimum_step"][1], c = 'g')
        #     plt.xscale('log')
        #     plt.ylim((-lossesdec[np.argmax(lossesdec)]*1.1,lossesdec[np.argmax(lossesdec)]*1.1))
        #     plt.savefig("plots/losslandscape"+ str(self.state['step']) +".png")
        #     plt.clf()

        #     #plot of current loss landscape and fading past ones
        #     len_fade = 10
        #     for i,d in enumerate(self.axlist[-len_fade:]):
        #         plt.plot(d["step_sizes"], d["lossesdec"], color = (0,0,1,(i+1)/len(self.axlist[-len_fade:])))#,'-gD', markevery = [next_pos])
        #       #  plt.plot(d["step_sizes"], d["armijo"], color = (0,0,0,(i+1)/len(self.axlist[-len_fade:])))
        #         plt.scatter(d["current_step"][0], d["current_step"][1], color = (1,0,0,(i+1)/len(self.axlist[-len_fade:])))
        #         plt.scatter(d["optimum_step"][0], d["optimum_step"][1], color = (0,1,0,(i+1)/len(self.axlist[-len_fade:])))
        #     plt.xscale('log')
        #     plt.ylim((-np.max(loss_surface[-len_fade:])*1.1,np.max(loss_surface[-len_fade:])*1.1))
        #     plt.savefig("plots/losslandscapeoverlapped"+ str(self.state['step']) +".png")
        #     plt.clf()


        #     #img plot of current loss landscape and past ones
        #     loss_surface_rgb = np.clip(loss_surface, -np.max(loss_surface), np.max(loss_surface))
        #     im = plt.imshow(loss_surface_rgb, cmap =plt.cm.RdBu )
        #     plt.colorbar(im)
        #     plt.xticks([0,len(step_sizes)//4,len(step_sizes)//2,(len(step_sizes)//4)*3,len(step_sizes)-1],[5e-3,5e-4,5e-5,5e-6,5e-7])
        #     plt.xlabel("step size")
        #     plt.ylabel("step from 0(oldest) to youngest")
        #     plt.scatter([np.argmin(np.abs(np.asarray(step_sizes)-a["current_step"][0])) for a in self.axlist], np.arange(len(self.axlist)), c = 'g')
        #     plt.savefig("plots/losslandscapeimg"+ str(self.state['step']) +".png")


        #     imgimg = Image.open("plots/losslandscapeimg"+ str(self.state['step']) +".png")
        #     imgimg = wandb.Image(imgimg, caption="losslandscapeimg " + str(self.state['step']))
        #     imglapped = Image.open("plots/losslandscapeoverlapped"+ str(self.state['step']) +".png")
        #     imglapped = wandb.Image(imglapped, caption="losslandscapeoverlapped " + str(self.state['step']))
        #     img = Image.open("plots/losslandscape"+ str(self.state['step']) +".png")
        #     img = wandb.Image(img, caption="step " + str(self.state['step'])+ " loss_change: " + str((loss-loss_next).item()))
        #     self.state["log_dict"] = {"loss landscape": img, "loss landscape img": imgimg, "loss landscape overlapped": imglapped}

        #     plt.clf()

        self.step_size = step_size
        self.try_sgd_precond_update(self.params,self.step_size, params_current, grad_current, self.momentum)

        
        self.save_state(self.step_size, loss, loss_next)

        if torch.isnan(self.params[0]).sum() > 0:
            raise ValueError('nans detected')
        
        if self.first_step:
            self.first_step = False

        return loss

 

    @torch.no_grad()
    def try_sgd_precond_update(self, params, step_size, params_current, grad_current, momentum):
        if self.gv_option in ['scalar']:
            zipped = zip(params, params_current, grad_current)

            for p_next, p_current, g_current in zipped:
                p_next.data[:] = p_current.data
                p_next.data.add_(g_current, alpha=- step_size)
        
        elif self.gv_option == 'per_param':
            if self.base_opt == 'adam':
                zipped = zip(params, params_current, grad_current, self.state['gv'], self.state['mv'])
                for p_next, p_current, g_current, gv_i, mv_i in zipped:
                    gv_i_scaled = scale_vector(gv_i, self.beta, self.state['step']+1)
                    pv_list = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)

                    if momentum == 0. or  self.mom_type == 'heavy_ball':
                        mv_i_scaled = g_current
                    elif self.mom_type == 'standard':
                        mv_i_scaled = scale_vector(mv_i, momentum, self.state['step']+1)
                    
                    p_next.data[:] = p_current.data
                    p_next.data.add_((pv_list *  mv_i_scaled), alpha=- step_size)
            elif self.base_opt == 'lion':
                zipped = zip(params, params_current, grad_current, self.state['mv'])
                for p_next, p_current, g_current, mv_i in zipped:
                    dk = torch.sign(momentum * mv_i + g_current * (1-momentum))
                    p_next.data[:] = p_current.data
                    p_next.data.add_(dk, alpha=- step_size)
            
            

            else:
                raise ValueError('%s does not exist' % self.base_opt)

        else:
            raise ValueError('%s does not exist' % self.gv_option)

    def get_pp_norm(self, grad_current):
        if self.pp_norm_method in ['pp_armijo', "just_pp"]:
            pp_norm = 0
            for g_i, gv_i, mv_i in zip(grad_current, self.state['gv'], self.state['mv']):
                if self.base_opt == 'adam':
                    gv_i_scaled = scale_vector(gv_i, self.beta, self.state['step']+1)
                    pv_i = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)
                    if self.pp_norm_method == 'pp_armijo':
                     #   sim = F.cosine_similarity(g_i, mv_i, dim = 0)
                     #   signsim = torch.sign(sim)
                        layer_norm = (((g_i)**2) * pv_i ).sum() #* F.cosine_similarity(g_i, mv_i, dim = 0)
                    elif self.pp_norm_method == "just_pp":
                        layer_norm = pv_i.sum()
                    else:
                        raise ValueError('%s not found' % self.base_opt)
                pp_norm += layer_norm
             #   pp_norm = torch.sqrt(pp_norm)
              #  pp_norms.append(layer_norm.item())

        else:
            raise ValueError('%s does not exist' % self.pp_norm_method)

        # zipped = zip( self.state['gv'], self.state['mv'])
        # update_size_list = []
        # for gv_i, mv_i in zipped:
        #   #  gv_i_scaled = scale_vector(gv_i, self.beta, self.state['step']+1)
        #     pv_list = 1. / (torch.sqrt(gv_i) + 1e-8)
        #     #mv_i_scaled = scale_vector(mv_i, momentum, self.state['step']+1)
        #     update_size_list.append(pv_list *  mv_i)

        grad_vector = torch.cat([g_i.flatten() for g_i in grad_current])
        momentum_vector = torch.cat([mv_i.flatten() for mv_i in self.state["mv"]])
        self.state["cosine_similarity"] = F.cosine_similarity(grad_vector, momentum_vector, dim = 0)
        # print("pp_norm", pp_norm)
        # print("similarity", sim)
        return pp_norm #* (sim**2)

def scale_vector(vector, alpha, step, eps=1e-8):
    scale = (1-alpha**(max(1, step)))
    return vector / scale

