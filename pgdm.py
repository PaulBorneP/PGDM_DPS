import torch.nn.functional as F
import torch
import torchvision
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from guided_diffusion.unet import create_model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device:", device)

class PGDM_DDIM:
  def __init__(self,H, model, classifier = None, eta = 0.5, inpainting = True,num_diffusion_timesteps=100,start_noise_level_noise=None):
    self.num_diffusion_timesteps = num_diffusion_timesteps
    self.reversed_time_steps = np.arange(self.num_diffusion_timesteps)[::-1]
    self.s_list = list(self.reversed_time_steps[1:]) + [-1]
    beta_start = 0.0001
    beta_end = 0.02
    self.betas = np.linspace(beta_start, beta_end, self.num_diffusion_timesteps,
                              dtype=np.float64)
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
    self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
    self.model = model
    self.imgshape = (1,3,256,256)
    self.eta = eta
    self.H = H
    self.inpainting = inpainting
    self.classifier = classifier
    self.start_noise_level_noise = start_noise_level_noise

  def cond_fn(self, x,t, y):
      with torch.enable_grad():
          x_in = x.detach().requires_grad_(True)
          logits = self.classifier(x_in, t)
          log_probs = F.log_softmax(logits, dim=-1)
          selected = log_probs[range(len(logits)), y.view(-1)]

          return torch.autograd.grad(selected.sum(), x_in, create_graph=True)[0]

  def get_eps_from_model(self, x, t,y):
    alpha_t = self.alphas_cumprod[t]
    if self.classifier is None :
      model_output = self.model(x, torch.tensor(t, device=device).unsqueeze(0))
      model_output = model_output[:,:3,:,:]
    else:
      model_output = self.model(x, t, y)[:, :3]
      model_output = model_output - np.sqrt(1 - alpha_t) * self.cond_fn(x, t,y)

    return(model_output)

  def predict_xstart_from_eps(self, x, eps, t):
    x_start = (
        np.sqrt(1.0 / self.alphas_cumprod[t])* x
        - np.sqrt(1.0 / self.alphas_cumprod[t] - 1) * eps
    )
    x_start = x_start.clamp(-1.,1.)
    return(x_start)


  def initialize(self,imgshape, y, t):
      y_0 = y
      H = self.H
      x_0 = H.H_pinv(y_0).view(*imgshape).detach()
      alpha_t = self.alphas_cumprod[t]
      return np.sqrt(alpha_t) * x_0 + np.sqrt(1 - alpha_t) * torch.randn_like(x_0)

  def initialize_beta_noise(self,imgshape,y,t):
    x = torch.randn(self.imgshape,device=device)*np.sqrt(self.betas[t])
    x.requires_grad = True
    return x


  def sample(self,y,x_true=None, show_steps=True, vis_y=None,):

    # initialize xt for t=T
    T = self.reversed_time_steps[-1]
    if self.start_noise_level_noise is None:
      x = self.initialize(self.imgshape,y,T)
    else:
       x = self.initialize_beta_noise(self.imgshape,y,self.start_noise_level_noise)
    x.requires_grad = True

    for i, (s,t) in enumerate(zip(self.s_list,self.reversed_time_steps)):

      n = x.size(0)
      alpha_s = self.alphas_cumprod[s]
      alpha_t = self.alphas_cumprod[t]
      c1 = np.sqrt(np.clip((1 - alpha_t / alpha_s ) * (1 - alpha_s ) / (1 - alpha_t),a_min=0, a_max=None))
      c2 = np.sqrt((1 - alpha_s) - c1 ** 2)

      # predict x0
      x.requires_grad = True
      eps = self.get_eps_from_model(x,t,y)
      x0_pred= self.predict_xstart_from_eps(x,eps,t)

      mat = (self.H.H_pinv(y) - self.H.H_pinv(self.H.H(x0_pred))).reshape(n, -1)
      mat_x = (mat.detach() * x0_pred.reshape(n, -1)).sum()
      grad_term = torch.autograd.grad(outputs=mat_x,inputs=x, retain_graph=True)[0]

      x0_pred = x0_pred.detach()

      noise = torch.randn_like(x)
      x= np.sqrt(alpha_s) * x0_pred+ c1 * noise + c2 * eps.detach() + grad_term *np.sqrt(alpha_t)


      if show_steps:
        if i%100==99 or t==10 or i==0:
          print('Iteration :', i, 't', t)
          if x_true is not None:
            pilimg = display_as_pilimg(torch.cat((x_true, x0_pred, x), dim=3))
      # this is a quick fix and should be debuged properly
      if i == len(self.s_list)-1:
        return x0_pred
    return(x0_pred)