
from math import sqrt
import numpy as np
import torch
import torchvision
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from guided_diffusion.unet import create_model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device:", device)

class DDPM:
  def __init__(self, model, num_diffusion_timesteps=1000):
    self.num_diffusion_timesteps = num_diffusion_timesteps
    self.reversed_time_steps = np.arange(self.num_diffusion_timesteps)[::-1]
    beta_start = 0.0001
    beta_end = 0.02
    self.betas = np.linspace(beta_start, beta_end, self.num_diffusion_timesteps,
                              dtype=np.float64)
    self.alphas = 1.0 - self.betas
    # liste avec le produit cumulatif sur la ligne
    self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
    self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
    self.model = model
    self.imgshape = (1,3,256,256)


  def get_eps_from_model(self, x, t):
    # the model outputs:
    # - an estimation of the noise eps (chanels 0 to 2)
    # - learnt variances for the posterior  (chanels 3 to 5)
    # (see Improved Denoising Diffusion Probabilistic Models
    # by Alex Nichol, Prafulla Dhariwal
    # for the parameterization)
    # We discard the second part of the output for this practice session.
    model_output = self.model(x, torch.tensor(t, device=device).unsqueeze(0))
    model_output = model_output[:,:3,:,:]
    return(model_output)

  def predict_xstart_from_eps(self, x, eps, t):
    x_start = (
        np.sqrt(1.0 / self.alphas_cumprod[t])* x
        - np.sqrt(1.0 / self.alphas_cumprod[t] - 1) * eps
    )
    x_start = x_start.clamp(-1.,1.)
    return(x_start)

  def sample(self, show_steps=True):
    with torch.no_grad():  # avoid backprop wrt model parameters
      x = torch.randn(self.imgshape,device=device)  # initialize x_t for t=T
      for i, t in enumerate(self.reversed_time_steps):
        t = int(round(t))
        alpha = self.alphas[t].item()
        alpha_cumprod = self.alphas_cumprod[t].item()
        beta = self.betas[t].item()

        z = torch.randn(self.imgshape,device=device)
        x = 1/sqrt(alpha)*(x-((1-alpha)/sqrt(1-alpha_cumprod))*self.get_eps_from_model(x,t))+ sqrt(beta)*z
        if show_steps:
          if (t+1)%100==0:
            print('Iteration :', t+1)
            eps = self.get_eps_from_model(x,t)
            xhat = self.predict_xstart_from_eps(x,eps,t)
            pilimg = display_as_pilimg(torch.cat((x, xhat), dim=3))

    return(x)


  def posterior_sampling(self, linear_operator, y, x_true=None, show_steps=True, viz_y = None):
    if viz_y is None:
      viz_y=y

    #init xT
    x = torch.randn(self.imgshape, device=device)
    x.requires_grad= True
    for i, t in enumerate(self.reversed_time_steps): #compute eps:
      eps = self.get_eps_from_model(x, t)
      #compute x_start:
      x_start = self.predict_xstart_from_eps(x, eps, t)
      #compute l2 error:
      l2_error = torch.sum((y-linear_operator(x_start))**2)
      zeta = 1/torch.sqrt(l2_error)
      gradx_l2_error = torch.autograd.grad (outputs=l2_error, inputs=x)[0]
      #compute
      posterior_mean = (
          self.betas[t]*np.sqrt(self.alphas_cumprod_prev[t]) / (1.0 - self.alphas_cumprod[t]) * x_start
          + (1.0 - self.alphas_cumprod_prev[t])*np.sqrt(self.alphas[t])/(1.0-self.alphas_cumprod[t]) * x)
      #sample p_theta(t-1|t)
      noise = torch.randn_like(x)
      x = posterior_mean + np.sqrt(self.betas[t])*noise - zeta*gradx_l2_error
      if show_steps:
        if i%100==99 or t==10 or i==0:
          print('Iteration :', i, 't', t)
          if x_true is not None:
            pilimg = display_as_pilimg(torch.cat((x_true, viz_y, x_start, x), dim=3))
    return x

if __name__ == '__main__':
  import torch
  import torchvision
  import numpy as np

  import matplotlib.pyplot as plt

  from PIL import Image
  from tqdm import tqdm
  from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  print("Device:", device)

  from utils.degredations import Deblurring, SuperResolution

  # Load model
  model_config = {'image_size': 256,
                  'num_channels': 128,
                  'num_res_blocks': 1,
                  'channel_mult': '',
                  'learn_sigma': True,
                  'class_cond': False,
                  'use_checkpoint': False,
                  'attention_resolutions': 16,
                  'num_heads': 4,
                  'num_head_channels': 64,
                  'num_heads_upsample': -1,
                  'use_scale_shift_norm': True,
                  'dropout': 0.0,
                  'resblock_updown': True,
                  'use_fp16': False,
                  'use_new_attention_order': False,
                  'model_path': 'ffhq_10m.pt'}
  model = create_model(**model_config)
  model = model.to(device)
  # use in eval mode:
  model.eval();

  ddpm = DDPM(model)
  ddpm.sample()
