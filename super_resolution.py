import torch
import torchvision
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from guided_diffusion.unet import create_model
from utils.degredations import  SuperResolution
from pgdm import PGDM_DDIM
from ddpm import DDPM

import os

def tensor2pil(t):
    t = 0.5+0.5*t.to('cpu')
    t = t.squeeze()
    t = t.clamp(0.,1.)
    pil_img = torchvision.transforms.ToPILImage()(t)
    return pil_img

def pilimg_to_tensor(pil_img):
  t = torchvision.transforms.ToTensor()(pil_img)
  t = 2*t-1 # [0,1]->[-1,1]
  t = t.unsqueeze(0)
  t = t.to(device)
  return(t)

if __name__ == '__main__':
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  print("Device:", device)
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
  
  # Load SuperResolution 
  channels = 3
  img_dim = 256

  H_sup_res = SuperResolution(channels,img_dim,2,device)
  # load models 
  
  os.makedirs('results_pgdm_supr', exist_ok=True)
  os.makedirs('results_dps_supr', exist_ok=True)

  
  # Save images
  pgdm = PGDM_DDIM(H_sup_res,model=model, inpainting = False,eta=1,start_noise_level_noise=50,num_diffusion_timesteps=100)
  ddpm = DDPM(model,num_diffusion_timesteps=1000)
  for idx in tqdm(range(750,1000)):
      if idx%50:
          print(idx)    
      x_true_pil = Image.open('ffhq256-1k-validation/'+str(idx).zfill(5)+'.png')
      x_true = pilimg_to_tensor(x_true_pil) # Ajoute une dimension batch
      x_true = x_true.to(device)    
      y = H_sup_res.H(x_true)
      # PGDM
      x_pgdm = pgdm.sample(y, x_true=x_true, show_steps=False, vis_y=None)
      pil_img = tensor2pil(x_pgdm)
      pil_img.save(f'results_pgdm_supr/{str(idx).zfill(5)}.png')
      # DDPM
      x_ddpm= ddpm.posterior_sampling(H_sup_res.H,y, x_true=x_true, show_steps=False,viz_y=False)
      pil_img = tensor2pil(x_ddpm)
      pil_img.save(f'results_dps_supr/{str(idx).zfill(5)}.png')
      