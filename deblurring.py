import torch
import torchvision
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from guided_diffusion.unet import create_model
from pgdm import PGDM_DDIM
from ddpm import DDPM

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device:", device)

from utils.degredations import Deblurring

if __name__ == '__main__':


    import os
    os.makedirs('results_pgdm_deblur', exist_ok=True)
    os.makedirs('results_dps_deblur', exist_ok=True)

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

    def pilimg_to_tensor(pil_img):
        t = torchvision.transforms.ToTensor()(pil_img)
        t = 2*t-1 # [0,1]->[-1,1]
        t = t.unsqueeze(0)
        t = t.to(device)
        return(t)

    def tensor2pil(t):
        t = 0.5+0.5*t.to('cpu')
        t = t.squeeze()
        t = t.clamp(0.,1.)
        pil_img = torchvision.transforms.ToPILImage()(t)
        return pil_img
    


    # use in eval mode:
    model.eval();
    k = torch.Tensor([1 / 9] * 9).to(device)
    channels = 3
    img_dim = 256
    H_deblurring = Deblurring(k, channels, img_dim, device)

    pgdm = PGDM_DDIM(H_deblurring,model=model, inpainting = False,eta=0.5,start_noise_level_noise=None,num_diffusion_timesteps=100)
    for idx in tqdm(range(1000)):
        x_true_pil = Image.open('ffhq256-1k-validation/'+str(idx).zfill(5)+'.png')
        x_true = pilimg_to_tensor(x_true_pil) # Ajoute une dimension batch
        x_true = x_true.to(device)    
        y = H_deblurring.H(x_true)
        x = pgdm.sample(y, x_true=x_true, show_steps=False, vis_y=None)
        # write the image to the results folder
        pilimg = tensor2pil(x)
        pilimg.save(f'results_pgdm_deblur/{str(idx).zfill(5)}.png')
        
    
    ddpm = DDPM(model=model,num_diffusion_timesteps=1000)
    for idx in tqdm(range(500,750)):
        if idx%50:
            print(idx)
        x_true_pil = Image.open('ffhq256-1k-validation/'+str(idx).zfill(5)+'.png')
        x_true = pilimg_to_tensor(x_true_pil) # Ajoute une dimension batch
        x_true = x_true.to(device)    
        y = H_deblurring.H(x_true)
        
        x = ddpm.posterior_sampling(H_deblurring.H,y, x_true=x_true, show_steps=False)
        pilimg = tensor2pil(x)
        pilimg.save(f'results_dps_deblur/{str(idx).zfill(5)}.png')
        