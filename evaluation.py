import numpy as np
from pytorch_fid import fid_score
import torch
from skimage.metrics import structural_similarity as compare_ssim
import lpips
import numpy as np
from PIL import Image
import torchvision


print("super resolution")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


fid_value = fid_score.calculate_fid_given_paths(['ffhq256-1k-validation/', 'results_pgdm_supr/'], 50, device, 2048)
print('FID for PGDM:', fid_value)

fid_value = fid_score.calculate_fid_given_paths(['ffhq256-1k-validation/', 'results_dps_supr/'], 50, device, 2048)
print('FID for DPS:', fid_value)

print("----------------------------------")

def pilimg_to_tensor(pil_img):
  t = torchvision.transforms.ToTensor()(pil_img)
  t = 2*t-1 # [0,1]->[-1,1]
  t = t.unsqueeze(0)
  t = t.to(device)
  return(t)

def display_as_pilimg(t):
  t = 0.5+0.5*t.to('cpu')
  t = t.squeeze()
  t = t.clamp(0.,1.)
  pil_img = torchvision.transforms.ToPILImage()(t)
  return(pil_img)


def psnr(uref,ut,M=1):
    rmse = np.sqrt(np.mean((np.array(uref.cpu())-np.array(ut.cpu()))**2))
    return 20*np.log10(M/rmse)


def ssim(uref,ut):
    uref = uref.cpu().numpy()
    ut = ut.cpu().numpy()
    return compare_ssim(uref,ut,multichannel=True,channel_axis=0,data_range= 2) # as images are normalized to [-1,1]

def compute_metrics(path1, path2):
    lpips_loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    psnr_values = []
    ssim_values = []
    lpips_values = []
    for i in range(1000):
        x1 = Image.open(f'{path1}/{str(i).zfill(5)}.png')
        x2 = Image.open(f'{path2}/{str(i).zfill(5)}.png')
        x1 = pilimg_to_tensor(x1).unsqueeze(0).to(device) # normalized to [-1,1]
        x2 = pilimg_to_tensor(x2).unsqueeze(0).to(device) # normalized to [-1,1]
        x1 = x1.squeeze()
        x2 = x2.squeeze()
        psnr_values.append(psnr(x1,x2))
        ssim_values.append(ssim(x1,x2))
        lpips_values.append(lpips_loss_fn_vgg(x1,x2).item())
    return psnr_values, ssim_values,lpips_values



# Compute metrics for PGDM

psnr_values, ssim_values,lpips_values = compute_metrics('ffhq256-1k-validation', 'results_pgdm_supr')
print("-----------------------------")
print("PGDM")
print("-----------------------------")
print('PSNR:', np.mean(psnr_values))
print('SSIM:', np.mean(ssim_values))
print('LPIPS:', np.mean(lpips_values))

# Compute metrics for DPS

psnr_values, ssim_values,lpips_values = compute_metrics('ffhq256-1k-validation', 'results_dps_supr')
print("-----------------------------")
print("DPS")
print("-----------------------------")
print('PSNR:', np.mean(psnr_values))
print('SSIM:', np.mean(ssim_values))
print('LPIPS:', np.mean(lpips_values))