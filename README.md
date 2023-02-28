# DiffuseIT with Stable diffusion
This is **NOT** the official repository of DiffuseIT!  
Original DiffuseIT is rely on gaussian diffusion, and I modify it to use Stable diffusion - only in case of Image-guide reference.

Original Paper Link : https://arxiv.org/abs/2209.15264

### Environment
Pytorch 1.9.0, Python 3.9, Diffusers 0.12.1

```
$ conda create --name DiffuseIT python=3.9
$ conda activate DiffuseIT
$ pip install ftfy regex matplotlib lpips kornia opencv-python torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install color-matcher
$ pip install git+https://github.com/openai/CLIP.git
$ pip install diffusers
```

### Model download

I used the Stablediffusion model from [[huggingface](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)].  
Because of memory issue, I used the stable diffusion 2.1 base, which is 512-res model.  
The usage is written below.  

### Text-guided Image translation

I tried to write the code for only Image-guided Image translation... so this is not provided.
  
  
### Image-guided Image translation
  
I write the code for both ddpm and ddim scheduling, but maybe ddpm has errors. Maybe I did some mistakes... So put the '--ddim' to use ddim schedule.  
  
Although Stable diffusion was applied, the performance was not better than before. I have thought the parameters are must be adjusted for stable diffusion. In original DiffuseIT, they use 200 timestep respacing, and skip 80 timesteps. But, here, that setting works poor. Instead, skip big timesteps - ex: 170 - get better result than fewers. I am not sure, but I hypothesized that there was a problem with the content of the source image being maintained within the stable diffusion, so I increased the timestep to skip to maintain the content.

And there is new argument, '--stop_grad'. If stop_grad = n, then after nth diffusion step, gradient calculating is stopped, i.e. original ddim process is working after nth diffusion step. This gets better result, but as of now, this cannot beat the existing one. 

```
python main.py -i "input_example/reptile1.jpg"  --output_path "./tests_170/output_reptile_sd_ddim_15" -tg "input_example/reptile2.jpg" --stop_grad 15 \
--use_range_restart --diff_iter 100 --timestep_respacing 200 --skip_timesteps 170 --use_noise_aug_all --stable_diffusion --guidance_scale 7.5 --ddim \
--iterations_num 10 --use_colormatch
```

To remove the color matching, deactivate ```--use_colormatch```

For additional information with arguments, see optimization/arguments.py


