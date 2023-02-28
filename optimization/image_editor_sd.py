import os
from pathlib import Path
from optimization.constants import ASSETS_DIR_NAME, RANKED_RESULTS_DIR

from utils_visualize.metrics_accumulator import MetricsAccumulator
from utils_visualize.video import save_video

from numpy import random
from optimization.augmentations import ImageAugmentations

from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF
from torch.nn.functional import mse_loss
from optimization.losses import range_loss, d_clip_loss
# import lpips
import numpy as np
from src.vqc_core import *
from model_vit.loss_vit import Loss_vit
from CLIP import clip
from guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from utils_visualize.visualization import show_tensor_image, show_editied_masked_image
from pathlib import Path
from id_loss import IDLoss

from color_matcher import ColorMatcher
from color_matcher.io_handler import load_img_file, save_img_file, FILE_EXTS
from color_matcher.normalizer import Normalizer

from guided_diffusion.guided_diffusion.respace import space_timesteps
# added for diffusers

import requests
#import torch
#from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import randn_tensor

mean_sig = lambda x:sum(x)/len(x)
class ImageEditor:
    def __init__(self, args) -> None:
        self.args = args
        os.makedirs(self.args.output_path, exist_ok=True)

        self.ranked_results_path = Path(self.args.output_path)
        os.makedirs(self.ranked_results_path, exist_ok=True)

        self.is_sd = self.args.stable_diffusion

        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)
        
        if not self.is_sd: # added!
            self.model_config = model_and_diffusion_defaults()
            if self.args.use_ffhq:
                self.model_config.update(
                {
                    "attention_resolutions": "16",
                    "class_cond": self.args.model_output_size == 512,
                    "diffusion_steps": 1000,
                    "rescale_timesteps": True,
                    "timestep_respacing": self.args.timestep_respacing,
                    "image_size": self.args.model_output_size,
                    "learn_sigma": True,
                    "noise_schedule": "linear",
                    "num_channels": 128,
                    "num_head_channels": 64,
                    "num_res_blocks": 1,
                    "resblock_updown": True,
                    "use_fp16": False,
                    "use_scale_shift_norm": True,
                }
            )
            else:
                self.model_config.update(
                {
                    "attention_resolutions": "32, 16, 8",
                    "class_cond": self.args.model_output_size == 512,
                    "diffusion_steps": 1000,
                    "rescale_timesteps": True,
                    "timestep_respacing": self.args.timestep_respacing,
                    "image_size": self.args.model_output_size,
                    "learn_sigma": True,
                    "noise_schedule": "linear",
                    "num_channels": 256,
                    "num_head_channels": 64,
                    "num_res_blocks": 2,
                    "resblock_updown": True,
                    "use_fp16": True,
                    "use_scale_shift_norm": True,
                }
            )

        #else: # added!
        # TODO: add arguments for SD, and use it to load pretrained SD
            # model = load_model_from_config(config, f"{opt.ckpt}")
            # self.model = model.model # DiffusionWraper, made from unet config. Maybe it acts similar to original unet?
            # pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.args.pretrained_model, torch_dtype=torch.float16)



        # Load models
        self.device = torch.device(
            f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        print("Using device:", self.device)

        
        if not self.is_sd: # added!
        
            self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
            
            if self.args.use_ffhq:
                self.model.load_state_dict(
                    torch.load(
                        "./checkpoints/ffhq_10m.pt",
                        map_location="cpu",
                    )
                )
                self.idloss = IDLoss().to(self.device)
            else:
                self.model.load_state_dict(
                torch.load(
                    "./checkpoints/256x256_diffusion_uncond.pt"
                    if self.args.model_output_size == 256
                    else "checkpoints/512x512_diffusion.pt",
                    map_location="cpu",
                )
            )
            self.model.requires_grad_(False).eval().to(self.device)
            for name, param in self.model.named_parameters():
                if "qkv" in name or "norm" in name or "proj" in name:
                    param.requires_grad_()
            if self.model_config["use_fp16"]:
                self.model.convert_to_fp16()
        
        else: #added!
            # load pipeline (=models)
            if self.args.mixed_precision == "fp16":
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.args.pretrained_model, torch_dtype=torch.float16)
            elif self.args.mixed_precision == "bf16":
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.args.pretrained_model, torch_dtype=torch.bfloat16)
        
            else:
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.args.pretrained_model)

            if self.args.ddim:
                from diffusers import DDIMScheduler
                pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            else:
                from diffusers import DDPMScheduler
                pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(self.device)
            self.pipe = pipe

            # Next: load the "model" and "diffusion", Dreambooth 개발된 걸 참고해보기. 또한 grad도 해결..해야함

            self.vae, self.text_encoder, self.tokenizer, self.unet, self.scheduler, self.feature_extractor = \
                pipe.vae, pipe.text_encoder, pipe.tokenizer, pipe.unet, pipe.scheduler, pipe.feature_extractor
            
            self.vae.requires_grad_(False).eval().to(self.device)
            self.text_encoder.requires_grad_(False).eval().to(self.device)
            # self.tokenizer.requires_grad_(False).eval().to(self.device)
            self.unet.requires_grad_(False).eval().to(self.device)
            for name, param in self.unet.named_parameters():
                if "qkv" in name or "norm" in name or "proj" in name:
                    #print(name)
                    param.requires_grad_()
                    

            self.betas = self.scheduler.betas.to(device=self.device)
            self.alphas_cumprod = self.scheduler.alphas_cumprod.to(device=self.device)
            # timestep respacing
            new_timesteps = space_timesteps(1000, self.args.timestep_respacing)

            last_alpha_cumprod = 1.0
            self.timestep_map = []
            new_betas = []
            #new_alphas_cumprod = []
            for i, alpha_cumprod in enumerate(self.alphas_cumprod):
                if i in new_timesteps:
                    #new_alphas_cumprod.append(alpha_cumprod)
                    new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                    last_alpha_cumprod = alpha_cumprod
                    self.timestep_map.append(i)
            self.num_timesteps = len(new_betas)
            self.timestep = torch.from_numpy(np.array(self.timestep_map[::-1])).to(self.device)
            self.scheduler.timesteps = self.timestep # scheduler timestep 수정
            # 수정된 부분,
            # self.betas = new_betas
            # alphas = 1 - self.betas
            # self.alphas_cumprod = np.cumprod(alphas, axis=0)
            self.alphas_cumprod_prev = torch.from_numpy(np.append(1.0, np.array(self.alphas_cumprod[:-1].cpu()))).cuda().to(self.device)
            # print(self.alphas_cumprod_prev.is_cuda)
            # self.alphas_cumprod = self.alphas_cumprod.flatten()
            self.sqrt_alphas_cumprod = (self.alphas_cumprod) ** 0.5
            self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod) ** 0.5
            self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
            self.sqrt_recip_alphas_cumprod = (1.0 / self.alphas_cumprod) ** 0.5
            self.sqrt_recipm1_alphas_cumprod = (1.0 / self.alphas_cumprod - 1) ** 0.5

            self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            )
            # log calculation clipped because the posterior variance is 0 at the
            # beginning of the diffusion chain.
            self.posterior_log_variance_clipped = torch.log(
                torch.from_numpy(np.append(np.array(self.posterior_variance[1].cpu()), np.array(self.posterior_variance[1:].cpu()))).cuda().to(self.device)
            )
            self.posterior_mean_coef1 = (
                self.betas * ((self.alphas_cumprod_prev) ** 0.5) / (1.0 - self.alphas_cumprod)
            )
            self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * ((self.scheduler.alphas).cuda().to(self.device) * 0.5) / (1.0 - self.alphas_cumprod)
            )
            # self.sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alpha_prod.flatten()
            self.final_alpha_cumprod = 1 if self.args.set_alpha_to_one else self.alpha_cumprod[0]
            # self.num_timesteps = len(self.scheduler.betas)

        # loading vit
        with open("model_vit/config.yaml", "r") as ff:
            config = yaml.safe_load(ff)

        cfg = config
        
        self.VIT_LOSS = Loss_vit(cfg, lambda_ssim=self.args.lambda_ssim,lambda_dir_cls=self.args.lambda_dir_cls,lambda_contra_ssim=self.args.lambda_contra_ssim,lambda_trg=args.lambda_trg).eval()#.requires_grad_(False)
      
        names = self.args.clip_models
        # init networks
        if self.args.target_image is None:
            self.clip_net = CLIPS(names=names, device=self.device, erasing=False)#.requires_grad_(False)
        
        self.cm = ColorMatcher()
        self.clip_size = 224
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
#         self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)

        self.image_augmentations = ImageAugmentations(self.clip_size, self.args.aug_num)
        self.metrics_accumulator = MetricsAccumulator()

    def noisy_aug(self,t,x,x_hat):
        if not self.is_sd:
            fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t]
        else:
            fac = self.sqrt_one_minus_alphas_cumprod[t]
        x_mix = x_hat * fac + x * (1 - fac)
        return x_mix

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def unscale_timestep(self, t):
        if not self.is_sd:
            unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()
        else:
            unscaled_timestep = (t * (self.num_timesteps / 1000)).long()
        return unscaled_timestep


    def edit_image_by_prompt(self):

        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        self.init_image_pil = Image.open(self.args.init_image).convert("RGB")
        self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
        self.init_image = (
            TF.to_tensor(self.init_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )
        
        self.target_image = None
        if self.args.target_image is not None:
            self.target_image_pil = Image.open(self.args.target_image).convert("RGB")
            self.target_image_pil = self.target_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
            self.target_image = (
                TF.to_tensor(self.target_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
            )
        
        
        self.prev = self.init_image.detach()
        if self.target_image is None:
            txt2 = self.args.prompt
            txt1 = self.args.source
            with torch.no_grad():
                self.E_I0 = E_I0 = self.clip_net.encode_image(0.5*self.init_image+0.5, ncuts=0)
                self.E_S, self.E_T = E_S, E_T =  self.clip_net.encode_text([txt1, txt2])
                self.tgt = (1 * E_T  - 0.4 * E_S + 0.2* E_I0).normalize()

            pred = self.clip_net.encode_image(0.5*self.prev+0.5, ncuts=0)
            clip_loss  = - (pred @ self.tgt.T).flatten().reduce(mean_sig)

            self.loss_prev = clip_loss.detach().clone()
        self.flag_resample=False
        total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1
        def cond_fn(x, t, y=None):
            if self.args.prompt == "":
                return torch.zeros_like(x)
            self.flag_resample=False
            with torch.enable_grad():
                frac_cont=1.0
                if self.target_image is None:
                    if self.args.use_prog_contrast:
                        if self.loss_prev > -0.5:
                            frac_cont = 0.5
                        elif self.loss_prev > -0.4:
                            frac_cont = 0.25
                    if self.args.regularize_content:
                        if self.loss_prev < -0.5:
                            frac_cont = 2
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t) # 이 t는 이미 scale된 것임을 유념.

                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )
                
                loss = torch.tensor(0)
                if self.target_image is None:
                    if self.args.clip_guidance_lambda != 0:
                        x_clip = self.noisy_aug(t[0].item(),x,out["pred_xstart"])
                        pred = self.clip_net.encode_image(0.5*x_clip+0.5, ncuts=self.args.aug_num)
                        clip_loss  = - (pred @ self.tgt.T).flatten().reduce(mean_sig)
                        loss = loss + clip_loss*self.args.clip_guidance_lambda
                        self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())
                        self.loss_prev = clip_loss.detach().clone()
                if self.args.use_noise_aug_all:
                    x_in = self.noisy_aug(t[0].item(),x,out["pred_xstart"])
                else:
                    x_in = out["pred_xstart"]
                
                if self.args.vit_lambda != 0:
                    
                        
                    if t[0]>self.args.diff_iter:
                        vit_loss,vit_loss_val = self.VIT_LOSS(x_in,self.init_image,self.prev,use_dir=True,frac_cont=frac_cont,target = self.target_image)
                    else:
                        vit_loss,vit_loss_val = self.VIT_LOSS(x_in,self.init_image,self.prev,use_dir=False,frac_cont=frac_cont,target = self.target_image)
                    loss = loss + vit_loss
                    
                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())
                if self.target_image is not None:
                    loss = loss + mse_loss( x_in, self.target_image) * self.args.l2_trg_lambda
                
                
                if self.args.use_ffhq:
                    loss =  loss + self.idloss(x_in,self.init_image) * self.args.id_lambda
                self.prev = x_in.detach().clone()
                
                if self.args.use_range_restart:
                    if t[0].item() < total_steps:
                        if self.args.use_ffhq:
                            if r_loss>0.1:
                                self.flag_resample =True
                        else:
                            if r_loss>0.01:
                                self.flag_resample =True
                
            return -torch.autograd.grad(loss, x)[0], self.flag_resample

        save_image_interval = self.diffusion.num_timesteps // 5
        for iteration_number in range(self.args.iterations_num):
            print(f"Start iterations {iteration_number}")
    
            sample_func = (
                self.diffusion.ddim_sample_loop_progressive
                if self.args.ddim
                else self.diffusion.p_sample_loop_progressive
            )
            samples = sample_func(
                self.model,
                (
                    self.args.batch_size,
                    3,
                    self.model_config["image_size"],
                    self.model_config["image_size"],
                ),
                clip_denoised=False,
                model_kwargs={}
                if self.args.model_output_size == 256
                else {
                    "y": torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)
                },
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=self.args.skip_timesteps,
                init_image=self.init_image,
                postprocess_fn=None,
                randomize_class=True,
            )
            if self.flag_resample:
                continue
            
            intermediate_samples = [[] for i in range(self.args.batch_size)]
            total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1 
            total_steps_with_resample= self.diffusion.num_timesteps - self.args.skip_timesteps - 1 + (self.args.resample_num-1)
            for j, sample in enumerate(samples):
                should_save_image = j % save_image_interval == 0 or j == total_steps_with_resample

                # self.metrics_accumulator.print_average_metric()

                for b in range(self.args.batch_size):
                    pred_image = sample["pred_xstart"][b]
                    visualization_path = Path(
                        os.path.join(self.args.output_path, self.args.output_file)
                    )
                    visualization_path = visualization_path.with_name(
                        f"{visualization_path.stem}_i_{iteration_number}_b_{b}{visualization_path.suffix}"
                    )

                    pred_image = pred_image.add(1).div(2).clamp(0, 1)
                    pred_image_pil = TF.to_pil_image(pred_image)
            ranked_pred_path = self.ranked_results_path / (visualization_path.name)
            
            if self.args.target_image is not None:
                if self.args.use_colormatch:
                    src_image = Normalizer(np.asarray(pred_image_pil)).type_norm()
                    trg_image = Normalizer(np.asarray(self.target_image_pil)).type_norm()
                    img_res = self.cm.transfer(src=src_image, ref=trg_image, method='mkl')
                    img_res = Normalizer(img_res).uint8_norm()
                    save_img_file(img_res, str(ranked_pred_path))
            else:
                pred_image_pil.save(ranked_pred_path)

    def ddpm_sampling(
        self, 
        timesteps,
        num_inference_steps,
        do_classifier_free_guidance,
        prompt_embeds,
        extra_step_kwargs,
        cond_fn,
        eta,
        latents
        ):
        # like ddim_sample_loop_progressive? denoising sampling
        # num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # print(timesteps)
        result = []
        for i, t in enumerate(timesteps):
            # prev_latents = latents.copy()
            
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            #print(latent_model_input.shape)
            # predict the noise residual
            with torch.no_grad():
                model_output = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                ).sample
                # 기존에는 q_sample로 샘플링을 하는데, 여기서는 그냥 unet에 넣어버림. 근데 이게 맞나?
                # 기존(x_start)의 noise version을 내뱉는 거에 있어서는 같음.
                # TODO? : 이거...확실히 하기
                # 아마 step 함수 내부에서 처리가 되는 듯 함. pred_sample_direction이 저 역할을 하는 거 같고, 그냥 이 형태 유지해도 될듯

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = model_output.chunk(2)
                    model_output = noise_pred_uncond + self.args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # step
                # predict x_t-1, previous noisy sample

                if model_output.shape[1] == latents.shape[1] * 2 and self.scheduler.variance_type in ["learned", "learned_range"]:
                    model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
                else:
                    predicted_variance = None

                alpha_prod_t = self.alphas_cumprod[t]
                alpha_prod_t_prev = self.alphas_cumprod[timesteps[i+1]] if i+1 != len(timesteps) else self.final_alpha_cumprod

                beta_prod_t = 1 - alpha_prod_t
                beta_prod_t_prev = 1 - alpha_prod_t_prev
                current_alpha_t = alpha_prod_t / alpha_prod_t_prev
                current_beta_t = 1 - current_alpha_t


                pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)   

                pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
                current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

                pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents
                # mean <=> pred_prev_sample

                variance = 0
                if t > 0:
                    device = model_output.device
                    variance_noise = randn_tensor(
                        model_output.shape, generator=None, device=self.device, dtype=model_output.dtype
                    )
                    if self.scheduler.variance_type == "fixed_small_log" or "learned_range":
                        variance = self.scheduler._get_variance(t, predicted_variance=predicted_variance) * variance_noise
                    else:
                        variance = (self.scheduler._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

                # 여기까지 p_mean_variance 부분.
                # t_scaled = torch.tensor([len(timesteps)-i-1] * latents.shape[0], device=self.device)
                t_tensor = torch.tensor([t] * latents.shape[0], device=self.device)
                # condition!
                out, flag = self.condition_mean(cond_fn, pred_prev_sample, variance, latents, i, t_tensor, model_kwargs={})
                # if self.scheduler.variance_type == "learned_range":
                if t > 0:
                    variance = torch.exp(0.5 * variance) * variance_noise
                #print(pred_prev_sample.shape, pred_original_sample.shape)
                pred_prev_sample = out + variance
                
                """
                # currently, eta = 0. and generator, variance_noise also None.
                if eta > 0:
                    device = model_output.device
                    if variance_noise is not None and generator is not None:
                        raise ValueError(
                            "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                            " `variance_noise` stays `None`."
                        )

                    if variance_noise is None:
                        variance_noise = randn_tensor(
                            model_output.shape, generator=generator, device=device, dtype=model_output.dtype
                        )
                    variance = std_dev_t * variance_noise

                    prev_sample = prev_sample + variance
                """
                # print(i)
                latents = pred_prev_sample
                flag = flag
                # yield {"sample": pred_prev_sample, "pred_xstart": pred_original_sample, "flag":flag}
                result.append({"sample": pred_prev_sample.clone().detach(), "pred_xstart": pred_original_sample.clone().detach(), "flag":flag})
        return result
    """
    def ddim_sampling(
        self, 
        timesteps,
        num_inference_steps,
        do_classifier_free_guidance,
        prompt_embeds,
        extra_step_kwargs,
        cond_fn,
        eta,
        latents
        ):
        # like ddim_sample_loop_progressive? denoising sampling
        # num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        for i, t in enumerate(timesteps):
            
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else self.latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
            ).sample
            # <=> model_output at gaussian diffusion

            if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.args.guidance_scale * (noise_pred_text - noise_pred_uncond)
            # predict x_t-1, previous noisy sample
            prev_latents = latents.detach()
            # start ddim sampling
             with torch.no_grad(): # 이거 잘 모르겠음 써야하나
                # output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs) # 이 함수가 사실상 ddim 샘플링하는 그거 전체인데, 여기엔 cond_fn이 없어서 풀어서 작성해야 할듯
                # prev_sample = output.prev_sample
                # pred_xstart = output.pred_original_sample


                # image = self.decode_latents(latents)
                # condition score를 latent 상에서 계산해야...하나...? 
                # latent space에서 diffusion process를 진행하니 그게 맞을 듯 하다
                t = torch.tensor([len(timesteps)-i-1] * latents.shape[0], device=self.device) # 여기서의 t는 condition score를 위한 변형. / i로 해야 하나
                if cond_fn is not None:
                    # out, flag = self.condition_score(cond_fn, prev_sample, pred_xstart, prev_latents, t, model_kwargs={})
                    out, flag = self.condition_score(cond_fn, noise_pred, None, prev_latents, t, model_kwargs={})
                
                # Usually our model outputs epsilon, but we re-derive it
                # in case we used x_start or x_prev prediction.
                eps = self._predict_eps_from_xstart(latents, t, out["pred_original_sample"])

                alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, prev_latents.shape)
                alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, prev_latents.shape)
                # output of get_variance <=> mean_pred
                sigma = (
                    eta
                    * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                    * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
                )
                # Equation 12.
                noise = torch.randn_like(prev_latents)
                # pred_sample <=> mean_pred
                mean_pred = (
                    out["pred_original_sample"] * torch.sqrt(alpha_bar_prev)
                    + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
                )
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(prev_latents.shape) - 1)))
                )  # no noise when t == 0
                sample = mean_pred + nonzero_mask * sigma * noise # case of eta > 0에 대응
            # ~ ddim_sampling
                yield {"sample": sample, "pred_xstart": out["pred_original_sample"], "flag":flag}
                latents = sample
                flag = flag
        # return {"sample": sample, "pred_xstart": out["pred_original_sample"], "flag":flag}
    """
        #gasdgafdgadgv
    def ddim_sampling(
        self, 
        timesteps,
        num_inference_steps,
        do_classifier_free_guidance,
        prompt_embeds,
        extra_step_kwargs,
        cond_fn,
        eta,
        latents
        ):
        # like ddim_sample_loop_progressive? denoising sampling
        # num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        result = []
        for i, t in enumerate(timesteps):
            # prev_latents = latents.copy()
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # print(i)
            # predict the noise residual
            with torch.no_grad():
                model_output = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                ).sample
                # 기존에는 q_sample로 샘플링을 하는데, 여기서는 그냥 unet에 넣어버림. 근데 이게 맞나?
                # 기존(x_start)의 noise version을 내뱉는 거에 있어서는 같음.
                # TODO? : 이거...확실히 하기
                # 아마 step 함수 내부에서 처리가 되는 듯 함. pred_sample_direction이 저 역할을 하는 거 같고, 그냥 이 형태 유지해도 될듯

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = model_output.chunk(2)
                    model_output = noise_pred_uncond + self.args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                # predict x_t-1, previous noisy sample
                
                alpha_prod_t = self.alphas_cumprod[t]
                alpha_prod_t_prev = self.alphas_cumprod[timesteps[i+1]] if i+1 != len(timesteps) else self.final_alpha_cumprod

                beta_prod_t = 1 - alpha_prod_t

                pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                # 여기까지 p_mean_variance 부분.

                variance = self.scheduler._get_variance(t, timesteps[i+1]) if i+1 != len(timesteps) else self.scheduler._get_variance(t, -1)
                std_dev_t = eta * variance ** (0.5)

                # t_scaled = torch.tensor([len(timesteps)-i-1] * latents.shape[0], device=self.device)
                t_tensor = torch.tensor([t] * latents.shape[0], device=self.device)

                # condition!
                if i < self.args.stop_grad:
                    out, flag = self.condition_score(cond_fn, pred_original_sample, None, latents, i, t_tensor, model_kwargs={})
                    pred_original_sample = out["pred_original_sample"]

                else:
                    flag = False
                # 일단은, 여기선 mean이 안 쓰여서 None으로 뺴 놓음.

                # clip denoising is false in original.

                pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output
                prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
                """
                # currently, eta = 0. and generator, variance_noise also None.
                if eta > 0:
                    device = model_output.device
                    if variance_noise is not None and generator is not None:
                        raise ValueError(
                            "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                            " `variance_noise` stays `None`."
                        )

                    if variance_noise is None:
                        variance_noise = randn_tensor(
                            model_output.shape, generator=generator, device=device, dtype=model_output.dtype
                        )
                    variance = std_dev_t * variance_noise

                    prev_sample = prev_sample + variance
                """

                latents = prev_sample
                flag = flag
                result.append({"sample": prev_sample.clone().detach(), "pred_xstart": pred_original_sample.clone().detach(), "flag":flag})
        return result
        # return {"sample": sample, "pred_xstart": out["pred_original_sample"], "flag":flag}

    def edit_image_by_prompt_stablediffusion(self):
            
            # height = height or self.unet.config.sample_size * self.vae_scale_factor
            # width = width or self.unet.config.sample_size * self.vae_scale_factor
            do_classifier_free_guidance = self.args.guidance_scale > 1.0

            device = self.pipe._execution_device # i can't conclude this device setting is right... I'm just a potato...
            print(device)
            # prepare image
            # self.image_size = (self.unet.config.sample_size, self.unet.config.sample_size)
            self.image_size = (self.args.size, self.args.size)
            self.init_image_pil = Image.open(self.args.init_image).convert("RGB")
            self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
            self.init_image = (
                TF.to_tensor(self.init_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
            ).to(self.device)

            prompt_embeds = self.pipe._encode_prompt(
                "", # just None?
                self.device,
                1,
                do_classifier_free_guidance,
                None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
            )
            # prepare latent
            '''
            self.init_latent = self.vae.encode(self.init_image).latent_dist.sample(None)
            self.init_latent = self.vae.config.scaling_factor * self.init_latent
            self.init_latent = torch.cat([self.init_latent], dim=0)
            shape = self.init_latents.shape
            noise = 
            '''
            
            #self.scheduler.set_timesteps(50, device=self.device)
            strength = (self.num_timesteps - self.args.skip_timesteps) / self.num_timesteps # 이걸로 skip 조절 가능할듯
            timesteps, num_inference_steps = self.pipe.get_timesteps(self.num_timesteps, strength, self.device) # (num_inference_steps, strength, device)
            latent_timestep = timesteps[:1].repeat(self.args.batch_size * 1) # batch_size * num_image_per_prompt

            # num_channels_latents = self.unet.in_channels
            self.latents = self.pipe.prepare_latents(
                self.init_image,
                latent_timestep,
                self.args.batch_size,
                1,
                # height,
                # width,
                prompt_embeds.dtype,
                device,
                None,
            ) # latents already has shape (N, C, H, W). So it does not need unsqueeze...?
            # self.latent = self.latents[0]

            extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(None, 0.0) # generator, eta
            # prepare target image
            self.target_image = None
            if self.args.target_image is not None:
                self.target_image_pil = Image.open(self.args.target_image).convert("RGB")
                self.target_image_pil = self.target_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
                self.target_image = (
                    TF.to_tensor(self.target_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
                ).to(self.device)
            
            
            self.prev = self.init_image.detach()
            if self.target_image is None: # Text-guided: Not yet. If necessary, then I'll edit to use SD.
                txt2 = self.args.prompt
                txt1 = self.args.source
                with torch.no_grad():
                    self.E_I0 = E_I0 = self.clip_net.encode_image(0.5*self.init_image+0.5, ncuts=0)
                    self.E_S, self.E_T = E_S, E_T =  self.clip_net.encode_text([txt1, txt2])
                    self.tgt = (1 * E_T  - 0.4 * E_S + 0.2* E_I0).normalize()

                pred = self.clip_net.encode_image(0.5*self.prev+0.5, ncuts=0)
                clip_loss  = - (pred @ self.tgt.T).flatten().reduce(mean_sig)

                self.loss_prev = clip_loss.detach().clone()
            self.flag_resample=False
            total_steps = self.num_timesteps - self.args.skip_timesteps - 1

            ###
            def cond_fn(x, i, t, y=None):
                if self.args.prompt == "":
                    return torch.zeros_like(x)
                self.flag_resample=False
                with torch.enable_grad():
                    frac_cont=1.0
                    if self.target_image is None:
                        if self.args.use_prog_contrast:
                            if self.loss_prev > -0.5:
                                frac_cont = 0.5
                            elif self.loss_prev > -0.4:
                                frac_cont = 0.25
                        if self.args.regularize_content:
                            if self.loss_prev < -0.5:
                                frac_cont = 2
                    x = x.detach().requires_grad_()
                    

                    #out = self.diffusion.p_mean_variance(
                        # self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                    #) 
                    #TODO - adjust to stable diffusion
                    # Maybe... that p_mean_variance <=> diffusers's "step" function...
                    # because they all predict x_start, previous timestep.
                    # but "step" function not returns variance. This may be get "_get_variance"??

                    # p_mean_variance 부분
                    latent_model_input = torch.cat([x] * 2) if do_classifier_free_guidance else x
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    # predict the noise residual
                    model_output = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=None,
                    ).sample


                    if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = model_output.chunk(2)
                            model_output = noise_pred_uncond + self.args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    # predict x_t-1, previous noisy sample
                    
                    alpha_prod_t = self.alphas_cumprod[t]
                    alpha_prod_t_prev = self.alphas_cumprod[timesteps[i+1]] if i+1 != len(timesteps) else self.final_alpha_cumprod

                    beta_prod_t = 1 - alpha_prod_t

                    pred_original_sample = (x - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)


                    out = self.decode_latents(pred_original_sample)
                    newx = self.decode_latents(x)
                    #print(out.shape, x.shape, self.init_image.shape, self.target_image.shape)
                    t = self.unscale_timestep(t) # t는 scale 상태 (0 ~ 1000 사이로 맞춰짐)임 이걸 unscale한다.

                    loss = torch.tensor(0)
                    if self.target_image is None: # Text-guided: Not yet. If necessary, I'll edit this.
                        if self.args.clip_guidance_lambda != 0:
                            x_clip = self.noisy_aug(t[0].item(),newx,out)
                            pred = self.clip_net.encode_image(0.5*newx_clip+0.5, ncuts=self.args.aug_num)
                            clip_loss  = - (pred @ self.tgt.T).flatten().reduce(mean_sig)
                            loss = loss + clip_loss*self.args.clip_guidance_lambda
                            self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())
                            self.loss_prev = clip_loss.detach().clone()
                    if self.args.use_noise_aug_all:
                        x_in = self.noisy_aug(t[0].item(),newx,out)
                    else:
                        x_in = out
                    
                    if self.args.vit_lambda != 0:
                        
                            
                        if t[0]>self.args.diff_iter:
                            # print(x_in.device, self.init_image.device,self.prev.device,self.target_image.device)
                            vit_loss,vit_loss_val = self.VIT_LOSS(x_in,self.init_image,self.prev,use_dir=True,frac_cont=frac_cont,target = self.target_image)
                        else:
                            vit_loss,vit_loss_val = self.VIT_LOSS(x_in,self.init_image,self.prev,use_dir=False,frac_cont=frac_cont,target = self.target_image)
                        loss = loss + vit_loss
                        
                    if self.args.range_lambda != 0:
                        r_loss = range_loss(out).sum() * self.args.range_lambda
                        loss = loss + r_loss
                        self.metrics_accumulator.update_metric("range_loss", r_loss.item())
                    if self.target_image is not None:
                        loss = loss + mse_loss( x_in, self.target_image) * self.args.l2_trg_lambda
                    
                    
                    if self.args.use_ffhq:
                        loss =  loss + self.idloss(x_in,self.init_image) * self.args.id_lambda
                    self.prev = x_in.detach().clone()
                    
                    
                    if self.args.use_range_restart:
                        if t[0].item() < total_steps:
                            if self.args.use_ffhq:
                                if r_loss>0.1:
                                    self.flag_resample =True
                            else:
                                if r_loss>0.01:
                                    self.flag_resample =True
                    
                return -torch.autograd.grad(loss, x)[0], self.flag_resample
            ###

            save_image_interval = self.num_timesteps // 5
            for iteration_number in range(self.args.iterations_num):
                print(f"Start iterations {iteration_number}")
                
                if self.args.ddim:
                    samples = self.ddim_sampling(
                        timesteps, num_inference_steps, do_classifier_free_guidance,
                        prompt_embeds, extra_step_kwargs, cond_fn, 0, self.latents)
                else:
                    samples = self.ddpm_sampling(
                        timesteps, num_inference_steps, do_classifier_free_guidance,
                        prompt_embeds, extra_step_kwargs, cond_fn, 0, self.latents)

#                sample_func = (
#                    self.diffusion.ddim_sample_loop_progressive
#                    if self.args.ddim
#
#                    else self.diffusion.p_sample_loop_progressive
#               )
#               samples = sample_func(
#                   self.model,
#                    (
#                        self.args.batch_size,
#                        3,
#                        self.model_config["image_size"],
#                        self.model_config["image_size"],
#                    ),
#                    clip_denoised=False,
#                    model_kwargs={}
#                    if self.args.model_output_size == 256
#                    else {
#                        "y": torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)
#                    },
#                    cond_fn=cond_fn,
#                    progress=True,
#                    skip_timesteps=self.args.skip_timesteps,
#                    init_image=self.init_image,
#                    postprocess_fn=None,
#                    randomize_class=True,
#                )
                if self.flag_resample:
                    print(self.flag_resample)
                    # continue

                intermediate_samples = [[] for i in range(self.args.batch_size)]
                total_steps = self.num_timesteps - self.args.skip_timesteps - 1 
                total_steps_with_resample= self.num_timesteps - self.args.skip_timesteps - 1 + (self.args.resample_num-1)

                for j, sample in enumerate(samples):
                    should_save_image = j % save_image_interval == 0 or j == total_steps_with_resample
                    # print(j)
                    # self.metrics_accumulator.print_average_metric()

                    for b in range(self.args.batch_size):
                        pred_image = sample["pred_xstart"] # "sample"이 prev_sample에 대응하는데, diffusers에선 그걸 샘플링한다. 여긴 이걸 하는데 뭐지

                        pred_image = torch.from_numpy(self.pipe.decode_latents(pred_image))
                        #print(pred_image.shape)
                        visualization_path = Path(
                            os.path.join(self.args.output_path, self.args.output_file)
                        )
                        visualization_path = visualization_path.with_name(
                            f"{visualization_path.stem}_i_{iteration_number}_b_{b}{visualization_path.suffix}"
                        )

                        pred_image = pred_image.add(1).div(2).clamp(0, 1)
                        pred_image = pred_image.permute(0, 3, 1, 2)
                        pred_image_pil = TF.to_pil_image(pred_image[0])
                    
                    #prev_sample = sample["sample"]
                ranked_pred_path = self.ranked_results_path / (visualization_path.name)
                print(str(ranked_pred_path))
                if self.args.target_image is not None and self.args.use_colormatch:
                    src_image = Normalizer(np.asarray(pred_image_pil)).type_norm()
                    trg_image = Normalizer(np.asarray(self.target_image_pil)).type_norm()
                    img_res = self.cm.transfer(src=src_image, ref=trg_image, method='mkl')
                    img_res = Normalizer(img_res).uint8_norm()
                    save_img_file(img_res, str(ranked_pred_path))

                else:
                    pred_image_pil.save(ranked_pred_path)

                """
                pred_image_2 = prev_sample.add(1).div(2).clamp(0, 1)
                pred_image_2_pil = TF.to_pil_image(pred_image_2)
                if self.args.target_image is not None:
                    if self.args.use_colormatch:
                        src_image = Normalizer(np.asarray(pred_image_2_pil)).type_norm()
                        trg_image = Normalizer(np.asarray(self.target_image_pil)).type_norm()
                        img_res = self.cm.transfer(src=src_image, ref=trg_image, method='mkl')
                        img_res = Normalizer(img_res).uint8_norm()
                        save_img_file(img_res, str(ranked_pred_path)+"sampl")
                else:
                    pred_image_pil.save(ranked_pred_path)
                """

    # From gaussian diffusion

    def condition_score(self, cond_fn, pred_xstart, mean, x, i, t, model_kwargs=None):
            """
            Compute what the p_mean_variance output would have been, should the
            model's score function be conditioned by cond_fn.
            See condition_mean() for details on cond_fn.
            Unlike condition_mean(), this instead uses the conditioning strategy
            from Song et al (2020).
            """
            # Now, this fcn's output is what the step()'s output wolud have been...
            # t = t.long()
            alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            # print(alpha_bar)
            # print(t)
            eps = self._predict_eps_from_xstart(x, t, pred_xstart)
            # print(self._scale_timesteps(t).dtype)
            gradient, flag = cond_fn(x, i, t, **model_kwargs)
            # gradient = self.vae.encode(gradient).latent_dist.sample(None) # hmm...
            if i < self.args.stop_grad*100:
                eps = eps - (1 - alpha_bar).sqrt() * gradient

            #out = p_mean_var.copy()
            out = {}
            out["pred_original_sample"] = self._predict_xstart_from_eps(x, t, eps)
            # out["pred_sample"], _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t) # how to...?
            return out,flag

    def condition_mean(self, cond_fn, mean, variance, x, i, t, model_kwargs=None):
            """
            Compute the mean for the previous step, given a function cond_fn that
            computes the gradient of a conditional log probability with respect to
            x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
            condition on y.
            This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
            """
            gradient,flag = cond_fn(x, i, t, **model_kwargs)
            # gradient = self.vae.encode(gradient).latent_dist.sample(None)  # hmm...
            new_mean = mean.float() + variance * gradient.float()
            return new_mean,flag

    def _predict_xstart_from_eps(self, x_t, t, eps):
            assert x_t.shape == eps.shape
            return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
            )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
            return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
            ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def decode_latents(self, latents):
        # print(self.vae.config.keys())
        latents = 1 / self.args.scaling_factor * latents
        image = self.vae.decode(latents).sample
        # image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        # image = image.permute(0, 2, 3, 1).float()
        return image

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    #res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)