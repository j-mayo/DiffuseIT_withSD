import argparse


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument(
        "-p", "--prompt", type=str, help="The prompt for the desired editing", required=False
    )
    parser.add_argument(
        "-s", "--source", type=str, help="The prompt for the source image", required=False
    )
    parser.add_argument(
        "-i", "--init_image", type=str, help="The path to the source image input", required=True
    )
    parser.add_argument(
        "-tg", "--target_image", type=str, help="The path to the target style image", required=False
    )

    parser.add_argument(
        "--skip_timesteps",
        type=int,
        help="How many steps to skip during the diffusion.",
        default=40,
    )

    parser.add_argument(
        "--ddim",
        help="Indicator for using DDIM instead of DDPM",
        action="store_true",
    )

    parser.add_argument(
        "--timestep_respacing",
        type=str,
        help="How to respace the intervals of the diffusion process (number between 1 and 1000).",
        default="100",
    )
    parser.add_argument(
        "--model_output_size",
        type=int,
        help="The resolution of the outputs of the diffusion model",
        default=256,
        choices=[256, 512],
    )
    parser.add_argument(
        "--clip_models",
        help="List for CLIP models",
        nargs="+",
        default=['RN50', 'RN50x4', 'ViT-B/32', 'RN50x16', 'ViT-B/16'],
    )
    # Augmentations
    parser.add_argument("--aug_num", type=int, help="The number of augmentation", default=8)
    parser.add_argument("--diff_iter", type=int, help="The number of augmentation", default=50)

    parser.add_argument(
        "--clip_guidance_lambda",
        type=float,
        help="Controls how much the image should look like the prompt",
        default=2000,
    )
    parser.add_argument(
        "--lambda_trg",
        type=float,
        help="style loss for target style image",
        default=2000,
    )
    parser.add_argument(
        "--l2_trg_lambda",
        type=float,
        help="l2 loss for target style image",
        default=3000,
    )
    parser.add_argument(
        "--range_lambda",
        type=float,
        help="Controls how far out of range RGB values are allowed to be",
        default=200,
    )
    parser.add_argument(
        "--vit_lambda", type=float, help="total vit loss", default=1,
    )
    parser.add_argument(
        "--lambda_ssim", type=float, help="key self similarity loss", default=1000,
    )
    parser.add_argument(
        "--lambda_dir_cls", type=float, help="semantic divergence loss", default=100,
    )
    parser.add_argument(
        "--lambda_contra_ssim", type=float, help="contrastive loss for keys", default=200,
    )
    
    parser.add_argument(
        "--id_lambda", type=float, help="identity loss", default=100,
    )
    parser.add_argument(
        "--resample_num", type=float, help="resampling number", default=10,
    )
    parser.add_argument("--seed", type=int, help="The random seed", default=None)
    parser.add_argument("--gpu_id", type=int, help="The GPU ID", default=0)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="The filename to save, must be png",
        default="output.png",
    )
    parser.add_argument("--iterations_num", type=int, help="The number of iterations", default=10)
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The number number if images to sample each diffusion process",
        default=1,
    )

    parser.add_argument(
        "--use_ffhq",
        action="store_true",
    )
    parser.add_argument(
        "--use_prog_contrast",
        action="store_true",
    )
    parser.add_argument(
        "--use_range_restart",
        action="store_true",
    )
    parser.add_argument(
        "--use_colormatch",
        action="store_true",
    )
    parser.add_argument(
        "--use_noise_aug_all",
        action="store_true",
    )
    parser.add_argument(
        "--regularize_content",
        action="store_true",
    )
    
    parser.add_argument(
        "--stable_diffusion",
        action="store_true"
    )

    parser.add_argument(
        "--pretrained_model",
        default="stabilityai/stable-diffusion-2-1-base",
        #default="runwayml/stable-diffusion-v1-5"
        type=str,
        help="The pretrained stable diffusion model name, or name of repo at huggingface. This must be used for stable diffusion"
    )

    parser.add_argument(
        "--mixed_precision",
        default=None,
        type=str,
        choices=["fp16","bf16"],
        help="When use stable diffusion, you can choose the dtype of weight. Choose fp16 or bf16, or nothing"
    )
    """
    parser.add_argument(
        "--strength",
        default=1,
        type=int,
        help="Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1."
    )
    """
    parser.add_argument(
        "--guidance_scale",
        default=7.5,
        type=float,
        help=""
    )

    parser.add_argument(
        "--set_alpha_to_one",
        default=True,
        type=float
    )

    parser.add_argument(
        "--scaling_factor",
        type=float,
        default=0.18215
    )

    parser.add_argument(
        "--size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--stop_grad",
        type=int,
        default=1000
    )
    args = parser.parse_args()
    return args
