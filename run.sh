# python main.py -p "Black Leopard" -s "Lion" -i "input_example/lion1.jpg" --output_path "./outputs/output_leopard" --use_range_restart --use_noise_aug_all --regularize_content



python main.py -i "input_example/reptile1.jpg"  --output_path "./tests_170/output_reptile_sd_ddim_25" -tg "input_example/reptile2.jpg" --stop_grad 25 --use_range_restart --diff_iter 100 --timestep_respacing 200 --skip_timesteps 170 --use_noise_aug_all --stable_diffusion --guidance_scale 7.5 --ddim --iterations_num 10 --use_colormatch
