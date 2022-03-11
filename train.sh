srun -c 3 --mem 15G  --gres=gpu:2,gpumem:20G --time=24:00:00 python train.py --dataroot /mnt/gpid08/users/jorge.pueyo/ScanNet/images --dirSem layout --model adgan --name scannet_clean \
--lr 0.001 --lambda_GAN 5 --lambda_A 1 --lambda_B 2 --lambda_cx 2 --n_layers 3 --batchSize 8 \
--pool_size 0 --resize_or_crop no --gpu_ids 0,1 --BP_input_nc 18 --SP_input_nc 18 --no_flip --which_model_netG ADGen --niter 2000 --niter_decay 10 --checkpoints_dir ./checkpoints --L1_type origin --n_layers_D 3 \
--with_D_PP 1 --with_D_PB 1 --display_id 0 --nThreads 5 \
--print_freq 10 --norm batch --save_epoch_freq 40 --use_cxloss 1 --continue_train --epoch_count 1370