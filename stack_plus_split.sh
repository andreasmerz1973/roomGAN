srun -c 2 --mem 10G --time=24:00:00 python3 util/stack_masks.py 
srun -c 2 --mem 10G --time=24:00:00 python3 util/create_masked_split.py
srun -c 3 --mem 15G --gres=gpu:2,gpumem:12G --time=24:00:00 bash train.sh
