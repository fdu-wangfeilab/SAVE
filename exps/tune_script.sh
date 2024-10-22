kl_scale=(100 150 200 250)
capacity=(25 50 100 200)
for kl in ${kl_scale[@]}; do
    for cap in ${capacity[@]}; do
        /mnt/sdc/lijiahao/miniconda3/envs/SAVE/bin/python ./exps/SAVE_mi_tune.py --kl_scale $kl --capacity $cap
    done
done