import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = "AIRNET"
seq_len = 36

for percent in [5, 10]:
    for pred_len in [24, 36, 48, 60]:
            os.system(
                f"python /home/..../run.py \ 
                --root_path /home/...../Air-Pollution/ \
                --data_path Aotizhongxin.csv \
                --model_id Aotizhongxin_{model}_{seq_len}_{pred_len}_{percent} \
                --data custom \
                --features M \
                --is_training 1 \
                --model {model} \
                --seq_len 36 \
                --label_len 18 \
                --pred_len {pred_len} \
                --e_layers 2 \
                --d_layers 1 \
                --factor 3 \
                --enc_in 11 \
                --dec_in 11\
                --c_out 11 \
                --d_model 768 \
                --d_ff 768 \
                --percent {percent} \
                --top_k 5 \
                --des 'Exp' \
                --itr 1"
            )


