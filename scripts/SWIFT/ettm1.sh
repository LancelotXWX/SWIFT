if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=SWIFT_Linear

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTM1
data_name=ETTm1

seq_len=720

python -u run_longExp.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --no_revin \
  --conv_kernel 3 \
  --use_convdropout \
  --conv_dropout 0.1 \
  --enc_in 7 \
  --train_epochs 30 \
  --patience 5 \
  --itr 1 --batch_size 64 --learning_rate 0.001

python -u run_longExp.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --no_revin \
  --conv_kernel 3 \
  --use_convdropout \
  --conv_dropout 0.1 \
  --enc_in 7 \
  --train_epochs 30 \
  --patience 5 \
  --itr 1 --batch_size 256 --learning_rate 0.005

python -u run_longExp.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --no_revin \
  --conv_kernel 3 \
  --use_convdropout \
  --conv_dropout 0.1 \
  --enc_in 7 \
  --train_epochs 30 \
  --patience 5 \
  --itr 1 --batch_size 256 --learning_rate 0.005

python -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len 720 \

    --no_revin \
    --conv_kernel 3 \
    --use_convdropout \
    --conv_dropout 0.1 \
    --enc_in 7 \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 256 --learning_rate 0.01