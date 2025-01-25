if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=SWIFT_Linear

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

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
  --conv_kernel 13 \
  --use_convdropout \
  --conv_dropout 0.4 \
  --no_revin \
  --enc_in 7 \
  --train_epochs 30 \
  --patience 5 \
  --itr 1 --batch_size 256 --learning_rate 0.02


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
  --conv_kernel 13 \
  --use_convdropout \
  --conv_dropout 0.4 \
  --no_revin \
  --enc_in 7 \
  --train_epochs 30 \
  --patience 5 \
  --itr 1 --batch_size 256 --learning_rate 0.03

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
  --conv_kernel 13 \
  --use_convdropout \
  --conv_dropout 0.4 \
  --no_revin \
  --enc_in 7 \
  --train_epochs 30 \
  --patience 5 \
  --itr 1 --batch_size 128 --learning_rate 0.01

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
  --conv_kernel 13 \
  --use_convdropout \
  --conv_dropout 0.4 \
  --no_revin \
  --enc_in 7 \
  --train_epochs 30 \
  --patience 5 \
  --itr 1 --batch_size 256 --learning_rate 0.03