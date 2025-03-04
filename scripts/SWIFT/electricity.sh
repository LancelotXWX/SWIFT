if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=SWIFT_MLP

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

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
  --conv_kernel 25 \
  --hidden_size 128 \
  --conv_dropout 0.01 \
  --enc_in 321 \
  --train_epochs 30 \
  --patience 5 \
  --itr 1 --batch_size 4 --learning_rate 0.001

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
  --conv_kernel 25 \
  --use_convdropout \
  --hidden_size 188 \
  --conv_dropout 0.02 \
  --enc_in 321 \
  --train_epochs 30 \
  --patience 5 \
  --itr 1 --batch_size 4 --learning_rate 0.001

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
  --conv_kernel 25 \
  --use_convdropout \
  --hidden_size 256 \
  --conv_dropout 0.02 \
  --enc_in 321 \
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
  --pred_len 720 \
  --use_convdropout \
  --hidden_size 512 \
  --conv_kernel 25 \
  --conv_dropout 0.02 \
  --enc_in 321 \
  --train_epochs 30 \
  --patience 5 \
  --itr 1 --batch_size 16 --learning_rate 0.005
