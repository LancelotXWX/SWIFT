if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=SWIFT_Linear

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
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
    --not_independent \
    --use_convdropout \
    --conv_dropout 0.01 \
    --enc_in 21 \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 16 --learning_rate 0.001
  
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
    --not_independent \
    --conv_dropout 0.01 \
    --enc_in 21 \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 16 --learning_rate 0.001

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
    --not_independent \
    --conv_dropout 0.01 \
    --enc_in 21 \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 16 --learning_rate 0.001

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
    --conv_kernel 25 \
    --not_independent \
    --use_convdropout \
    --conv_dropout 0.01 \
    --enc_in 21 \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 16 --learning_rate 0.001