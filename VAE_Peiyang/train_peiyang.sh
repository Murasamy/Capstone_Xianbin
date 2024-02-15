/usr/local/bin/python3 src/train_peiyang_v2.py \
--lr 0.0001 \
--epochs 10 \
--batch_size 1 \
--train_data_dir data \
--output_dir checkpoints \
--read_data_format txt \
--txt_path txtfiles/SSE50.txt \
--save_steps 10 \
--debug_mode True \
--shuffle_train_data True \
--eval_data_proportion 0.2 \
--save_eval_img True \
--save_train_img True \
--save_input_img True \
--save_cat_img True \

