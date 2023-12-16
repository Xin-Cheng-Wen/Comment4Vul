
# python run_remove.py \
#     --output_dir ./saved_models/Devign_remove \
#     --model_name_or_path ./models \
#     --do_train \
#     --train_data_file data/Devign/remove/train_cdata.jsonl \
#     --eval_data_file data/Devign/remove/valid_cdata.jsonl \
#     --test_data_file data/Devign/remove/test_cdata.jsonl \
#     --num_train_epochs 5 \
#     --block_size 512 \
#     --train_batch_size 32 \
#     --eval_batch_size 16 \
#     --learning_rate 2e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 2>&1 | tee train_Devign_remove_acc.log


# python run_remove.py \
#     --output_dir ./saved_models/Devign_remove \
#     --model_name_or_path ./models \
#     --do_test \
#     --train_data_file data/Devign/remove/train_cdata.jsonl \
#     --eval_data_file data/Devign/remove/valid_cdata.jsonl \
#     --test_data_file data/Devign/remove/test_cdata.jsonl \
#     --num_train_epochs 5 \
#     --block_size 512 \
#     --train_batch_size 32 \
#     --eval_batch_size 16 \
#     --learning_rate 2e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456


python run_remove.py \
    --output_dir ./saved_models/Devign_remove \
    --model_name_or_path ./models \
    --do_train \
    --train_data_file data/Devign/commentV2/train_cdata_results_V2without_here_matchV3.jsonl \
    --eval_data_file data/Devign/commentV2/valid_cdata_results_V2without_here_matchV3.jsonl \
    --test_data_file data/Devign/commentV2/test_cdata_results_V2without_here_matchV3.jsonl \
    --num_train_epochs 8 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 2>&1 | tee train_Devign_remove_acc_commentV4.log


python run_remove.py \
    --output_dir ./saved_models/Devign_remove \
    --model_name_or_path ./models \
    --do_test \
    --train_data_file data/Devign/commentV2/train_cdata_results_V2without_here_matchV3.jsonl \
    --eval_data_file data/Devign/commentV2/valid_cdata_results_V2without_here_matchV3.jsonl \
    --test_data_file data/Devign/commentV2/test_cdata_results_V2without_here_matchV3.jsonl \
    --num_train_epochs 8 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456