cd /mnt/data2/mxdi/archive/lm-evaluation-harness

origin_path='/mnt/data2/mxdi/archive/FastChat/checkpoints/finetuned_4epoch/checkpoint-7569'

export TMPDIR=/mnt/data2/mxdi/tmp
save1='gsm8k'
save2='gsm8k_5shot'

tasks='gsm8k'
tasks_='gsm8k_5shot'

new_dir='fschat_format_finetuned_4epoch_checkpoint-7569'

(CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
    --model_args pretrained=$origin_path,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks $tasks \
    --batch_size auto \
    --max_batch_size 32 \
    --output_path /mnt/data2/mxdi/archive/lm-evaluation-harness/results/${new_dir}/${save1}.json ) &

(CUDA_VISIBLE_DEVICES=1 lm_eval --model vllm \
    --model_args pretrained=$origin_path,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks $tasks_ \
    --batch_size auto \
    --max_batch_size 32 \
    --output_path /mnt/data2/mxdi/archive/lm-evaluation-harness/results/${new_dir}/${save2}.json ) &
