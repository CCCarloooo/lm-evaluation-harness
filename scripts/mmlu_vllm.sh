cd /mnt/data2/mxdi/archive/lm-evaluation-harness

origin_path='/mnt/data2/mxdi/archive/FastChat/checkpoints/a40_0115_rank1_1e-4_0.999_3epoch'
model_path='/mnt/data2/mxdi/archive/FastChat/checkpoints/a40_plora_r1_0113_interval400_1e-4_0999_constant_08lora'

export TMPDIR=/mnt/data2/mxdi/tmp
save1='a40_0115_rank1_1e-4_0.999_3epoch'
save2='a40_plora_r1_0113_interval400_1e-4_0999_constant_08lora'

tasks='mmlu'
new_dir='fschat_format'

(CUDA_VISIBLE_DEVICES=2 lm_eval --model vllm \
    --model_args pretrained=$origin_path,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks $tasks \
    --batch_size auto \
    --max_batch_size 32 \
    --limit 0.5 \
    --output_path /mnt/data2/mxdi/archive/lm-evaluation-harness/results/${tasks}/${new_dir}/${save1}.json ) &

(CUDA_VISIBLE_DEVICES=3 lm_eval --model vllm \
    --model_args pretrained=$model_path,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks $tasks \
    --batch_size auto \
    --max_batch_size 32 \
    --limit 0.5 \
    --output_path /mnt/data2/mxdi/archive/lm-evaluation-harness/results/${tasks}/${new_dir}/${save2}.json ) &
