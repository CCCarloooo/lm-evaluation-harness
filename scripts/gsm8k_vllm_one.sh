cd /mnt/data2/mxdi/archive/lm-evaluation-harness

origin_path='/mnt/data2/mxdi/archive/FastChat/checkpoints/a40_plora_r8_500_02_4-6_7046-savemodel'

export TMPDIR=/mnt/data2/mxdi/tmp
save1='a40_plora_r8_500_02_4-6_7046-savemodel'

tasks='gsm8k'
new_dir='0-shot'

CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
    --model_args pretrained=$origin_path,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks $tasks \
    --batch_size auto \
    --max_batch_size 32 \
    --output_path /mnt/data2/mxdi/archive/lm-evaluation-harness/results/${tasks}/${new_dir}/${save1}.json
