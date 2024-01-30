cd /mnt/data2/mxdi/archive/lm-evaluation-harness

origin_path='/mnt/data2/mxdi/archive/FastChat/checkpoints/plora_r1/0105_just_clearopti_test_warmup'
model_path='/mnt/data2/mxdi/archive/FastChat/checkpoints/rank1lora/0105_intergration'

export TMPDIR=/mnt/data2/mxdi/tmp
save1='0105_just_clearopti_test_warmup'
save2='rank1_0105_intergration'

tasks='gsm8k'
new_dir='with_format'

(CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
    --model_args pretrained=$origin_path,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks $tasks \
    --batch_size auto \
    --max_batch_size 32 \
    --output_path /mnt/data2/mxdi/archive/lm-evaluation-harness/results/${tasks}/${new_dir}/${save1}.json ) &

(CUDA_VISIBLE_DEVICES=1 lm_eval --model vllm \
    --model_args pretrained=$model_path,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks $tasks \
    --batch_size auto \
    --max_batch_size 32 \
    --output_path /mnt/data2/mxdi/archive/lm-evaluation-harness/results/${tasks}/${new_dir}/${save1}.json ) &
