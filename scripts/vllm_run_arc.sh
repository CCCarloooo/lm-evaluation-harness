cd /mnt/data2/mxdi/archive/lm-evaluation-harness
model_path='/mnt/data2/mxdi/archive/FastChat/checkpoints/finetuned_1000_z3'
origin_path='/mnt/data2/mxdi/archive/hf-mirror/llama-7b'
export TMPDIR=/mnt/data2/mxdi/tmp
save1='finetuned'
save2='origin'
tasks='gsm8k'
(CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
    --model_args pretrained=$origin_path,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks $tasks \
    --batch_size auto \
    --output_path /mnt/data2/mxdi/archive/lm-evaluation-harness/results/${tasks}/${save1}.json ) &

(CUDA_VISIBLE_DEVICES=1 lm_eval --model vllm \
    --model_args pretrained=$model_path,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks $tasks \
    --batch_size auto \
    --output_path /mnt/data2/mxdi/archive/lm-evaluation-harness/results/${tasks}/${save2}.json ) &
