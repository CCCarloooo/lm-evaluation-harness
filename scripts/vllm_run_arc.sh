cd /mnt/data2/mxdi/archive/lm-evaluation-harness

CUDA_VISIBLE_DEVICES=4,5 lm_eval --model vllm \
    --model_args pretrained=/mnt/data2/mxdi/archive/hf-mirror/vicuna-7b-v1.5,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks mmlu \
    --batch_size auto