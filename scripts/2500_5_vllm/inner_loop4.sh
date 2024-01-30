cd /mnt/data2/mxdi/archive/lm-evaluation-harness
task_name=gsm8k
device=5
echo $device

for i in {2100..2500..100}
do
    SAVE_PATH="rank1_ckpt-${i}"

    echo $SAVE_PATH
    CUDA_VISIBLE_DEVICES=$device lm_eval --model vllm \
        --model_args pretrained=/mnt/data2/mxdi/archive/models/test_llama_rank1/${SAVE_PATH},tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
        --tasks $task_name \
        --batch_size auto \
        --max_batch_size 32 \
        --output_path results/rank1/${task_name}/${SAVE_PATH}.json 
done