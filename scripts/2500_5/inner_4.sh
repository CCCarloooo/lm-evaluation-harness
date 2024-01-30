cd /mnt/data2/mxdi/archive/lm-evaluation-harness
task_name=mmlu
device=4
echo $device

for i in {2100..2500..100}
do
    SAVE_PATH="rank1_ckpt-${i}"

    echo $SAVE_PATH
    CUDA_VISIBLE_DEVICES=$device lm_eval --model hf \
        --model_args pretrained=/mnt/data2/mxdi/archive/models/test_llama_rank1/${SAVE_PATH},revision=step100000,dtype="float" \
        --tasks $task_name \
        --batch_size auto:32 \
        --output_path results/rank1_novllm/${task_name}/${SAVE_PATH}.json \
        --limit 0.5 
done