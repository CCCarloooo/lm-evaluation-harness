cd /mnt/data2/mxdi/archive/lm-evaluation-harness

task_name=race
origin_path=llama7b
SAVE_PATH=llamatuned

(CUDA_VISIBLE_DEVICES='0' lm_eval --model hf \
    --model_args pretrained=/mnt/data2/mxdi/archive/hf-mirror/llama-7b,revision=step100000,dtype="float" \
    --tasks $task_name \
    --batch_size auto:4 \
    --output_path results/${task_name}/${origin_path}.json \
    --max_batch_size 8) &

(CUDA_VISIBLE_DEVICES='1' lm_eval --model hf \
    --model_args pretrained=/mnt/data2/mxdi/archive/FastChat/checkpoints/finetuned_1000_z3,revision=step100000,dtype="float" \
    --tasks $task_name \
    --batch_size auto:4 \
    --output_path results/${task_name}/${SAVE_PATH}.json \
    --max_batch_size 8 ) &