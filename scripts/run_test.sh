cd /mnt/data2/mxdi/archive/lm-evaluation-harness

task_name=bbh
origin_path='open_llama_3b'
SAVE_PATH='finetuned_open_llama_3b_'
pretrain_path='/mnt/data2/mxdi/archive/hf-mirror/open_llama_3b'
finetuned_path='/mnt/data2/mxdi/archive/models/1221_finetune_orca/checkpoint-21582'

(CUDA_VISIBLE_DEVICES='0' lm_eval --model hf \
    --model_args pretrained=$pretrain_path,revision=step100000,dtype="float" \
    --tasks $task_name \
    --batch_size auto:4 \
    --output_path results/${task_name}/sys-new/${origin_path}.json) &

(CUDA_VISIBLE_DEVICES='2' lm_eval --model hf \
    --model_args pretrained=$finetuned_path,revision=step100000,dtype="float" \
    --tasks $task_name \
    --batch_size auto:4 \
    --output_path results/${task_name}/sys-new/${SAVE_PATH}.json ) &