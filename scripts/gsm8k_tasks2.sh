cd /mnt/data2/mxdi/archive/lm-evaluation-harness

origin_path='/mnt/data2/mxdi/archive/FastChat/checkpoints/plora_multitask_r1_0110_qv_interval400-200_1e-4_0.999_enoughsave_scaling'

export TMPDIR=/mnt/data2/mxdi/tmp
save1='gsm8k'
save2='gsm8k_5shot'

tasks='gsm8k'
tasks_='gsm8k_5shot'

new_dir='plora_multitask_r1_0110_qv_interval400-200_1e-4_0.999_enoughsave_scaling'

(CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args pretrained=$origin_path,revision=step100000,dtype="float" \
    --tasks $tasks \
    --batch_size auto \
    --max_batch_size 2 \
    --output_path /mnt/data2/mxdi/archive/lm-evaluation-harness/results/${new_dir}/${save1}.json \
    --limit 200) &

(CUDA_VISIBLE_DEVICES=1 lm_eval --model hf \
    --model_args pretrained=$origin_path,revision=step100000,dtype="float" \
    --tasks $tasks_ \
    --batch_size auto \
    --max_batch_size 2 \
    --output_path /mnt/data2/mxdi/archive/lm-evaluation-harness/results/${new_dir}/${save2}.json \
    --limit 200) &
