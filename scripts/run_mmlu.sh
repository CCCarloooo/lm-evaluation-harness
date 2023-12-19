cd /mnt/data2/mxdi/archive/lm-evaluation-harness

CUDA_VISIBLE_DEVICES=0,1 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=/mnt/data2/mxdi/archive/hf-mirror/vicuna-7b-v1.5,revision=step100000,dtype="float" \
    --tasks mmlu \
    --batch_size auto:4