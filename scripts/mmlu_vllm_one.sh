cd /mnt/data2/mxdi/archive/lm-evaluation-harness

origin_path='/mnt/data2/mxdi/archive/models/luoooo/plora_r8_400_07_save2_2523-savemodel'

export TMPDIR=/mnt/data2/mxdi/tmp
save1='plora_r8_400_07_save2_2523-savemodel_'

tasks='mmlu'


CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 127.0.0.1:5678 --wait-for-client lm_eval --model vllm \
    --model_args pretrained=$origin_path,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1 \
    --tasks $tasks \
    --batch_size auto \
    --max_batch_size 32 \
    --output_path results/${save1}/${tasks}.json
