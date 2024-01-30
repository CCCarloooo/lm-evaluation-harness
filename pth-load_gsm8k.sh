# 获得文件夹中后缀为.pth的列表
# 用于从pth文件中提取参数
# 不用name中有7569的

cd /mnt/data2/mxdi/archive/lm-evaluation-harness
model_path='/mnt/data2/mxdi/archive/hf-mirror/llama-7b'



dir='/mnt/data2/mxdi/archive/models/luoooo/plora_r8_400_07_save2/'
ckpt='/mnt/data2/mxdi/archive/models/luoooo/plora_r8_400_07_save2_'
array=()

for file in `ls $dir`
do
    if [[ $file =~ .pth$ ]]
    then
        array+=($file)
    fi
done

# 遍历6次
for i in {0..5}
do  
    adapter_path=${array[$i+12]}
    save_path=$ckpt${array[$i+12]%.pth}
    save1=${save_path##*/}

    (CUDA_VISIBLE_DEVICES=$i python load_lora_or_plora.py \
        --from_pth True \
        --model_path $model_path\
        --adapter_path $dir$adapter_path \
        --save_path $save_path \
        --rank 8 \
        --target_modules 'q_proj' 'k_proj' 'v_proj' 'o_proj' 'gate_proj' 'up_proj' 'down_proj' \

    CUDA_VISIBLE_DEVICES=$i lm_eval --model vllm \
        --model_args pretrained=$save_path,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.7,data_parallel_size=1 \
        --tasks gsm8k \
        --batch_size auto \
        --max_batch_size 32 \
        --output_path /home/mxd/archive/lm-evaluation-harness/results/${save1}/gsm8k.json) &

done