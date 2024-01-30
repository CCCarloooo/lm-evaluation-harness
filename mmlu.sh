cd /mnt/data2/mxdi/archive/lm-evaluation-harness


dir='/mnt/data2/mxdi/archive/models/luoooo/'
array=()

for file in `ls $dir`
do
    if [[ $file =~ -savemodel$ ]] || [[ $file =~ _checkpoint ]]
    then
        array+=($file)
    fi
done

declare -A arrays

# split the array into 4 arrays
for i in {0..3}
do
    divid=${#array[@]}/4
    arrays[$i]=${array[@]:$((divid*i)):divid}
done

for i in {0..3}
do 
(    for j in ${arrays[$i]}
    do 
        save_path=$dir$j
        save1=${save_path##*/}

        CUDA_VISIBLE_DEVICES=$i lm_eval --model vllm \
        --model_args pretrained=$save_path,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1 \
        --tasks mmlu \
        --batch_size auto \
        --max_batch_size 32 \
        --output_path results/${save1}/mmlu.json

    done) & 

done