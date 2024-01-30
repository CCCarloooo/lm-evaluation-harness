cd /mnt/data2/mxdi/archive/lm-evaluation-harness/scripts/2500_5_vllm

task_name=gsm8k

for i in {0..4..1}
do
    (bash inner_loop${i}.sh) &
done