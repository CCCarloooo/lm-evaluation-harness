cd /mnt/data2/mxdi/archive/lm-evaluation-harness/scripts/2500_5

task_name=mmlu

for i in {0..4..1}
do
    (bash inner_${i}.sh) &
done