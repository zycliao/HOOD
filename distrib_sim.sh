#
#!/bin/bash

# 定义可用的GPU IDs
# 假设可用的GPU IDs是0, 1, 2
available_gpus=(0 2 3 4 5 6 7)
num_gpus=${#available_gpus[@]}

# 定义需要运行的次数
num_runs=3

# 遍历每个GPU ID
for gpu_id in "${available_gpus[@]}"; do
    # 对每个GPU运行三次Python脚本
    for ((i=0; i<num_runs; i++)); do
        # 在后台运行Python脚本并传递GPU ID
        task_id=$((gpu_id * num_runs + i))
        task_num=$(())
        nohup CUDA_VISIBLE_DEVICES=$gpu_id python scripts/sim_supervise/distrib_sim.py ${task_id} ${task_num}  > "output_${task_id}.log" 2>&1 &
    done
done

# 等待所有后台进程完成
wait

CUDA_VISIBLE_DEVICES=0 python scripts/sim_supervise/distrib_sim.py 0 14