import subprocess
import signal
import os

available_gpus = (0, 2, 3, 4, 5, 6, 7)
num_runs = 2

task_num = len(available_gpus) * num_runs
processes = []

# 信号处理函数，用于终止所有子进程
def terminate_processes(signum, frame):
    for proc in processes:
        try:
            # 发送SIGTERM信号
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception as e:
            print(f"Failed to kill process {proc.pid}: {e}")

# 注册信号处理函数
signal.signal(signal.SIGINT, terminate_processes)  # 处理Ctrl-C
signal.signal(signal.SIGTERM, terminate_processes) # 处理kill

for gpu_i, gpu in enumerate(available_gpus):
    for run_i in range(num_runs):
        task_id = gpu_i * num_runs + run_i
        log_file_name = f"output_{task_id}.log"

        command = f"CUDA_VISIBLE_DEVICES={gpu} python scripts/sim_supervise/distrib_sim.py {task_id} {task_num} > {log_file_name} 2>&1"
        print(command)
        proc = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)

        processes.append(proc)

try:
    # 等待所有子进程完成
    for proc in processes:
        proc.wait()
except KeyboardInterrupt:
    # 如果用户中断执行(通常是通过Ctrl+C), 终止所有子进程
    terminate_processes(None, None)
