import subprocess

for i in range(1, 8):
    for j in range(10):
        cmd = ['venv/bin/python', 'YOUth_train.py', f'configs/a0{i}_config.yaml', 'Ablation', '--test', '--log_test_results']
        print(cmd)
        subprocess.Popen(cmd).wait()
