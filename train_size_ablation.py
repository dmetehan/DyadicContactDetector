import subprocess

for i in range(10):
    for j in range(10):
        cmd = ['venv/bin/python', 'YOUth_train.py', f'configs/a3{i}_config.yaml', 'YOUth_ablations', '--test', '--log_test_results']
        print(cmd)
        subprocess.Popen(cmd).wait()
