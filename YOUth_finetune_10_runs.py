import subprocess

for i in range(10):
    cmd = ['venv/bin/python', 'YOUth_train.py', 'configs/001_config.yaml', 'YOUth_experiments', '--finetune', '--test', '--log_test_results']
    print(cmd)
    subprocess.Popen(cmd).wait()
