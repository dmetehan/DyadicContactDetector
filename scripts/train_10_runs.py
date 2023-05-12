import subprocess

# venv/bin/python train.py configs/001_config.yaml --test --log_test_results
for i in range(10):
    cmd = ['venv/bin/python', 'train.py', 'configs/001_config.yaml', '--test', '--log_test_results']
    print(cmd)
    subprocess.Popen(cmd).wait()
