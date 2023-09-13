import subprocess

# venv/bin/python train.py configs/main/010_config.yaml exp/YOUth --test --log_test_results
# cmd = ['venv/bin/python', 'train.py',
#                                 'configs/main/012_config.yaml', '--test', '--log_test_results']
# cmd = ['venv/bin/python', 'YOUth_train.py',
#                                 'configs/main/014_config.yaml', 'exp/YOUth', '--test', '--log_test_results']
for i in range(10):
    cmd = ['venv/bin/python', 'YOUth_train.py',
                                    'configs/main/010_config.yaml', 'exp/YOUth', '--test', '--log_test_results']
    print(cmd)
    subprocess.Popen(cmd).wait()
