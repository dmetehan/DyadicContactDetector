conda create -n openmmlab python=3.8 pytorch=1.6.0 cudatoolkit=10.1 torchvision -c pytorch -y
conda activate openmmlab
~/anaconda3/envs/openmmlab/bin/pip install --no-cache-dir --upgrade pip wheel setuptools
~/anaconda3/envs/openmmlab/bin/pip install --no-cache-dir mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
cd mmpose || exit
~/anaconda3/envs/openmmlab/bin/pip install -r requirements/build.txt
~/anaconda3/envs/openmmlab/bin/pip install --no-cache-dir -e .
~/anaconda3/envs/openmmlab/bin/pip install mmdet
