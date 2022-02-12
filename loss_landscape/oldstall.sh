#!/usr/bin/env zsh

python -m pip install torch==1.7.0+cu92 \
  torchvision==0.8.1+cu92 \
  torchaudio==0.7.0 -f https://download.pytorch.org/whl/cu92/torch_stable.html \
  pandas dill python-box matplotlib scikit-learn ipdb ipython tensorboard tqdm
