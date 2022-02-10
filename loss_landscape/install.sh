#!/usr/bin/env zsh

python -m pip install torch==1.10.1+cu113 \
  torchvision==0.11.2+cu113 \
  torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html \
  pandas dill python-box matplotlib scikit-learn ipdb ipython tensorboard tqdm