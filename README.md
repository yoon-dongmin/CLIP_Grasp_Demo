## Table of Contents
* [Environment Setup](#environment-setup)
* [Code Demo](#code-demo)
* [Pretrain Model](#pretrain-model)

## Environment Setup
1. create conda enviroment with Python=3.10  
`conda create -n clipgrasp python=3.10`  
`conda activate clipgrasp`
2. install pytorch 1.13.0, torchvision 0.14.0 with compatible cuda version (or any compatible torch version)  
`conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia`
3. install required package  
`pip install -r requirements.txt`

### Code Demo
```bash
python demo.py --text_query "Can you give me something to drink?"
```

### Pretrained Model
[download](https://drive.google.com/drive/u/0/folders/1AZK4dWNfBO1QhC3HYdJLm5Th_J59wyrD)

