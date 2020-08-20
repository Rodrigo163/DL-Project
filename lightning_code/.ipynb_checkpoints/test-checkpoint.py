import torch

if torch.cuda.is_available():
    print('FInally using CUDA!')
else:
    print('still nothing :/')