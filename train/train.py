#%%
import os
import sys
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(os.path.join(os.pardir, 'games'))
from mlx import core as mx

from dataset.make_data import episode_loader
from models.cnn_policy import CNNPolicy
#%%
episode_paths = glob.glob(os.path.join(os.pardir, "logs", "*.npz"))
tr_iter = episode_loader(episode_paths)

# %%
cnn = CNNPolicy(3, 512)
for batch in tr_iter:
    print(batch)
    output = cnn(mx.array(batch["state"]))
    print(output)
    print(output.shape)
    break
# %%





# %%
