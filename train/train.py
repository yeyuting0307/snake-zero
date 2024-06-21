#%%
import os
import sys
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(os.path.join(os.pardir, 'games'))
from dataset.make_data import episode_loader

episode_paths = glob.glob(os.path.join(os.pardir, "logs", "*.npz"))
tr_iter = episode_loader(episode_paths)
# %%
for batch in tr_iter:
    print(batch)
    break
# %%
