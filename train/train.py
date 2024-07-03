#%%
import os
import sys
import glob
import time
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(os.path.join(os.pardir, 'games'))

from mlx import nn
from mlx import core as mx
from mlx import optimizers as optim
from functools import partial

from dataset.make_data import episode_loader
from models.snake_zero import SnakeRL

episode_paths = glob.glob(os.path.join(os.pardir, "logs", "*.npz"))
train_iter = episode_loader(episode_paths)

# %%
policy_net = SnakeRL(3, 4)
target_net = copy.deepcopy(policy_net)
optimizer = optim.AdamW(learning_rate=1e-4)
GAMMA = 0.99

def loss_fn(policy_net, target_net, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
    state_action_values = policy_net(state_batch) # (32, 4)

    action_index = action_batch[:,mx.newaxis].astype(mx.int16)
    state_action_values = mx.take_along_axis(state_action_values, action_index, axis=-1)

    next_state_values = target_net(next_state_batch).max(1)

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    expected_state_action_values = mx.expand_dims(expected_state_action_values, 1)
    return nn.losses.smooth_l1_loss(state_action_values, expected_state_action_values)


loss_and_grad_fn = nn.value_and_grad(policy_net, loss_fn)
#%%

def train_epoch(policy_net, target_net, train_iter, optimizer, epoch):
    samples_per_sec = []
    losses = []
    for i, batch in enumerate(train_iter):
        tic = time.perf_counter()
        state_batch = mx.array(batch["state"]).astype(mx.float16)
        action_batch = mx.array(batch["action"]).astype(mx.float16)
        reward_batch = mx.array(batch["reward"]).astype(mx.float16)
        next_state_batch = mx.array(batch["next_state"]).astype(mx.float16)
        done_batch = mx.array(batch["done"]).astype(mx.float16)
        
        loss, grads = loss_and_grad_fn(policy_net, target_net, state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        optimizer.update(policy_net, grads)

        toc = time.perf_counter()
        losses.append(loss.item())

        target_net = copy.deepcopy(policy_net)

        throughput = state_batch.shape[0] / (toc - tic)
        samples_per_sec.append(throughput)
        if i % 1 == 0:
            print(
                " | ".join(
                    (
                        f"Epoch {epoch:02d} [{i:03d}]",
                        f"Train loss {loss.item():.3f}",
                        f"Throughput: {throughput:.2f} images/second",
                    )
                )
            )
    mean_tr_loss = mx.mean(mx.array(losses))
    samples_per_sec = mx.mean(mx.array(samples_per_sec))
    return mean_tr_loss, samples_per_sec

#%%
epochs = 5
for epoch in range(epochs):
    mean_tr_loss, samples_per_sec = train_epoch(policy_net, target_net, train_iter, optimizer, epoch)
    print(f"{epoch} | {mean_tr_loss} | {samples_per_sec}")

