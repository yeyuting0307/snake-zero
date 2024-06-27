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
#%%
episode_paths = glob.glob(os.path.join(os.pardir, "logs", "*.npz"))
tr_iter = episode_loader(episode_paths)

# %%
policy_net = SnakeRL(3, 4)
target_net = copy.deepcopy(policy_net)
optimizer = optim.AdamW(learning_rate=1e-4)

#%%
def train_epoch(policy_net, target_net, train_iter, optimizer, epoch):
    def train_step(
            policy_net, 
            target_net,
            state_batch, 
            action_batch, 
            reward_batch, 
            next_state_batch, 
            done_batch,
        ):
        state_action_values = mx.expand_dims(policy_net(state_batch).max(-1),1)

        next_state_values= target_net(next_state_batch).max(-1)
        # final_indices = [i for i, d in enumerate(done_batch) if d == 1]
        # next_state_values[final_indices] = mx.zeros(len(final_indices))
        GAMMA=0.99
        expected_state_action_values = ((next_state_values * GAMMA) + reward_batch)


        loss = mx.mean(nn.losses.mse_loss(state_action_values, mx.expand_dims(expected_state_action_values, 1)))
        acc = mx.mean(mx.argmax(state_action_values, axis=1) == expected_state_action_values)
        return loss, acc

    losses = []
    accs = []
    samples_per_sec = []

    state = [policy_net.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(
        state_batch, 
        action_batch, 
        reward_batch, 
        next_state_batch, 
        done_batch
        ):
        train_step_fn = nn.value_and_grad(policy_net, train_step)
        (loss, acc), grads = train_step_fn(
            policy_net,
            target_net, 
            state_batch, 
            action_batch, 
            reward_batch, 
            next_state_batch, 
            done_batch
        )
        optimizer.update(policy_net, grads)
        return loss, acc

    for batch_counter, batch in enumerate(train_iter):
        state_batch = mx.array(batch["state"]).astype(mx.float16)
        action_batch = mx.array(batch["action"]).astype(mx.float16)
        reward_batch = mx.array(batch["reward"]).astype(mx.float16)
        next_state_batch = mx.array(batch["next_state"]).astype(mx.float16)
        done_batch = mx.array(batch["done"]).astype(mx.float16)
        
        tic = time.perf_counter()
        loss, acc = step(
            state_batch, 
            action_batch, 
            reward_batch, 
            next_state_batch, 
            done_batch
        )
        target_net = copy.deepcopy(policy_net)
        
        toc = time.perf_counter()
        loss = loss.item()
        acc = acc.item()
        losses.append(loss)
        accs.append(acc)
        throughput = state_batch.shape[0] / (toc - tic)
        samples_per_sec.append(throughput)
        if batch_counter % 10 == 0:
            print(
                " | ".join(
                    (
                        f"Epoch {epoch:02d} [{batch_counter:03d}]",
                        f"Train loss {loss:.3f}",
                        f"Train acc {acc:.3f}",
                        f"Throughput: {throughput:.2f} images/second",
                    )
                )
            )

    mean_tr_loss = mx.mean(mx.array(losses))
    mean_tr_acc = mx.mean(mx.array(accs))
    samples_per_sec = mx.mean(mx.array(samples_per_sec))
    return mean_tr_loss, mean_tr_acc, samples_per_sec

#%%

train_epoch(policy_net, target_net, tr_iter, optimizer, 10)

# %%
