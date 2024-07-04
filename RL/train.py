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
policy_net = SnakeRL(3, 5)


def train_policy(policy_net):
    try:
        episode_paths = glob.glob(os.path.join("logs", "*.npz"))
        print("episode_paths: ", episode_paths)
        train_iter = episode_loader(episode_paths)

        target_net = copy.deepcopy(policy_net)
        optimizer = optim.AdamW(learning_rate=1e-5)
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

        def valid_shape(batch):
            shape = None
            for b in batch:
                if shape is None:
                    shape = b.shape
                    continue
                if b.shape != shape:
                    print("Invalid shape: ", b.shape, "!=", shape)
                    return False
            return True

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

                if not valid_shape(state_batch):
                    continue
                
                loss, grads = loss_and_grad_fn(policy_net, target_net, state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                optimizer.update(policy_net, grads)

                mx.eval(policy_net, grads)
                toc = time.perf_counter()
                losses.append(loss.item())

                if i % 20 == 0:
                    target_net = copy.deepcopy(policy_net)

                tictoc = toc - tic
                if i % 1 == 0:
                    print(
                        " | ".join(
                            (
                                f"Epoch {epoch:02d} [{i:03d}]",
                                f"Train loss {loss.item():.3f}",
                                f"Time: {tictoc:.2f} sec",
                            )
                        )
                    )
            
            mean_tr_loss = mx.mean(mx.array(losses))
            
            return mean_tr_loss

        for epoch in range(1):
            train_iter = episode_loader(episode_paths)
            mean_tr_loss = train_epoch(policy_net, target_net, train_iter, optimizer, epoch)
            print(f"Epoch {epoch:02d} | Mean train loss: {mean_tr_loss.item():.3f} ")

        policy_net.save_weights(os.path.join("checkpoints", "snake_zero_weights.npz"))
        print("saved weights!")
    except Exception as e:
        print(e)
    finally:
        for file in episode_paths:
            os.remove(file)
    return policy_net