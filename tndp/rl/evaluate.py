import numpy as np
import torch

from tndp.core.network import TransitNetwork
from tndp.rl.env import HALT, TndpEnv
from tndp.rl.model import edge_tensors, node_features


# odigraj epizodu politikom; sample=True vuče iz distribucije (trening),
# sample=False uzima argmax (greedy dekodiranje). vraća mrežu, log prob
# sumu i entropiju (za REINFORCE)
def rollout(policy, env, sample=True):
    edge_index, edge_attr = edge_tensors(env.city)
    log_probs, entropies = [], []
    env.reset()
    while not env.done:
        decision, mask = env.decision()
        h = policy.encode(node_features(env), edge_index, edge_attr)
        logits = policy.action_logits(h, decision, mask)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample() if sample else logits.argmax()
        log_probs.append(dist.log_prob(a))
        entropies.append(dist.entropy())
        # halt je poslednji logit kad je odluka HALT
        is_halt = decision == HALT and int(a) == len(logits) - 1
        env.step(-1 if is_halt else int(a))
    reward, res = env.reward()
    return (TransitNetwork(routes=env.routes), reward, res,
            torch.stack(log_probs).sum(), torch.stack(entropies).mean())


# greedy dekodiranje politike na zadatom gradu
@torch.no_grad()
def decode(policy, city, num_routes, min_len=2, max_len=8, alpha=0.5):
    env = TndpEnv(city, num_routes, min_len, max_len, alpha)
    net, reward, res, _, _ = rollout(policy, env, sample=False)
    return net, res


# najbolja od k sampled epizoda (jeftino poboljšanje, Kool et al. trik)
@torch.no_grad()
def decode_sampling(policy, city, num_routes, k=32, min_len=2, max_len=8,
                    alpha=0.5, seed=0):
    torch.manual_seed(seed)
    env = TndpEnv(city, num_routes, min_len, max_len, alpha)
    best_net, best_res, best_r = None, None, -np.inf
    for _ in range(k):
        net, reward, res, _, _ = rollout(policy, env, sample=True)
        if reward > best_r:
            best_net, best_res, best_r = net, res, reward
    return best_net, best_res
