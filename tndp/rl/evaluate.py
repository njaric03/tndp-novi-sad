import numpy as np
import torch

from tndp.core.network import TransitNetwork
from tndp.rl.env import HALT, TndpEnv
from tndp.rl.features import edge_tensors, node_features


# odigraj epizodu politikom; sample=True vuce iz distribucije (trening), sample=False uzima argmax (greedy dekodiranje)
def rollout(policy, env, sample=True, gen=None):
    edge_index, edge_attr = edge_tensors(env.city)
    log_probs, entropies = [], []
    env.reset()
    while not env.done:
        decision, mask = env.decision()
        h = policy.encode(node_features(env, policy.features), edge_index, edge_attr)
        logits = policy.action_logits(h, decision, mask, env.ends)
        dist = torch.distributions.Categorical(logits=logits)
        if sample:
            a = torch.multinomial(dist.probs, 1, generator=gen).squeeze(0)
        else:
            a = logits.argmax()
        log_probs.append(dist.log_prob(a))
        entropies.append(dist.entropy())
        # halt je poslednji logit kad je odluka HALT
        is_halt = decision == HALT and int(a) == len(logits) - 1
        env.step(-1 if is_halt else int(a))
    reward, res = env.reward()
    return (TransitNetwork(routes=env.routes), reward, res,
            torch.stack(log_probs).sum(), torch.stack(entropies).mean())


@torch.no_grad()
def decode(policy, city, num_routes, min_len=2, max_len=8, alpha=0.5):
    env = TndpEnv(city, num_routes, min_len, max_len, alpha)
    net, reward, res, _, _ = rollout(policy, env, sample=False)
    return net, res


# najbolja od k sampled epizoda (jeftino poboljsanje, Kool et al
@torch.no_grad()
def decode_sampling(policy, city, num_routes, k=32, min_len=2, max_len=8,
                    alpha=0.5, seed=0):
    gen = torch.Generator().manual_seed(seed)
    env = TndpEnv(city, num_routes, min_len, max_len, alpha)
    best_net, best_res, best_r = None, None, -np.inf
    any_net, any_res, any_r = None, None, -np.inf  # i ako nijedna nije validna
    for _ in range(k):
        net, reward, res, _, _ = rollout(policy, env, sample=True, gen=gen)
        if reward > any_r:
            any_net, any_res, any_r = net, res, reward
        # maskiranje ne garantuje validnost van distribucije treninga (na Mumfordu je min_len=10 pa se linija ume zaglaviti
        if net.check(city, num_routes, min_len, max_len):
            continue
        if reward > best_r:
            best_net, best_res, best_r = net, res, reward
    if best_net is None:  # nijedan uzorak nije validan; neka pozivalac vidi
        return any_net, any_res
    return best_net, best_res
