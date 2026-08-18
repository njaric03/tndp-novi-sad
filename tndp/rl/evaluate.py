import numpy as np
import torch

from tndp.core.network import TransitNetwork
from tndp.rl.env import HALT, TndpEnv
from tndp.rl.features import edge_tensors, node_features

# Podrazumevani raspored temperatura za uzorkovano dekodiranje. Ista k uzoraka se
# dele preko ovih vrednosti umesto da svi idu na T=1. Mereno na 20 held-out gradova
# i 5 instanci iz literature: bolje na oba, jer best-of-k zivi od raznovrsnosti
# kandidata, a koliko sirenja treba zavisi od toga koliko je instanca daleko od
# trening raspodele i unaprijed se ne zna.
TEMPS = (1.0, 2.0, 4.0, 8.0)


# odigraj epizodu politikom; sample=True vuce iz distribucije (trening), sample=False uzima argmax (greedy dekodiranje)
# temp deli logite samo pri uzorkovanju: <1 izostrava, >1 siri. log_prob i entropija ostaju na
# NEDIRNUTOJ raspodeli, inace bi trening menjao ono sto optimizuje kad neko dira dekoder
def rollout(policy, env, sample=True, gen=None, temp=1.0):
    edge_index, edge_attr = edge_tensors(env.city)
    log_probs, entropies = [], []
    env.reset()
    while not env.done:
        decision, mask = env.decision()
        h = policy.encode(node_features(env, policy.features), edge_index, edge_attr)
        logits = policy.action_logits(h, decision, mask, env.ends)
        dist = torch.distributions.Categorical(logits=logits)
        if sample:
            probs = (dist.probs if temp == 1.0
                     else torch.softmax(logits / temp, dim=-1))
            a = torch.multinomial(probs, 1, generator=gen).squeeze(0)
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
                    alpha=0.5, seed=0, temps=TEMPS):
    gen = torch.Generator().manual_seed(seed)
    env = TndpEnv(city, num_routes, min_len, max_len, alpha)
    best_net, best_res, best_r = None, None, -np.inf
    any_net, any_res, any_r = None, None, -np.inf  # i ako nijedna nije validna
    # temps rasporedjuje ista k uzoraka preko vise temperatura, umesto svih na jednoj.
    # Koliko sirenje vredi zavisi od toga koliko je instanca daleko od trening
    # raspodele, a to se ne zna unaprijed; posto se cilj racuna za svaki uzorak,
    # izbor se napravi sam i ukupan broj evaluacija ostaje isti.
    temps = tuple(temps) or (1.0,)
    raspored = [t for i, t in enumerate(temps)
                for _ in range(k // len(temps) + (1 if i < k % len(temps) else 0))]
    for t in raspored:
        net, reward, res, _, _ = rollout(policy, env, sample=True, gen=gen,
                                         temp=t)
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
