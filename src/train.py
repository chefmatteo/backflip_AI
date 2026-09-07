import torch
torch.set_num_threads(1)
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
import numpy as np
import math
import socket
import struct
from time import sleep
import os


# dimensions
NUM_ENVS   = 25
NUM_LINKS  = 14
NUM_JOINTS = 13

RAW_STATE_DIM = 2 + NUM_LINKS * 14
PAD_DIM       = 9                        # unused tail the saved checkpoints were trained with
state_dim     = RAW_STATE_DIM + PAD_DIM
action_dim    = 3 * NUM_JOINTS

# training
N = 100000
T = 1024
K_epochs = 5
ENT_COEF = 0.0
CLIP     = 5.0
VEL_SCALE = 0.01

# skeleton layout
JOINTS = [(7,6),(8,7),(13,8),(9,8),(10,8),(11,9),(12,10),(4,6),(5,6),(2,4),(3,5),(0,2),(1,3)]
CHILD  = torch.tensor([c for c, p in JOINTS])
PARENT = torch.tensor([p for c, p in JOINTS])
ROOT   = 6
END_EFFECTORS = [0, 1, 11, 12]
NON_END_LINKS = [i for i in range(NUM_LINKS) if i not in END_EFFECTORS]
LINK_MASS = torch.tensor([1., 1, 2.5, 2.5, 4.5, 4.5, 5, 7, 8, 1.8, 1.8, 1.2, 1.2, 3])
JOINT_W   = torch.tensor([.5, .5, .3, .3, .3, .2, .2, .5, .5, .3, .3, .2, .2])

# rest pose
REST = torch.tensor([
    0,0,0,   0,0,0,   0,0,0,
    0,0,-math.pi/2,  0,0,-math.pi/2,   # shoulders
    0,0,0,  0,0,0,                     # elbows
    0,0,-math.pi/2,  0,0,-math.pi/2,   # hips
    0,0,0,  0,0,0,                     # knees
    0,0, math.pi/2,  0,0, math.pi/2,   # ankles
], dtype=torch.float32)

# a full 360 flip looks identical to a loop, so flips must force this on
FORCE_ACYCLIC = False
# true only where the reference deliberately grounds the torso (roll, getup)
CONTACT_SKILL = False


# quaternion helpers
def qmul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw*bw - ax*bx - ay*by - az*bz,
                        aw*bx + ax*bw + ay*bz - az*by,
                        aw*by - ax*bz + ay*bw + az*bx,
                        aw*bz + ax*by - ay*bx + az*bw], dim=-1)
def qconj(q):
    return q * torch.tensor([1., -1, -1, -1])
def qrot(q, v):
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return qmul(qmul(q, qv), qconj(q))[..., 1:]


# policy and value nets
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, action_dim)
        )
        # zero net output means standing, not identity joints
        nn.init.uniform_(self.actor_net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.actor_net[-1].bias)
        self.register_buffer("action_offset", REST.clone())
        self.register_buffer("log_std", torch.full((action_dim,), math.log(0.157)))

        # obs normalisation, as buffers so it survives a resume
        self.register_buffer("obs_mean",  torch.zeros(state_dim))
        self.register_buffer("obs_var",   torch.ones(state_dim))
        self.register_buffer("obs_count", torch.tensor(1e-4))

        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self, s):
        s_in = torch.clamp((s - self.obs_mean) / (self.obs_var.sqrt() + 1e-4), -CLIP, CLIP)
        body = s_in[:, 2:RAW_STATE_DIM].view(-1, NUM_LINKS, 14)
        body[..., 13] = 0.0

        mean = self.actor_net(s_in) + self.action_offset
        dist = Normal(mean, torch.exp(self.log_std))
        return dist, self.critic_net(s_in)

model = ActorCritic()
opt = optim.Adam([
    {'params': model.actor_net.parameters(),  'lr': 1e-4},
    {'params': model.critic_net.parameters(), 'lr': 1e-3},
])


# rollout storage and rewards
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.fells = []
        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []
        self.term_sum = torch.zeros(5)
        self.term_n = 0
        self.last_root_err = torch.zeros(NUM_ENVS)

    def store(self, state, action, reward, done, fell, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done.float())
        self.fells.append(fell.float())
        self.log_probs.append(log_prob)
        self.values.append(value.squeeze(-1))

    def compute_gae(self, gamma=0.95, lam=0.95):
        avg_reward = torch.stack(self.rewards).mean().item()
        with torch.no_grad():
            _, last_value = model(state_batch)
        values = self.values + [last_value.squeeze(-1)]

        gae = torch.zeros(NUM_ENVS)
        adv = []
        for t in reversed(range(len(self.rewards))):
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * values[t + 1] * (1.0 - self.fells[t]) - values[t]
            gae = delta + gamma * lam * mask * gae
            adv.insert(0, gae.clone())

        self.advantages = adv
        self.returns = [a + v for a, v in zip(adv, self.values)]
        t = self.term_sum / max(self.term_n, 1)
        print(f"\n[{iteration}] reward: {avg_reward:.3f} | advantages mean: {torch.stack(adv).mean().item():.3f}"
              f" | pose {t[0]:.3f} vel {t[1]:.3f} end {t[2]:.3f} root {t[3]:.3f} com {t[4]:.3f}")
        return avg_reward

    def getReward(self, state):
        i = ref.idx(state[:, 0])
        body = state[:, 2:RAW_STATE_DIM].view(-1, NUM_LINKS, 14)
        pos, quat, vel, ang = body[..., 0:3], body[..., 3:7], body[..., 7:10], body[..., 10:13]

        # pose: per-joint plus root rotation error
        q_sim = qmul(qconj(quat[:, PARENT]), quat[:, CHILD])
        dq = qmul(qconj(q_sim), ref.joint_q[i])
        pose_err = (JOINT_W * (2.0 * torch.acos(dq[..., 0].abs().clamp(max=1.0))).pow(2)).sum(-1)

        dq_root = qmul(qconj(quat[:, ROOT]), ref.root_q[i])
        root_rot_err = (2.0 * torch.acos(dq_root[..., 0].abs().clamp(max=1.0))).pow(2)
        self.last_root_err = root_rot_err.detach()
        pose_err = pose_err + root_rot_err
        r_pose = torch.exp(-(2.0 / 15.0 * NUM_JOINTS) * pose_err)

        # velocity: per-joint angular velocity error
        ang_rel_sim = qrot(qconj(quat[:, PARENT]), ang[:, CHILD] - ang[:, PARENT])
        vel_err = (JOINT_W * (ang_rel_sim - ref.joint_av[i]).pow(2).sum(-1)).sum(-1)
        r_vel = torch.exp(-VEL_SCALE * vel_err)

        # end effectors: hand and foot position
        de = pos[:, END_EFFECTORS] - ref.link_p[i][:, END_EFFECTORS]
        r_end = torch.exp(-10.0 * de.pow(2).sum(-1).sum(-1))

        # centre of mass position
        com = (pos * LINK_MASS.view(1, -1, 1)).sum(1) / LINK_MASS.sum()
        r_com = torch.exp(-10.0 * (com - ref.com[i]).pow(2).sum(-1))

        # root: position, rotation, linear and angular velocity together
        root_err = ((pos[:, ROOT] - ref.link_p[i][:, ROOT]).pow(2).sum(-1)
                    + 0.1   * root_rot_err
                    + 0.01  * (vel[:, ROOT] - ref.root_vel[i]).pow(2).sum(-1)
                    + 0.001 * (ang[:, ROOT] - ref.root_ang_vel[i]).pow(2).sum(-1))
        r_root = torch.exp(-5.0 * root_err)

        self.term_sum += torch.tensor([r_pose.mean(), r_vel.mean(), r_end.mean(),
                                       r_root.mean(), r_com.mean()])
        self.term_n += 1
        # DeepMimic weights .5/.05/.15/.2/.1, renormalised to sum 1
        return 0.556 * r_pose + 0.056 * r_vel + 0.167 * r_end + 0.222 * r_root + 0.111 * r_com

    def isDone(self, state):
        # raw contact signal; the loop decides how many frames of it end an episode
        body = state[:, 2:RAW_STATE_DIM].view(-1, NUM_LINKS, 14)
        return (body[:, NON_END_LINKS, 13] > 0.5).any(dim=-1)

    def clear(self):
        self.__init__()

buffer = RolloutBuffer()


# reference motion
class Reference:
    def __init__(self, state):
        if os.path.exists("motion.npz"):
            d = np.load("motion.npz")
            self.joint_q  = torch.tensor(d["joint_q"],  dtype=torch.float32)
            self.link_p   = torch.tensor(d["link_p"],   dtype=torch.float32)
            self.joint_av = torch.tensor(d["joint_av"], dtype=torch.float32)
            self.com      = torch.tensor(d["com"],      dtype=torch.float32)
            self.root_vel     = torch.tensor(d["root_vel"],     dtype=torch.float32)
            self.root_ang_vel = torch.tensor(d["root_ang_vel"], dtype=torch.float32)
            self.root_q       = torch.tensor(d["root_q"],       dtype=torch.float32)
            print(f"Loaded reference motion: {self.joint_q.shape[0]} frames")
        else:
            body = state[0, 2:RAW_STATE_DIM].view(NUM_LINKS, 14)
            quat = body[:, 3:7]
            self.joint_q  = qmul(qconj(quat[PARENT]), quat[CHILD]).unsqueeze(0)
            self.link_p   = body[:, 0:3].unsqueeze(0)
            self.joint_av = torch.zeros(1, NUM_JOINTS, 3)
            self.com      = (self.link_p * LINK_MASS.view(1, -1, 1)).sum(1) / LINK_MASS.sum()
            self.root_vel     = torch.zeros(1, 3)
            self.root_ang_vel = torch.zeros(1, 3)
            self.root_q       = quat[ROOT].unsqueeze(0)
            print("No motion.npz -> using captured standing pose as 1-frame reference")

        # match the sim's frame: link xz relative to pelvis, y absolute
        root_xz = self.link_p[:, ROOT:ROOT+1, :].clone()
        root_xz[..., 1] = 0.0
        self.link_p = self.link_p - root_xz
        self.com    = (self.link_p * LINK_MASS.view(1, -1, 1)).sum(1) / LINK_MASS.sum()

        self.F = self.joint_q.shape[0]
        self.dphase = 1.0 / self.F if self.F > 1 else 0.0

        # airborne window, used to split launch spawns from flight spawns
        airborne = torch.where(torch.minimum(self.link_p[:, 0, 1], self.link_p[:, 1, 1]) > 0.10)[0]
        if len(airborne) and self.F > 1:
            self.air_lo = airborne[0].item() / (self.F - 1)
            print(f"airborne window: frames {airborne[0].item()}..{airborne[-1].item()} "
                  f"({len(airborne)}/{self.F} frames)")
        else:
            self.air_lo = None

        # loop seam: a wide gap means phase must not wrap
        if self.F > 1:
            dy   = (self.link_p[-1, ROOT, 1] - self.link_p[0, ROOT, 1]).abs().item()
            dang = torch.rad2deg(2.0 * torch.acos(
                       (self.joint_q[-1] * self.joint_q[0]).sum(-1).abs().clamp(max=1.0))).mean().item()
            droot = torch.rad2deg(2.0 * torch.acos(
                       (self.root_q[-1] * self.root_q[0]).sum(-1).abs().clamp(max=1.0))).item()
            dang = max(dang, droot)
            self.acyclic = FORCE_ACYCLIC or dy > 0.10 or dang > 15.0
            print(f"loop seam: {dy:.3f} m / {dang:.1f} deg -> "
                  f"{'acyclic' if self.acyclic else 'cyclic (phase wraps)'}")
        else:
            self.acyclic = False

    def idx(self, phase):
        return torch.round(phase * (self.F - 1)).long().clamp(0, self.F - 1)

    def target(self, phase, action):
        return action.view(NUM_ENVS, NUM_JOINTS, 3)


# sim link
class UDP:
    def __init__(self, ip, recv_port, send_port):
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind((ip, recv_port))
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = (ip, send_port)

    def receive_state(self, num_floats=RAW_STATE_DIM * NUM_ENVS):
        data, _ = self.recv_sock.recvfrom(num_floats * 4)
        return struct.unpack(f"{num_floats}f", data)

    def send_actions(self, actions):
        self.send_sock.sendto(struct.pack(f"{len(actions)}f", *actions), self.send_addr)

    def pad(self, raw):
        return torch.cat([raw, torch.zeros(raw.shape[0], PAD_DIM)], dim=1)

    def get_state(self):
        self.send_actions([-100.0] * action_dim)
        raw = torch.tensor(self.receive_state(), dtype=torch.float32).view(NUM_ENVS, RAW_STATE_DIM)
        return self.pad(raw)

    def step(self, target):
        self.send_actions(target.flatten().tolist())
        raw = torch.tensor(self.receive_state(), dtype=torch.float32).view(NUM_ENVS, RAW_STATE_DIM)
        # clamp not wrap: the last step of an acyclic clip scores against its final frame
        raw[:, 0] = torch.clamp(phase + ref.dphase, max=1.0)
        s = self.pad(raw)
        return s, buffer.getReward(s), buffer.isDone(s)

    def send_reset(self, env_idx, ph):
        self.send_actions([-69.0, float(env_idx), float(ph)])

udp = UDP("127.0.0.1", 5006, 5005)


# ppo update
def updateModel(K_epochs):
    states        = torch.stack(buffer.states).view(-1, state_dim)
    actions       = torch.stack(buffer.actions).view(-1, action_dim)
    old_log_probs = torch.stack(buffer.log_probs).view(-1)
    returns       = torch.stack(buffer.returns).view(-1).clamp(0, 20)
    advantages    = torch.stack(buffer.advantages).view(-1)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = advantages.clamp(-5, 5)

    N_samples = states.shape[0]
    batch_size = 1024

    stop = False
    for epoch in range(K_epochs):
        if stop: break
        perm = torch.randperm(N_samples)
        for start in range(0, N_samples, batch_size):
            idx = perm[start:start + batch_size]

            dist, values = model(states[idx])
            new_log_probs = dist.log_prob(actions[idx]).sum(-1)
            entropy = dist.entropy().sum(-1)

            # trust region guard, written as not(<=) so NaN trips it too
            with torch.no_grad():
                approx_kl = (old_log_probs[idx] - new_log_probs).mean().item()
            if not (approx_kl <= 0.02):
                stop = True
                break

            ratio = torch.exp((new_log_probs - old_log_probs[idx]).clamp(-20, 20))
            clipped = torch.clamp(ratio, 0.8, 1.2)
            surr = torch.min(ratio * advantages[idx], clipped * advantages[idx])
            surr = torch.where(advantages[idx] < 0, torch.max(surr, 3.0 * advantages[idx]), surr)

            actor_loss   = -surr.mean()
            critic_loss  = (values.squeeze(-1) - returns[idx]).pow(2).mean()
            entropy_loss = -entropy.mean()
            loss = actor_loss + 0.5 * critic_loss + ENT_COEF * entropy_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

    print(f"loss {loss.item():.2f} | actor {actor_loss.item():.3f} | "
          f"critic {critic_loss.item():.3f} | std {torch.exp(model.log_std).mean().item():.3f} | "
          f"ep_len {steps_alive.mean().item():.1f} | launch {lau_succ}/{lau_done} | "
          f"rc_lo {rc_lo:.2f} | nofall {nofall:.2f} | success {success:.2f}")


# load reference and checkpoint
print("Starting training loop...")
state_batch = udp.get_state()
ref = Reference(state_batch)
phase = torch.rand(NUM_ENVS)

# True keeps weights and obs stats but resets optimiser and curriculum
NEW_SKILL = False

start_iteration = 0
resume_ckpt = None
if os.path.exists("POLICY_3D.pt"):
    ckpt = torch.load("POLICY_3D.pt")
    model.load_state_dict(ckpt['model'])
    if NEW_SKILL:
        print("Warm start: weights + obs stats kept, optimizer reset, curriculum restarted")
    else:
        opt.load_state_dict(ckpt['optimizer'])
        start_iteration = ckpt['iteration'] + 1
        resume_ckpt = ckpt
        print(f"Resuming from iteration {start_iteration}")


# episode settings
EP_LIMIT = 80
ACTION_STD = 0.15
steps_alive = torch.zeros(NUM_ENVS)
contact_frames = torch.zeros(NUM_ENVS)
# frames of non-foot contact tolerated before an episode counts as a fall
TC_GRACE = 30 if CONTACT_SKILL else 4

# reverse curriculum over the RSI start phase
RC_ENABLE   = False   # off for cyclic gaits, where every phase is reachable anyway
RC_LO_START = 0.9     # first window: last ~10% of the clip
RC_LO_MIN   = 0.0     # fully widened, same as uniform RSI
RC_STEP     = 0.03
RC_RMAX     = 0.90    # success above this widens the window
RC_RMIN     = 0.10    # success below this pulls it back
RC_ROT_TOL  = 0.30    # mean root orientation error an episode may average
RC_WINDOW   = 12      # iterations of history per decision, also the minimum per stage
RC_FRONTIER_P    = 0.3   # share of spawns biased onto the newest frames
RC_FRONTIER_BAND = 0.04
RC_LO_OVERRIDE   = 0.0   # float forces a stage on resume, None uses the checkpoint's
FREEZE_STAGE     = False

rc_lo = float(resume_ckpt['rc_lo']) if (resume_ckpt and 'rc_lo' in resume_ckpt) else RC_LO_START
if RC_LO_OVERRIDE is not None:
    print(f"curriculum OVERRIDE: {rc_lo:.2f} -> {RC_LO_OVERRIDE:.2f}")
    rc_lo = RC_LO_OVERRIDE
if not RC_ENABLE:
    rc_lo = RC_LO_MIN

rc_hist = []
nofall = success = 0.0
rot_sum = torch.zeros(NUM_ENVS)
ep_start = phase.clone()
lau_done = lau_succ = 0


def sample_phase():
    # frontier bias onto the frames the curriculum just opened, rest as rehearsal
    if RC_ENABLE and float(torch.rand(1)) < RC_FRONTIER_P:
        hi = min(1.0, rc_lo + RC_FRONTIER_BAND)
        return float(torch.empty(1).uniform_(rc_lo, hi))
    return float(torch.empty(1).uniform_(rc_lo, 1.0))

best_reward = resume_ckpt.get('best_reward', -1.0) if resume_ckpt else -1.0
rc_since    = resume_ckpt.get('rc_since', -10**9) if resume_ckpt else 0
BEST_MIN_ITERS = RC_WINDOW

phase = torch.tensor([sample_phase() for _ in range(NUM_ENVS)])
ep_start = phase.clone()
print(f"reverse curriculum: start phase in [{rc_lo:.2f}, 1.00]" if RC_ENABLE
      else "reverse curriculum: disabled (uniform RSI)")


# training loop
for iteration in range(start_iteration, N):
    model.log_std.data.fill_(math.log(ACTION_STD))
    buffer.clear()
    ep_done = ep_fell = ep_succ = 0
    lau_done = lau_succ = 0

    # rollout
    for t in range(T):
        state_batch[:, 0] = phase
        with torch.no_grad():
            dist, value = model(state_batch)
        action = torch.clamp(dist.sample(), -math.pi, math.pi)
        log_prob = dist.log_prob(action).sum(-1)

        target = ref.target(phase, action)
        next_state, reward, contact = udp.step(target.view(NUM_ENVS, action_dim))
        bad = ~torch.isfinite(next_state).all(dim=1)
        steps_alive += 1

        contact_frames = torch.where(contact, contact_frames + 1, torch.zeros_like(contact_frames))
        fell = (contact_frames > TC_GRACE) | bad

        # acyclic clip ends at the last frame rather than teleporting back to the first
        next_phase = phase + ref.dphase
        wrapped = (next_phase >= 1.0) if ref.acyclic else torch.zeros(NUM_ENVS, dtype=torch.bool)
        done = fell | wrapped | (steps_alive >= EP_LIMIT)

        # success = stayed up and tracked the reference rotation over the episode
        rot_sum += buffer.last_root_err
        rot_mean = rot_sum / steps_alive.clamp(min=1)
        won = done & ~fell & (rot_mean <= RC_ROT_TOL)
        ep_done += int(done.sum()); ep_fell += int(fell.sum()); ep_succ += int(won.sum())
        # split by where the episode started, so flight-only spawns cannot hide a bad takeoff
        if ref.air_lo is not None:
            launch = done & (ep_start < ref.air_lo)
            lau_done += int(launch.sum()); lau_succ += int((launch & won).sum())

        reward = torch.where(fell, torch.zeros_like(reward), reward)
        buffer.store(state_batch, action, reward, done, fell, log_prob, value)

        # one NaN would poison the whole gradient update
        next_state = torch.where(bad.unsqueeze(-1), state_batch, next_state)
        state_batch = next_state
        phase = (phase + ref.dphase) % 1.0

        if done.any():
            for i in range(NUM_ENVS):
                if done[i]:
                    phase[i] = sample_phase()
                    ep_start[i] = phase[i]
                    steps_alive[i] = 0
                    contact_frames[i] = 0
                    rot_sum[i] = 0.0
                    udp.send_reset(i, phase[i])
            sleep(0.005)
            fresh = udp.get_state()
            for i in range(NUM_ENVS):
                if done[i]:
                    state_batch[i] = fresh[i]

    avg_reward = buffer.compute_gae()

    # advance or retreat the curriculum window
    nofall  = 1.0 - (ep_fell / max(ep_done, 1))
    success = ep_succ / max(ep_done, 1)
    if RC_ENABLE and not FREEZE_STAGE:
        rc_hist.append(success)
        if len(rc_hist) >= RC_WINDOW:
            del rc_hist[:-RC_WINDOW]
            s = sum(rc_hist) / RC_WINDOW
            if s > RC_RMAX and rc_lo > RC_LO_MIN:
                rc_lo = max(RC_LO_MIN, rc_lo - RC_STEP)
                rc_hist.clear(); best_reward = -1.0; rc_since = iteration
                print(f"  curriculum ADVANCE -> start phase [{rc_lo:.2f}, 1.00]   (success {s:.2f})")
            elif s < RC_RMIN and rc_lo < RC_LO_START:
                rc_lo = min(RC_LO_START, rc_lo + RC_STEP)
                rc_hist.clear(); best_reward = -1.0; rc_since = iteration
                print(f"  curriculum RETREAT -> start phase [{rc_lo:.2f}, 1.00]   (success {s:.2f})")

    updateModel(K_epochs)

    # update obs stats after the epochs, so one rollout uses one set of stats
    with torch.no_grad():
        batch = torch.stack(buffer.states).view(-1, state_dim)
        n = batch.shape[0]
        d   = batch.mean(0) - model.obs_mean
        tot = model.obs_count + n
        model.obs_var.copy_((model.obs_var * model.obs_count + batch.var(0, unbiased=False) * n
                             + d.pow(2) * model.obs_count * n / tot) / tot)
        model.obs_mean.add_(d * n / tot)
        model.obs_count.copy_(tot)

    # save
    ckpt = {'iteration': iteration, 'model': model.state_dict(), 'optimizer': opt.state_dict(),
            'rc_lo': rc_lo, 'best_reward': best_reward, 'rc_since': rc_since}
    if iteration % 10 == 0 and iteration > 0:
        torch.save(ckpt, "POLICY_3D.pt")

    # best_reward resets on every curriculum move, so let the stage settle first
    if avg_reward > best_reward and (iteration - rc_since) >= BEST_MIN_ITERS:
        best_reward = avg_reward
        torch.save(ckpt, "POLICY_3D_best.pt")
