import atexit
import math
import select
import socket
import struct
import sys
import termios
import tty
from time import perf_counter, sleep

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)


# settings
class Config:
    NUM_ENVS = 1
    DPHASE = 1.0 / 48
    STEP_DT = 1.0 / 30.0
    START_PHASE = 0.33      # phase the RSI preamble hands off to the policy at
    RSI_PLAYBACK = True     # replay the clip up to START_PHASE before the policy takes over
    LOOP = True
    SPEED_PLAYBACK = 1.0
    CKPT = "POLICY_3D.pt"

    NUM_LINKS, NUM_JOINTS = 14, 13
    RAW_STATE_DIM = 2 + NUM_LINKS * 14
    PAD_DIM = 9
    STATE_DIM = RAW_STATE_DIM + PAD_DIM
    ACTION_DIM = 3 * NUM_JOINTS

C = Config


# policy
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.actor_net = nn.Sequential(
            nn.Linear(C.STATE_DIM, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, C.ACTION_DIM),
        )
        self.log_std = nn.Parameter(torch.full((C.ACTION_DIM,), math.log(0.05)))

        # buffers the training checkpoint carries
        self.register_buffer("action_offset", torch.zeros(C.ACTION_DIM))
        self.register_buffer("obs_mean", torch.zeros(C.STATE_DIM))
        self.register_buffer("obs_var", torch.ones(C.STATE_DIM))
        self.register_buffer("obs_count", torch.tensor(1e-4))

        self.critic_net = nn.Sequential(
            nn.Linear(C.STATE_DIM, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
        )

    def act(self, state):
        x = torch.clamp((state - self.obs_mean) / (self.obs_var.sqrt() + 1e-4), -5.0, 5.0)
        body = x[:, 2:C.RAW_STATE_DIM].view(-1, C.NUM_LINKS, 14)
        body[..., 13] = 0.0
        return torch.clamp(self.actor_net(x) + self.action_offset, -math.pi, math.pi)

model = ActorCritic()
model.load_state_dict(torch.load(C.CKPT)["model"])
model.eval()


def pad(raw):
    return torch.cat([raw, torch.zeros(raw.shape[0], C.PAD_DIM)], dim=1)

NON_END_LINKS = [i for i in range(C.NUM_LINKS) if i not in (0, 1, 11, 12)]

def is_done(state):
    body = state[:, 2:C.RAW_STATE_DIM].view(-1, C.NUM_LINKS, 14)
    return (body[:, NON_END_LINKS, 13] > 0.5).any(dim=-1)


# sim link
class UDP:
    def __init__(self, ip="127.0.0.1", recv_port=5006, send_port=5005):
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind((ip, recv_port))
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = (ip, send_port)

    def send(self, values):
        self.send_sock.sendto(struct.pack(f"{len(values)}f", *values), self.send_addr)

    def recv(self):
        count = C.RAW_STATE_DIM * C.NUM_ENVS
        data, _ = self.recv_sock.recvfrom(count * 4)
        values = struct.unpack(f"{count}f", data)
        return torch.tensor(values, dtype=torch.float32).view(C.NUM_ENVS, C.RAW_STATE_DIM)

    def state(self):
        self.send([-100.0] * C.ACTION_DIM)
        return pad(self.recv())

    def step(self, target):
        self.send(target.flatten().tolist())
        return pad(self.recv())

    def reset(self, env, phase):
        self.send([-69.0, float(env), float(phase)])


# reference clip
class Motion:
    MASS = np.array([1.0, 1.0, 2.5, 2.5, 4.5, 4.5, 5.0, 7.0, 8.0, 1.8, 1.8, 1.2, 1.2, 3.0])

    def __init__(self, path="baked_motion.csv"):
        self.data = np.loadtxt(path, delimiter=",")
        self.frames = self.data.shape[0]

    def frame(self, phase):
        return min(int(round(phase * (self.frames - 1))), self.frames - 1)

    def com(self, state):
        body = state[0, 2:C.RAW_STATE_DIM].view(C.NUM_LINKS, 14).numpy()
        total = self.MASS.sum()
        pos = (body[:, :3] * self.MASS[:, None]).sum(0) / total
        vel = (body[:, 7:10] * self.MASS[:, None]).sum(0) / total
        return pos, vel

    def ref_com(self, frame):
        pos = self.data[min(frame, self.frames - 1), 52:94].reshape(C.NUM_LINKS, 3)
        return (pos * self.MASS[:, None]).sum(0) / self.MASS.sum()


# raw terminal input
class Keyboard:
    def __init__(self):
        self.enabled = False
        self.fd = sys.stdin.fileno()
        try:
            self.old_term = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            atexit.register(
                lambda: termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_term))
            self.enabled = True
        except termios.error:
            pass

    def poll(self):
        if not self.enabled:
            return None
        return sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else None


# playback runner
class Runner:
    def __init__(self):
        self.udp = UDP()
        self.motion = Motion()
        self.keyboard = Keyboard()
        self.phase = torch.full((C.NUM_ENVS,), C.START_PHASE)
        self.paused = False

    def reset(self):
        for i in range(C.NUM_ENVS):
            self.udp.reset(i, self.phase[i])
        sleep(0.005)
        return self.udp.state()

    def playback_to_start(self):
        # Kinematic replay, not physics. Every frame uses the same -69 reset that hands off
        # at START_PHASE, so the handoff frame is reached exactly like all the others and
        # there is no visible snap where the policy takes over.
        p = 0.0
        while p < C.START_PHASE:
            for i in range(C.NUM_ENVS):
                self.udp.reset(i, p)
            sleep(C.STEP_DT / C.SPEED_PLAYBACK)
            p += C.DPHASE
        self.phase[:] = C.START_PHASE
        return self.reset()

    def start(self):
        if C.RSI_PLAYBACK:
            return self.playback_to_start()
        self.phase[:] = C.START_PHASE
        return self.reset()

    def show_snapshot(self, state):
        frame = self.motion.frame(C.START_PHASE)
        pos, vel = self.motion.com(state)
        ref_pos = self.motion.ref_com(frame)
        ref_vel = (self.motion.ref_com(min(frame + 1, self.motion.frames - 1))
                   - self.motion.ref_com(max(frame - 1, 0))) / (2 * C.STEP_DT)
        grounded = state[0, 2:C.RAW_STATE_DIM].view(C.NUM_LINKS, 14)[:, 13]

        print(
            f"[R] snapped to phase {C.START_PHASE:.2f} = frame {frame}"
            "  -  frozen, press r to run from there\n"
            f"     com pos  sim [{pos[0]:+7.3f} {pos[1]:+7.3f} {pos[2]:+7.3f}]"
            f"   ref [{ref_pos[0]:+7.3f} {ref_pos[1]:+7.3f} {ref_pos[2]:+7.3f}]\n"
            f"     com vel  sim [{vel[0]:+7.3f} {vel[1]:+7.3f} {vel[2]:+7.3f}]"
            f"   ref [{ref_vel[0]:+7.3f} {ref_vel[1]:+7.3f} {ref_vel[2]:+7.3f}]\n"
            f"     err      pos {np.linalg.norm(pos - ref_pos)*100:5.1f} cm"
            f"    vel {np.linalg.norm(vel - ref_vel):5.2f} m/s\n"
            f"     feet     R {'DOWN' if grounded[0] > 0.5 else 'up'}"
            f"  L {'DOWN' if grounded[1] > 0.5 else 'up'}",
            flush=True,
        )

    def handle_key(self, key, state):
        if key == "q":
            return False, state
        if key != "r":
            return True, state

        if self.paused:
            self.paused = False
            self.next_tick = perf_counter() + C.STEP_DT
            print("[R] running", flush=True)
            return True, state

        self.phase[:] = C.START_PHASE
        state = self.reset()
        self.paused = True
        self.show_snapshot(state)
        return True, state

    def run(self):
        print("Running policy (no reference, no training)...")
        state = self.udp.state()
        state = self.start()
        self.next_tick = perf_counter() + C.STEP_DT / C.SPEED_PLAYBACK
        print(f"[R] press r to snap to phase {C.START_PHASE:.2f}, r again to run from there")

        while True:
            key = self.keyboard.poll()
            running, state = self.handle_key(key, state)
            if not running:
                break
            if self.paused:
                sleep(0.02)
                continue

            state[:, 0] = self.phase
            with torch.no_grad():
                target = model.act(state)

            print(f"phase {float(self.phase[0]):.3f}  "
                  f"vy {state[0, 2:C.RAW_STATE_DIM].view(C.NUM_LINKS, 14)[6, 8].item():+.2f}",
                  flush=True)

            state = self.udp.step(target)
            finished = (self.phase + C.DPHASE) >= 1.0
            self.phase = self.phase + C.DPHASE

            if C.LOOP and finished.any():
                # Replay the preamble on every loop. A bare reset here would teleport from
                # wherever the policy left the body straight to the reference pose, which is
                # its own snap once per cycle.
                state = self.start()
                self.next_tick = perf_counter() + C.STEP_DT / C.SPEED_PLAYBACK

            now = perf_counter()
            if self.next_tick > now:
                sleep(self.next_tick - now)
                self.next_tick += C.STEP_DT / C.SPEED_PLAYBACK
            else:
                self.next_tick = now + C.STEP_DT / C.SPEED_PLAYBACK

            done = (~torch.isfinite(state).all(dim=1)
                    | (not C.LOOP) & (is_done(state) | finished))
            if done.any():
                print("clip finished" if finished[0] else "FELL")
                self.phase[:] = C.START_PHASE


if __name__ == "__main__":
    Runner().run()
