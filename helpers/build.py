"""Build the reference clips training reads."""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dm_common as C
import ballistic
import writecsv
import checkclip
import angular
from retarget import retarget, resample

HERE = os.path.dirname(os.path.abspath(__file__))
DM_DIR = os.path.join(HERE, "clips")
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "animations"))

# skill -> (DeepMimic clip, frame window or None for the whole clip, description)
RECIPES = {
    "summersalt":   ("backflip", None, "backflip / somersault"),
    "karate_kick":  ("kick", None, "front snap kick"),
    "karate_spin":  ("spinkick", None, "spinning back kick"),
    "boxing_combo": ("punch", None, "punch combination"),
    "cartwheel":    ("cartwheel", None, "cartwheel"),
    "roll":         ("roll", None, "forward roll"),
    "jump":         ("jump", None, "standing jump"),
}


def _finish(jq, rp, rq, name, out, verbose=True, slow=1.0, rigid=True):
    """Shared tail: ground-clamp, ballistic-correct, re-clamp, write, verify."""
    if verbose:
        print(f"\n--- {name} ---")
    if slow != 1.0:
        jq, rp, rq = angular.slow_down(jq, rp, rq, slow, verbose=verbose)

    rp = ballistic.ground_clamp(rp, jq, rq)

    # 1. hang time each flight window's push-off actually pays for
    lp, lq = C.forward_kinematics(rq, rp, jq)
    jq, rp, rq = ballistic.retime_flight(jq, rp, rq, lp, lq, verbose=verbose)

    # 1b. hold the pose and spin at a constant rate, so L is conserved by construction
    if rigid:
        lp, lq = C.forward_kinematics(rq, rp, jq)
        wins = ballistic.find_flight(lp, lq)
        if wins:
            jq, rq = angular.rigid_flight(jq, rp, rq, wins, verbose=verbose)

    # 2. make each arc a true parabola, iterated to a fixed point
    for it in range(6):
        rp = ballistic.ground_clamp(rp, jq, rq)
        lp, lq = C.forward_kinematics(rq, rp, jq)
        before = ballistic.find_flight(lp, lq)
        rp, _ = ballistic.fix_translation(rp, lp, lq,
                                          verbose=(verbose and it == 0))
        rp = ballistic.ground_clamp(rp, jq, rq)
        lp, lq = C.forward_kinematics(rq, rp, jq)
        if ballistic.find_flight(lp, lq) == before:
            break

    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = writecsv.write(out, jq, rp, rq)
    print(f"  wrote {out}  ({rows.shape[0]} frames)")
    return out


def cmd_fix(args):
    raw, jq, lp, rq = checkclip.load(args.src)
    out = args.out or args.src.replace(".csv", "_fixed.csv")
    _finish(jq, lp[:, C.ROOT], rq, os.path.basename(args.src), out)
    checkclip.check(out)


def cmd_dm(args):
    src = os.path.join(DM_DIR, f"{args.clip}.txt")
    jq, rp, rq, dur, loop = retarget(src)
    jq, rp, rq = resample(jq, rp, rq, dur)
    if args.lo is not None or args.hi is not None:
        lo, hi = args.lo or 0, args.hi if args.hi is not None else len(jq)
        jq, rp, rq = jq[lo:hi], rp[lo:hi], rq[lo:hi]
    out = args.out or os.path.join(OUT_DIR, f"{args.clip}.csv")
    _finish(jq, rp, rq, args.clip, out)
    checkclip.check(out)


def cmd_all(args):
    made = []
    for name, (clip, win, desc) in RECIPES.items():
        src = os.path.join(DM_DIR, f"{clip}.txt")
        if not os.path.exists(src):
            print(f"skip {name}: {src} missing")
            continue
        jq, rp, rq, dur, loop = retarget(src)
        jq, rp, rq = resample(jq, rp, rq, dur)
        if win:
            jq, rp, rq = jq[win[0]:win[1]], rp[win[0]:win[1]], rq[win[0]:win[1]]
        out = os.path.join(OUT_DIR, f"{name}.csv")
        _finish(jq, rp, rq, f"{name}  ({desc})", out)
        made.append(out)
    print("\n================ verification ================")
    allok = True
    for m in made:
        allok &= checkclip.check(m)
    print("\nALL PASS" if allok else "\nsome clips still fail - see above")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fix");  f.add_argument("src"); f.add_argument("-o", "--out")
    f.set_defaults(fn=cmd_fix)
    d = sub.add_parser("dm");   d.add_argument("clip"); d.add_argument("-o", "--out")
    d.add_argument("--lo", type=int); d.add_argument("--hi", type=int)
    d.set_defaults(fn=cmd_dm)
    a = sub.add_parser("all");  a.set_defaults(fn=cmd_all)
    args = p.parse_args()
    args.fn(args)
