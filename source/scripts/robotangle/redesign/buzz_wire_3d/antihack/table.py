"""The anti-hack table of a tier: the strict reference (old oracle of the same tier) next to the new oracle (or its
kinematic chain when no Session run exists yet), from record.py recordings (metrics.trajectory / wand_change) or a
chain.json. usage: table.py <old metrics.json> <tier> [--rec new rec.npz] [--chain chain.json] -> markdown on stdout"""
import argparse, json
import numpy as np
from .metrics import trajectory, wand_change

KEYS = [("rows", "rows"), ("grasp_count", "grasps"), ("regrasp_count", "regrasps"), ("phases_total", "phases total"),
        ("waypoints_total", "waypoints"), ("tcp_path_m_total", "TCP path (m)"), ("joint_path_rad_total", "joint path (rad)")]


def row_from_rec(path):
    rec = np.load(path)
    tr = trajectory(rec); wc = wand_change(rec)
    return {**{k: tr[k] for k, _ in KEYS}, "wand": wc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old"); ap.add_argument("tier"); ap.add_argument("--rec"); ap.add_argument("--chain")
    a = ap.parse_args()
    old = json.load(open(a.old)).get(a.tier)
    new = row_from_rec(a.rec) if a.rec else None
    ch = json.load(open(a.chain)) if a.chain else None
    hdr = ["", "old oracle (strict reference)", "new oracle"]
    print("| " + " | ".join(hdr) + " |"); print("|---|---|---|")
    for k, lab in KEYS:
        o = old.get(k) if old else None
        n = new.get(k) if new else None
        print(f"| {lab} | {o if o is not None else '-'} | {n if n is not None else '-'} |")
    if old or new:
        f = lambda w: f"{w['translation_m']} m / {w['rotation_deg']} deg; path {w['path_m']} m, turned {w['accumulated_rotation_deg']} deg"
        print(f"| wand pose change | {f(old['wand']) if old else '-'} | {f(new['wand']) if new else '-'} |")
    if ch:
        print(f"| forced held segments (all-pairs, {len(ch['grasps_tested'])} grasps, {ch['pairs_tested']} rest pairs) | - | "
              f"{ch.get('held_segments')} (parks {ch.get('parks')}) |")


if __name__ == "__main__":
    main()
