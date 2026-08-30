"""Deterministic risk descriptors for DexJoCo single-arm action chunks.

DexJoCo counterpart of `robocasa_descriptors.py`.  Everything here is *computed*
from the planned action chunk -- no VLM, no learning -- so it can be used as

  * hard rules that veto compression (`computed_risk` / `action_rule_block`), and
  * a `facts_text()` block that is handed to the VLM judge as already-resolved
    declarative English, so the VLM only has to reason about semantics.

ACTION LAYOUT (single-arm DexJoCo, 22 dims), verified against
`serve_policy_dexjoco.py` and the LeRobot metadata of the checkpoint:

    a[:,  0: 3]  action.arm_pos   end-effector position, metres, ABSOLUTE
    a[:,  3: 6]  action.arm_rot   end-effector orientation, rotation vector, ABSOLUTE
    a[:,  6:22]  action.hand      16-DoF dexterous hand joint targets, ABSOLUTE

*** ABSOLUTE, NOT DELTAS. ***  Evidence (see DEXJOCO_SETUP.md):
  1. the dataset metadata of the GR00T checkpoint marks every action group
     `"absolute": true`;
  2. `DexJoCoOpenPIEnv.stay()` holds the pose by re-sending the *current state*
     as the action -- a delta controller would hold with zeros;
  3. `DexJoCoOpenPIEnv._process_action` converts rotvec->quat and hands the raw
     pose to the opspace controller as a servo target.

Consequences for the descriptors: RoboCasa's "sum the deltas / does the sum
clip" logic is meaningless here.  The K2 risk for absolute targets is that
skipping every other target forces the controller to service TWO steps of motion
in ONE control tick -- so the analogue of RoboCasa's `clip_excess` is
`skip_excess`: the fraction of merged commands whose one-tick jump exceeds what
the operational-space controller can realistically track.

Unlike RoboCasa there is no binary gripper dim; the 16-DoF hand is fully
continuous, so "grasp / release" has to be detected as hand-joint motion.
"""

import numpy as np

# --- thresholds -------------------------------------------------------------
# Per-control-step (30 Hz) end-effector translation, metres.  A K2 merge doubles
# the commanded jump; beyond ~2x a normal transit step the opspace controller
# starts lagging its target.  Calibrated from real GR00T chunks on water_plant
# (see DEXJOCO_SETUP.md "descriptor calibration").
JUMP_LIMIT_POS = 0.030      # m  per executed control step  (~p97 of K2 merges)
JUMP_LIMIT_ROT = 0.200      # rad per executed control step (~p97 of K2 merges)
JUMP_LIMIT_HAND = 0.300     # rad, L2 over the 16 hand joints (~p98 of K2 merges)

# "the arm is barely moving" / "moving fast", metres per control step
# (observed single-step translation: p50 8 mm, p90 16 mm, p99 30 mm)
SLOW_POS = 0.004
FAST_POS = 0.015
# A direction reversal only counts when both legs carry real motion; below this
# the "reversal" is diffusion-head jitter and fires on ~95% of chunks.
REVERSAL_MIN = 0.012        # m per step (~p75 of observed single-step motion)
REVERSAL_COS = -0.5         # > 120 deg turn
# Hand joints are considered actively reconfiguring above this L2 step size
# (observed: p50 0.028, p90 0.077, p99 0.23 -- 0.03 would flag 87% of chunks).
HAND_ACTIVE = 0.100

POS = slice(0, 3)
ROT = slice(3, 6)
HAND = slice(6, 22)


def _rotvec_step_angles(rv):
    """Geodesic angle (rad) between consecutive rotation vectors, shape (T-1,)."""
    if len(rv) < 2:
        return np.zeros(1)
    try:
        from scipy.spatial.transform import Rotation as R
        r = R.from_rotvec(np.asarray(rv, dtype=float))
        rel = (r[:-1].inv() * r[1:]).as_rotvec()
        return np.linalg.norm(rel, axis=1)
    except Exception:
        # Fallback: plain rotvec difference. Fine for small steps.
        return np.linalg.norm(np.diff(np.asarray(rv, dtype=float), axis=0), axis=1)


def _merged_rot_angles(rv, k=2):
    """Geodesic angle of the jump the controller sees when block-last K-merging."""
    rv = np.asarray(rv, dtype=float)
    if len(rv) < k + 1:
        return np.zeros(1)
    keep = rv[k - 1::k]                     # the targets that survive the merge
    seq = np.concatenate([rv[:1], keep])    # prepend current commanded pose
    return _rotvec_step_angles(seq)


def descriptors(a, f=0, n=16, k=2):
    """Exact risk signals for the window a[f:f+n] of a planned chunk.

    Args:
        a: (T, 22) planned single-arm DexJoCo action chunk (absolute targets).
        f: start index of the window.
        n: window length (one chunk = 16 steps ~ 0.53 s at 30 Hz).
        k: compression factor the gate is being asked about (block-last skip).
    Returns:
        dict of floats. Boolean signals are 0.0/1.0 floats.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[None]
    w = a[f:f + n]
    if len(w) == 0:
        w = a[:1]

    pos = w[:, POS]
    rot = w[:, ROT]
    hand = w[:, HAND]

    # --- per-step motion ----------------------------------------------------
    dp = np.diff(pos, axis=0) if len(pos) > 1 else np.zeros((1, 3))
    speed = np.linalg.norm(dp, axis=1)                       # m / step
    rot_speed = _rotvec_step_angles(rot)                     # rad / step
    dh = np.diff(hand, axis=0) if len(hand) > 1 else np.zeros((1, 16))
    hand_speed = np.linalg.norm(dh, axis=1)                  # rad(L2) / step

    # --- direction reversal (sharp turn under significant motion) -----------
    reversal = 0.0
    if len(dp) >= 2:
        v1, v2 = dp[:-1], dp[1:]
        m1 = np.linalg.norm(v1, axis=1)
        m2 = np.linalg.norm(v2, axis=1)
        cos = np.sum(v1 * v2, axis=1) / np.maximum(m1 * m2, 1e-12)
        reversal = float(np.any((m1 > REVERSAL_MIN) & (m2 > REVERSAL_MIN)
                                & (cos < REVERSAL_COS)))

    # --- hand activity ------------------------------------------------------
    hand_change = float(hand_speed.max() > HAND_ACTIVE)
    hand_active_frac = float((hand_speed > HAND_ACTIVE).mean())
    # mean joint angle rising => fingers flexing (closing); falling => opening.
    hand_mean = hand.mean(axis=1)
    hand_trend = float(hand_mean[-1] - hand_mean[0]) if len(hand_mean) > 1 else 0.0

    # --- "holding something while creeping along" ---------------------------
    # hand static (already grasped) + arm slow => precise placement / insertion.
    closed_slow = float(((hand_speed <= HAND_ACTIVE) & (speed[:len(hand_speed)] < SLOW_POS)).mean() > 0.5)

    # --- decelerating to a stop --------------------------------------------
    decel = float(
        len(speed) >= 8
        and speed[-4:].mean() < SLOW_POS
        and speed[:4].mean() > speed[-4:].mean()
    )

    # --- feasibility of the K-merge (RoboCasa's clip_excess analogue) -------
    # Block-last skipping keeps every k-th target; the controller must then
    # traverse k steps of motion within one control tick.
    if len(pos) > k:
        keep_pos = pos[k - 1::k]
        seq_pos = np.concatenate([pos[:1], keep_pos], axis=0)
        merged_pos = np.linalg.norm(np.diff(seq_pos, axis=0), axis=1)
        keep_hand = hand[k - 1::k]
        seq_hand = np.concatenate([hand[:1], keep_hand], axis=0)
        merged_hand = np.linalg.norm(np.diff(seq_hand, axis=0), axis=1)
    else:
        merged_pos = speed * k
        merged_hand = hand_speed * k
    merged_rot = _merged_rot_angles(rot, k)

    skip_excess_pos = float(np.mean(merged_pos > JUMP_LIMIT_POS))
    skip_excess_rot = float(np.mean(merged_rot > JUMP_LIMIT_ROT))
    skip_excess_hand = float(np.mean(merged_hand > JUMP_LIMIT_HAND))
    skip_excess = float(max(skip_excess_pos, skip_excess_rot, skip_excess_hand))

    return {
        "speed_mean": float(speed.mean()),
        "speed_max": float(speed.max()),
        "rot_speed_mean": float(rot_speed.mean()),
        "rot_speed_max": float(rot_speed.max()),
        "hand_speed_mean": float(hand_speed.mean()),
        "hand_speed_max": float(hand_speed.max()),
        "hand_change": hand_change,
        "hand_active_frac": hand_active_frac,
        "hand_trend": hand_trend,
        "reversal": reversal,
        "closed_slow": closed_slow,
        "decel": decel,
        "skip_excess": skip_excess,
        "skip_excess_pos": skip_excess_pos,
        "skip_excess_rot": skip_excess_rot,
        "skip_excess_hand": skip_excess_hand,
        "merged_pos_max": float(merged_pos.max()),
        "merged_rot_max": float(merged_rot.max()),
        "merged_hand_max": float(merged_hand.max()),
    }


def facts_text(x):
    """Render the computed signals as declarative English.

    Same contract as robocasa_descriptors.facts_text: thresholds are already
    resolved here, so the sentences are conclusions, not measurements to be
    interpreted by the VLM.
    """
    parts = []

    if x["hand_change"]:
        verb = "closing" if x["hand_trend"] > 0 else "opening"
        parts.append(f"the dexterous hand is actively {verb} its fingers during this window")
    else:
        parts.append("the hand keeps a fixed finger configuration throughout")

    parts.append("the end-effector reverses direction sharply" if x["reversal"]
                 else "the end-effector keeps a consistent direction")

    sp = ("barely moving" if x["speed_mean"] < SLOW_POS else
          "moving at a normal pace" if x["speed_mean"] < FAST_POS else "moving fast")
    parts.append(f"the arm is {sp} "
                 f"(mean {x['speed_mean'] * 1000:.0f} mm per control step, "
                 f"peak {x['speed_max'] * 1000:.0f} mm)")

    if x["rot_speed_mean"] > 0.02:
        parts.append(f"the wrist is reorienting at {np.degrees(x['rot_speed_mean']):.0f} "
                     "degrees per control step")

    if x["closed_slow"]:
        parts.append("it is holding a fixed grasp while creeping along")
    if x["decel"]:
        parts.append("it is decelerating to a near stop")

    tail = ""
    if x["skip_excess"] > 0.05:
        tail = (" Executing only every second target would make the controller "
                f"traverse more than it can track on {x['skip_excess']:.0%} of the "
                "merged commands"
                f" (largest merged jump {x['merged_pos_max'] * 1000:.0f} mm, "
                f"{np.degrees(x['merged_rot_max']):.0f} deg).")

    return ("MEASURED FROM THE PLANNED MOTION over the next ~0.5 second (these are "
            "computed facts, not estimates): " + "; ".join(parts) + "." + tail)


def computed_risk(x):
    """Hard flags that are never delegated to the VLM."""
    return {
        "hand_transition": x["hand_change"],
        "reversal": x["reversal"],
        "precise_hold": x["closed_slow"],
        "infeasible_merge": float(x["skip_excess"] > 0.20),
    }


def action_rule_block(chunk, k=2):
    """Runtime mirror of computed_risk for the eval client.

    Args:
        chunk: (T, 22) planned single-arm action sub-chunk (absolute targets).
    Returns:
        (block: bool, reason: str) -- block=True means "do NOT compress".
    """
    a = np.asarray(chunk, dtype=float)
    if a.ndim != 2 or a.shape[1] < 22:
        return False, ""
    x = descriptors(a, 0, len(a), k=k)
    if x["hand_change"]:
        return True, "hand_transition"
    if x["closed_slow"]:
        return True, "precise_hold"
    if x["reversal"]:
        return True, "reversal"
    if x["skip_excess"] > 0.20:
        return True, "skip_excess"
    return False, ""


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    demo = np.cumsum(rng.normal(0, 0.005, size=(16, 22)), axis=0)
    d = descriptors(demo)
    print({k: round(v, 4) for k, v in d.items()})
    print(facts_text(d))
    print(computed_risk(d))
    print(action_rule_block(demo))
