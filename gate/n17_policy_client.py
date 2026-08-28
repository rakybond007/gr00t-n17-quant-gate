"""Standalone GR00T-N1.7 policy client (wire protocol only — no `gr00t` import).

WHY THIS FILE EXISTS
--------------------
The RoboCasa eval client runs in conda env ``quant_gate_eval``, which is pinned to
numpy 1.23.5 / transformers 4.51.3 (RoboCasa needs numpy 1.23.x).  GR00T-N1.7 needs
transformers 4.57.3, supplied as a PYTHONPATH overlay (``pylibs/tf4573``) that is put
on the *server* process only.  Therefore the client may not import the N1.7 ``gr00t``
package at all; it talks to ``gr00t.policy.server_client.PolicyServer`` purely over
the wire, using only ``zmq``, ``msgpack`` and ``numpy``.

WIRE PROTOCOL (mirrors Isaac-GR00T-n17/gr00t/policy/server_client.py)
--------------------------------------------------------------------
* ZeroMQ REQ/REP; server binds ``tcp://{host}:{port}``.
* Request  = msgpack of ``{"endpoint": <name>, "data": {...}, "api_token": <opt>}``.
  ``endpoint`` defaults to ``get_action`` server-side.  NOTE the N1.7 server calls
  ``handler(**data)`` (keyword expansion) — unlike N1.5 which passed ``data``
  positionally.  So ``get_action`` must be sent as
  ``{"observation": ..., "options": ...}``.
* Response = msgpack of the handler's return value.  ``get_action`` returns the
  2-tuple ``(action, info)``, which msgpack flattens to a 2-element list.
* Custom types:
    - ndarray       -> {"__ndarray_class__": True, "as_npy": <np.save bytes>}
    - ModalityConfig-> {"__ModalityConfig_class__": True, "as_json": {...}}
  Decoding uses ``np.load(BytesIO(...), allow_pickle=False)``.  We decode
  ModalityConfig into a plain dict (we cannot import the dataclass).
* Endpoints: ``ping``, ``kill``, ``get_action``, ``reset``, ``get_modality_config``.
  ``ping`` / ``kill`` / ``get_modality_config`` take no input.

OBSERVATION KEYS: what RoboCasa produces vs. what the N1.7 server wants
----------------------------------------------------------------------
Our client's env stack is ``RoboCasaWrapper -> RecordVideo -> MultiStepWrapper``.
Its observation dict is FLAT and *unbatched*:

    video.left_view / video.right_view / video.wrist_view   uint8  (T=1, H, W, 3)
    state.<name>  (many, incl. unused ones)                 float64(T=1, D)
    annotation.human.action.task_description                ["<instruction>"]

The N1.7 server is started with ``--use-sim-policy-wrapper``, so
``Gr00tSimPolicyWrapper`` accepts exactly the same FLAT key names
(``video.<key>`` / ``state.<key>``, language under its modality key) — i.e. **no key
renaming is required**; the RoboCasa names already match the modality config in
``vlm_gate/n17/robocasa_modality_config.py``:

    video: left_view, right_view, wrist_view
    state: end_effector_position_relative, end_effector_rotation_relative,
           gripper_qpos, base_position, base_rotation
    lang : annotation.human.action.task_description

What DOES differ from N1.5, and is fixed here:

  1. BATCH DIM.  N1.7 ``check_observation`` asserts video ``ndim == 5`` (B,T,H,W,C)
     and state ``ndim == 3`` (B,T,D).  The RoboCasa obs has no batch axis, so we
     prepend B=1.  (N1.5's policy batched internally.)
  2. DTYPES.  N1.7 asserts ``uint8`` for video and ``float32`` for state; RoboCasa
     hands out float64 state.  We cast.
  3. LANGUAGE.  The sim wrapper wants a list/tuple of ``B`` plain strings.  RoboCasa
     already gives ``[instruction]`` (B=1); we normalise nested/ndarray forms to a
     flat list of ``str``.  (The wrapper's ``task`` ->
     ``annotation.human.coarse_action`` legacy patch does not apply to us, since our
     language modality key IS ``annotation.human.action.task_description``.)
  4. UNUSED KEYS.  RoboCasa emits many extra state keys (joint_position, ...).  We
     send only the keys the server's modality config asks for (fetched once at
     connect time) — smaller payloads, and it avoids surprises if the server ever
     validates strictly.  If the modality config cannot be fetched we fall back to
     sending every ``video.*``/``state.*`` key.
  5. ACTION SHAPE.  The sim wrapper returns ``{"action.<key>": (B, 16, D) float32}``.
     N1.5 returned ``(16, D)``.  We squeeze the leading B=1 so the compression
     client's ``collect_chunk`` sees the exact N1.5 shape.  Action key names are
     identical to the RoboCasa action space
     (end_effector_position / end_effector_rotation / gripper_close / base_motion /
     control_mode), so nothing is renamed.  Action layout inside the 12-dim command
     is 0:4 base_motion, 4:5 control_mode, 5:8 ee pos, 8:11 ee rot, 11:12 gripper —
     but that packing is done by the env wrapper, not here.

  NOTE on ``--judge-url internal``: the N1.7 checkpoints currently carry no
  ``_gate_prob`` output, and ``Gr00tSimPolicyWrapper`` drops non-``action.*`` keys
  anyway.  We therefore merge the ``info`` dict of the ``(action, info)`` reply into
  the returned dict (any non-``action.``-prefixed scalars survive), so an internal
  gate can be plumbed through ``info`` later without touching the client again.
"""

import argparse
import io
import sys

import msgpack
import numpy as np
import zmq

__all__ = ["N17PolicyClient", "MsgSerializer"]


# --------------------------------------------------------------------------- #
# serialization (byte-compatible with gr00t.policy.server_client.MsgSerializer)
# --------------------------------------------------------------------------- #
class MsgSerializer:
    """msgpack (+ numpy) codec matching the N1.7 server, with no gr00t import."""

    @staticmethod
    def to_bytes(data):
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data):
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            # np.save refuses some views; ascontiguousarray also undoes the negative
            # strides left by the client's np.flip on the camera views.
            np.save(output, np.ascontiguousarray(obj), allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        if isinstance(obj, np.generic):  # np scalar -> python scalar
            return obj.item()
        return obj

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            # We cannot construct gr00t's ModalityConfig here; a plain dict carries
            # the same information (delta_indices / modality_keys).
            return obj["as_json"]
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #
_VIDEO_PREFIX = "video."
_STATE_PREFIX = "state."


class N17PolicyClient:
    """Drop-in replacement for N1.5's ``SimulationInferenceClient``.

    Surface used by ``robocasa_service_compress.py``:
        ``get_action(obs) -> {"action.<key>": np.ndarray (T, D), ...}``
        ``get_modality_config() -> dict``
    plus ``ping()`` / ``reset()`` / ``kill_server()`` for parity.
    """

    def __init__(
        self,
        host="localhost",
        port=5555,
        timeout_ms=120000,
        api_token=None,
        language_key="annotation.human.action.task_description",
        squeeze_batch=True,
    ):
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.api_token = api_token
        self.language_key = language_key
        self.squeeze_batch = bool(squeeze_batch)
        self.context = zmq.Context()
        self.socket = None
        self._init_socket()
        self._modality_config = None  # lazily fetched, used to filter obs keys

    # ---------------- transport ----------------
    def _init_socket(self):
        if self.socket is not None:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect("tcp://{}:{}".format(self.host, self.port))

    def call_endpoint(self, endpoint, data=None, requires_input=True):
        request = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data if data is not None else {}
        if self.api_token:
            request["api_token"] = self.api_token
        try:
            self.socket.send(MsgSerializer.to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again:
            # REQ socket is stuck waiting for a reply that will never come.
            self._init_socket()
            raise
        if message == b"ERROR":
            raise RuntimeError("Server error (N1.5-style ERROR frame). Wrong server?")
        response = MsgSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError("Server error: {}".format(response["error"]))
        return response

    def ping(self):
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except (zmq.error.ZMQError, zmq.error.Again):
            self._init_socket()
            return False

    def kill_server(self):
        return self.call_endpoint("kill", requires_input=False)

    def reset(self, options=None):
        return self.call_endpoint("reset", {"options": options})

    def get_modality_config(self):
        if self._modality_config is None:
            self._modality_config = self.call_endpoint(
                "get_modality_config", requires_input=False
            )
        return self._modality_config

    # ---------------- observation / action adaptation ----------------
    def _wanted_keys(self):
        """(video_keys, state_keys) as flat 'video.x'/'state.x' names, or (None, None)."""
        try:
            cfg = self.get_modality_config()
        except Exception:
            return None, None
        if not isinstance(cfg, dict):
            return None, None

        def keys_of(mod):
            entry = cfg.get(mod)
            if isinstance(entry, dict):
                mk = entry.get("modality_keys")
                if isinstance(mk, (list, tuple)):
                    return [str(k) for k in mk]
            return None

        v, s = keys_of("video"), keys_of("state")
        if v is None or s is None:
            return None, None
        return ([_VIDEO_PREFIX + k for k in v], [_STATE_PREFIX + k for k in s])

    @staticmethod
    def _batched(arr, dtype, ndim):
        """Cast to ``dtype`` and prepend a batch axis until ``arr.ndim == ndim``."""
        a = np.asarray(arr)
        if a.dtype != dtype:
            a = a.astype(dtype, copy=False)
        while a.ndim < ndim:
            a = a[None, ...]
        if a.ndim != ndim:
            raise ValueError(
                "expected {}-D after batching, got shape {}".format(ndim, a.shape)
            )
        return np.ascontiguousarray(a)

    @staticmethod
    def _flatten_language(value):
        """Anything nested -> flat list[str] of length B (B=1 for our eval)."""
        if isinstance(value, str):
            return [value]
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            out = []
            for item in value:
                out.extend(N17PolicyClient._flatten_language(item))
            return out or [""]
        return [str(value)]

    def build_observation(self, observations):
        """RoboCasa flat/unbatched obs -> N1.7 sim-wrapper flat/batched obs."""
        want_v, want_s = self._wanted_keys()
        obs = {}
        for key, value in observations.items():
            if key.startswith(_VIDEO_PREFIX):
                if want_v is not None and key not in want_v:
                    continue
                obs[key] = self._batched(value, np.uint8, 5)  # (B, T, H, W, C)
            elif key.startswith(_STATE_PREFIX):
                if want_s is not None and key not in want_s:
                    continue
                obs[key] = self._batched(value, np.float32, 3)  # (B, T, D)
        lang = observations.get(self.language_key)
        if lang is None:
            raise KeyError(
                "observation is missing language key {!r}".format(self.language_key)
            )
        obs[self.language_key] = self._flatten_language(lang)
        if want_v is not None:
            missing = [k for k in want_v + want_s if k not in obs]
            if missing:
                raise KeyError(
                    "observation is missing server-required keys: {}".format(missing)
                )
        return obs

    def _unbatch_action(self, arr):
        a = np.asarray(arr)
        if self.squeeze_batch and a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        return a

    def get_action(self, observations, options=None):
        """Return the N1.5-shaped flat action dict ``{"action.<key>": (T, D)}``."""
        response = self.call_endpoint(
            "get_action",
            {"observation": self.build_observation(observations), "options": options},
        )
        # server returns a (action, info) tuple -> msgpack list of 2
        info = {}
        if isinstance(response, (list, tuple)) and len(response) == 2:
            action, info = response[0], response[1]
        else:
            action = response
        if not isinstance(action, dict):
            raise RuntimeError("unexpected get_action reply: {!r}".format(type(action)))

        out = {}
        for key, value in action.items():
            out[key] = self._unbatch_action(value)
        # Let any extra scalars (e.g. a future "_gate_prob") through untouched.
        if isinstance(info, dict):
            for key, value in info.items():
                if key not in out:
                    out[key] = value
        return out

    def close(self):
        try:
            if self.socket is not None:
                self.socket.close(linger=0)
        except Exception:
            pass

    def __del__(self):
        self.close()


# --------------------------------------------------------------------------- #
# self test (no server, no GPU)
# --------------------------------------------------------------------------- #
# Golden bytes produced by the *server's* codec (msgpack + np.save), so a change in
# either library that would break the wire is caught here rather than in a job.
_GOLD_PING = bytes.fromhex("81a8656e64706f696e74a470696e67")
_GOLD_NDARRAY = bytes.fromhex(
    "82b15f5f6e6461727261795f636c6173735f5fc3a661735f6e7079c498934e554d5059"
    "010076007b276465736372273a20273c6634272c2027666f727472616e5f6f72646572"
    "273a2046616c73652c20277368617065273a2028312c20322c2033292c207d20202020"
    "2020202020202020202020202020202020202020202020202020202020202020202020"
    "202020202020202020202020202020200a000000000000803f00000040000040400000"
    "80400000a040"
)


def _selftest():
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print("[{}] {}{}".format("ok " if cond else "FAIL", label,
                                 "" if cond else "  <- " + str(detail)))
        ok = ok and bool(cond)

    # 1. request framing matches the server's expectation, byte for byte
    got = MsgSerializer.to_bytes({"endpoint": "ping"})
    check("ping request bytes", got == _GOLD_PING, got.hex())

    # 2. ndarray encoding matches the golden blob, byte for byte
    a = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    got = MsgSerializer.to_bytes(a)
    check("ndarray encode bytes", got == _GOLD_NDARRAY, got.hex())

    # 3. decode the golden blob back into the same array
    back = MsgSerializer.from_bytes(_GOLD_NDARRAY)
    check(
        "ndarray decode golden",
        isinstance(back, np.ndarray)
        and back.dtype == np.float32
        and back.shape == (1, 2, 3)
        and np.array_equal(back, a),
    )

    # 4. full round trip of a realistic get_action reply
    reply = [
        {
            "action.end_effector_position": np.zeros((1, 16, 3), np.float32),
            "action.end_effector_rotation": np.zeros((1, 16, 3), np.float32),
            "action.gripper_close": np.ones((1, 16, 1), np.float32),
            "action.base_motion": np.zeros((1, 16, 4), np.float32),
            "action.control_mode": np.zeros((1, 16, 1), np.float32),
        },
        {},
    ]
    rt = MsgSerializer.from_bytes(MsgSerializer.to_bytes(reply))
    check(
        "get_action reply round trip",
        isinstance(rt, list)
        and len(rt) == 2
        and sorted(rt[0]) == sorted(reply[0])
        and all(rt[0][k].shape == reply[0][k].shape for k in reply[0]),
    )

    # 5. ModalityConfig marker decodes to a plain dict (no gr00t import)
    mc = MsgSerializer.from_bytes(
        MsgSerializer.to_bytes(
            {
                "video": {
                    "__ModalityConfig_class__": True,
                    "as_json": {
                        "delta_indices": [0],
                        "modality_keys": ["left_view", "right_view", "wrist_view"],
                    },
                }
            }
        )
    )
    check(
        "ModalityConfig decode",
        mc["video"]["modality_keys"] == ["left_view", "right_view", "wrist_view"],
        mc,
    )

    # 6. observation adaptation: RoboCasa shapes/dtypes -> N1.7 shapes/dtypes
    client = N17PolicyClient.__new__(N17PolicyClient)  # no socket, no server
    client.language_key = "annotation.human.action.task_description"
    client.squeeze_batch = True
    client._modality_config = {
        "video": {"delta_indices": [0],
                  "modality_keys": ["left_view", "right_view", "wrist_view"]},
        "state": {"delta_indices": [0],
                  "modality_keys": ["end_effector_position_relative",
                                    "end_effector_rotation_relative",
                                    "gripper_qpos", "base_position",
                                    "base_rotation"]},
    }
    raw_obs = {
        "video.left_view": np.zeros((1, 8, 8, 3), np.uint8),
        # a flipped view, as the eval client produces (negative strides)
        "video.right_view": np.flip(np.zeros((1, 8, 8, 3), np.uint8), axis=1),
        "video.wrist_view": np.zeros((1, 8, 8, 3), np.uint8),
        "state.end_effector_position_relative": np.zeros((1, 3), np.float64),
        "state.end_effector_rotation_relative": np.zeros((1, 4), np.float64),
        "state.gripper_qpos": np.zeros((1, 2), np.float64),
        "state.base_position": np.zeros((1, 3), np.float64),
        "state.base_rotation": np.zeros((1, 4), np.float64),
        "state.joint_position": np.zeros((1, 7), np.float64),  # extra: dropped
        "annotation.human.action.task_description": ["pick up the mug"],
    }
    obs = client.build_observation(raw_obs)
    check("video batched to (B,T,H,W,C) uint8",
          obs["video.left_view"].shape == (1, 1, 8, 8, 3)
          and obs["video.left_view"].dtype == np.uint8,
          obs["video.left_view"].shape)
    check("flipped view is contiguous after encode",
          MsgSerializer.from_bytes(
              MsgSerializer.to_bytes(obs["video.right_view"])).shape
          == (1, 1, 8, 8, 3))
    check("state batched to (B,T,D) float32",
          obs["state.gripper_qpos"].shape == (1, 1, 2)
          and obs["state.gripper_qpos"].dtype == np.float32,
          obs["state.gripper_qpos"].shape)
    check("unconfigured state key dropped", "state.joint_position" not in obs)
    check("language is list[str] of length B",
          obs["annotation.human.action.task_description"] == ["pick up the mug"])

    # 7. action unbatching gives N1.5's (T, D)
    check("action unbatched to (T,D)",
          client._unbatch_action(np.zeros((1, 16, 3), np.float32)).shape == (16, 3))

    print("\nSELFTEST {}".format("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--selftest", action="store_true",
                   help="round-trip the serializer against golden bytes; no server needed")
    p.add_argument("--ping", action="store_true", help="ping a running N1.7 server")
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--timeout-ms", type=int, default=10000)
    args = p.parse_args()

    if args.selftest:
        return _selftest()
    if args.ping:
        c = N17PolicyClient(host=args.host, port=args.port, timeout_ms=args.timeout_ms)
        alive = c.ping()
        print("ping {}:{} -> {}".format(args.host, args.port, alive))
        if alive:
            print("modality config keys:", list(c.get_modality_config().keys()))
        return 0 if alive else 1
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
