"""
artificial_eye_api.py  --  the control layer for ArtificialEye.blend

Everything physical in the rig (eyeball rotation, eyelids, pupil, muscle stretch, muscle colour) is driven
by 6 custom properties on the empty object "EYE_CONTROL", using native Blender drivers. So the rig moves
whether or not any Python runs. This file is the convenient way to SET those properties from Python:

    eye = ArtificialEye()
    eye.set_gaze(15, -5)          # degrees: +x = right side of the rendered image, +y = up
    eye.set_blink(1.0)            # 0 open .. 1 closed
    eye.key(frame=10)             # keyframe it

It provides:
  * ArtificialEye.apply / set_* / key        drive the rig from any script
  * load_csv / load_job                      replay eye_twin output (labels.csv or a blender job.json)
  * run_script(path)                         run YOUR Python file with `eye` ready to use
  * start_live_link(port)                    accept JSON lines from an external Python process (eye_link_client.py)
  * a sidebar panel (3D viewport, press N -> "Artificial Eye" tab)

Inside Blender: open this text block and press "Run Script" once (or allow auto-run) to get the panel.
Headless:  blender -b ArtificialEye.blend -P your_script.py     (your script does:  from artificial_eye_api import ArtificialEye
           after adding this folder to sys.path, or uses bpy.data.texts["artificial_eye_api.py"].as_module())
"""
import csv
import json
import math
import os
import runpy
import socket
import sys

try:
    import bpy
except ImportError:  # lets you syntax-check this file outside Blender
    bpy = None

CTRL_NAME = "EYE_CONTROL"

# name -> (default, min, max).  These are the custom properties on EYE_CONTROL.
PROPS = {
    "gaze_x_deg": (0.0, -50.0, 50.0),    # horizontal gaze; + = right side of the rendered (front-camera) image
    "gaze_y_deg": (0.0, -40.0, 40.0),    # vertical gaze;   + = up
    "vergence_deg": (0.0, 0.0, 40.0),    # total convergence of the two eyes (0 = looking far away)
    "blink": (0.0, 0.0, 1.0),            # eyelid closure, 0 open .. 1 closed
    "pupil_mm": (4.0, 1.5, 9.0),         # pupil diameter in millimetres
    "show_lids": (1.0, 0.0, 1.0),        # 1 = eyelids visible, 0 = hide them to see the muscles
}
# Accepted alternative key names (eye_twin labels, gaze_core-style names...).
ALIASES = {
    "gx_deg": "gaze_x_deg", "gy_deg": "gaze_y_deg", "yaw_deg": "gaze_x_deg", "pitch_deg": "gaze_y_deg",
    "eyelid_closure": "blink", "pupil": "pupil_mm", "vergence": "vergence_deg",
}
ANIMATED = ("gaze_x_deg", "gaze_y_deg", "vergence_deg", "blink", "pupil_mm")


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class ArtificialEye:
    """Thin wrapper around the EYE_CONTROL empty."""

    def __init__(self, ctrl=None):
        if bpy is None:
            raise RuntimeError("ArtificialEye must run inside Blender")
        self.ctrl = ctrl or bpy.data.objects.get(CTRL_NAME)
        if self.ctrl is None:
            raise RuntimeError(f"Object '{CTRL_NAME}' not found. Open ArtificialEye.blend first.")
        self._live = None

    # ------------------------------------------------------------------ setting values
    def _set(self, name, value):
        _default, lo, hi = PROPS[name]
        self.ctrl[name] = float(_clamp(float(value), lo, hi))

    def set_gaze(self, x_deg, y_deg, vergence_deg=None):
        """Gaze direction in degrees. +x = right of the front-camera image, +y = up."""
        self._set("gaze_x_deg", x_deg)
        self._set("gaze_y_deg", y_deg)
        if vergence_deg is not None:
            self._set("vergence_deg", vergence_deg)

    def set_blink(self, amount):
        self._set("blink", amount)

    def set_pupil(self, mm):
        self._set("pupil_mm", mm)

    def apply(self, state, frame=None):
        """Apply a dict of values. Unknown keys are ignored; None values are skipped.
        If `frame` is given, the values are also keyframed at that frame."""
        touched = []
        for key, value in state.items():
            name = ALIASES.get(key, key)
            if name in PROPS and value is not None:
                self._set(name, value)
                touched.append(name)
        if frame is not None:
            self.key(frame, [n for n in touched if n in ANIMATED])
        return touched

    def state(self):
        return {n: float(self.ctrl.get(n, PROPS[n][0])) for n in PROPS}

    def refresh(self):
        """Force drivers to re-evaluate (needed in background mode and for live updates)."""
        self.ctrl.update_tag()
        bpy.context.view_layer.update()

    # ------------------------------------------------------------------ animation
    def key(self, frame, names=None):
        """Insert linear keyframes for the given properties (default: all animated ones)."""
        prefs = bpy.context.preferences.edit
        old = prefs.keyframe_new_interpolation_type
        prefs.keyframe_new_interpolation_type = "LINEAR"
        try:
            for n in names or ANIMATED:
                self.ctrl.keyframe_insert(data_path=f'["{n}"]', frame=frame)
        finally:
            prefs.keyframe_new_interpolation_type = old

    def clear_animation(self):
        self.ctrl.animation_data_clear()

    # ------------------------------------------------------------------ muscles (schematic)
    def activations(self):
        """Illustrative muscle activation 0..1 per muscle, the same formula that colours the muscles.
        Reciprocal and schematic: NOT measured electromyography."""
        s = self.state()
        gx, gy, v = s["gaze_x_deg"], s["gaze_y_deg"], s["vergence_deg"]

        def c(x):
            return max(0.0, min(1.0, x))

        out = {}
        for side, yaw, add in (("R", gx + v / 2, 1), ("L", gx - v / 2, -1)):
            a = yaw * add  # positive = adduction (towards the nose)
            out[f"MR_{side}"] = 0.15 + 0.85 * c(a / 40)
            out[f"LR_{side}"] = 0.15 + 0.85 * c(-a / 40)
            out[f"SR_{side}"] = 0.15 + 0.85 * c(gy / 35)
            out[f"IR_{side}"] = 0.15 + 0.85 * c(-gy / 35)
            out[f"SO_{side}"] = 0.15 + 0.51 * c(-gy / 35)
            out[f"IO_{side}"] = 0.15 + 0.51 * c(gy / 35)
        return out

    # ------------------------------------------------------------------ loading eye_twin data
    def load_csv(self, path, sequence_id=None, start_frame=1, fps=None):
        """Replay an eye_twin labels.csv (columns t, gaze_x_deg, gaze_y_deg, blink, pupil_mm[, sequence_id])."""
        scene = bpy.context.scene
        fps = fps or scene.render.fps
        with open(bpy.path.abspath(path), newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and "sequence_id" in rows[0]:
            sequence_id = sequence_id or rows[0]["sequence_id"]
            rows = [r for r in rows if r["sequence_id"] == sequence_id]
        if not rows:
            raise ValueError("no rows found in the CSV")
        self.clear_animation()
        last = start_frame
        for r in rows:
            frame = start_frame + int(round(float(r["t"]) * fps))
            state = {k: float(r[k]) for k in ("gaze_x_deg", "gaze_y_deg", "blink", "pupil_mm") if r.get(k) not in (None, "")}
            self.apply(state, frame)
            last = max(last, frame)
        scene.frame_start, scene.frame_end = start_frame, last
        return len(rows)

    def load_job(self, path, start_frame=1):
        """Replay a JSON file with per-frame eye states (e.g. the job.json from `eye_twin.py blender-job`).
        Tolerant: finds the list under frames/states/samples/sequence/timeline and accepts common key names."""
        scene = bpy.context.scene
        with open(bpy.path.abspath(path)) as f:
            data = json.load(f)
        frames = data if isinstance(data, list) else next(
            (data[k] for k in ("frames", "states", "samples", "sequence", "timeline") if isinstance(data.get(k), list)), None)
        if not frames:
            keys = list(data) if isinstance(data, dict) else "a non-list value"
            raise ValueError(f"could not find a list of frames in the JSON (top-level keys: {keys})")
        fps = (data.get("fps") if isinstance(data, dict) else None) or scene.render.fps
        self.clear_animation()
        last = start_frame
        for i, fr in enumerate(frames):
            st = _flatten_state(fr)
            if not st:
                continue
            t = st.pop("t", i / fps)
            frame = start_frame + int(round(t * fps))
            self.apply(st, frame)
            last = max(last, frame)
        scene.frame_start, scene.frame_end = start_frame, last
        return len(frames)

    # ------------------------------------------------------------------ run the user's own script
    def run_script(self, path):
        """Run a Python file with `eye` (this ArtificialEye), `bpy` and `ArtificialEye` already defined."""
        path = os.path.abspath(bpy.path.abspath(path))
        folder = os.path.dirname(path)
        if folder not in sys.path:
            sys.path.insert(0, folder)  # so your script can import its sibling modules
        runpy.run_path(path, init_globals={"eye": self, "bpy": bpy, "ArtificialEye": ArtificialEye}, run_name="__main__")
        self.refresh()

    # ------------------------------------------------------------------ live link (external Python -> Blender)
    def start_live_link(self, host="127.0.0.1", port=9876, hz=60):
        """Listen for newline-delimited JSON like {"gaze_x_deg": 5, "blink": 0.2} and apply it live.
        Binds to localhost by default; anything that can reach the port can move the eye, so do not expose it."""
        self.stop_live_link()
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(4)
        srv.setblocking(False)
        live = {"srv": srv, "clients": {}, "applied": 0}

        def poll():
            return self._poll_once(live, 1.0 / hz)

        live["poll"] = poll
        self._live = live
        bpy.app.timers.register(poll, first_interval=0.0)
        return port

    def stop_live_link(self):
        live, self._live = self._live, None
        if not live:
            return
        try:
            bpy.app.timers.unregister(live["poll"])
        except Exception:
            pass
        for c in list(live["clients"]):
            c.close()
        live["srv"].close()

    def _poll_once(self, live, interval):
        """One non-blocking pass: accept clients, read lines, apply the newest state. Returns the next timer interval."""
        if self._live is not live:
            return None  # stopped: unregisters the timer
        try:
            conn, _addr = live["srv"].accept()
            conn.setblocking(False)
            live["clients"][conn] = b""
        except (BlockingIOError, OSError):
            pass
        merged = {}
        for conn in list(live["clients"]):
            try:
                chunk = conn.recv(65536)
                if not chunk:
                    raise ConnectionError
                live["clients"][conn] += chunk
            except BlockingIOError:
                pass
            except Exception:
                conn.close()
                del live["clients"][conn]
                continue
            *lines, rest = live["clients"][conn].split(b"\n")
            live["clients"][conn] = rest[-65536:]  # never let a client grow the buffer without limit
            for line in lines:
                try:
                    msg = json.loads(line)
                    if isinstance(msg, dict):
                        merged.update(msg)  # later messages win; older ones are dropped, so there is no lag build-up
                except ValueError:
                    pass
        if merged:
            self.apply(merged)
            self.refresh()
            live["applied"] += 1
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()
        return interval


def _flatten_state(fr):
    """Pull eye values out of one frame record, looking one level deep for nested dicts."""
    if not isinstance(fr, dict):
        return {}
    flat = dict(fr)
    for nest in ("gaze", "eye", "state", "pose", "labels"):
        if isinstance(fr.get(nest), dict):
            flat.update(fr[nest])
    out = {}
    if "t" in flat:
        out["t"] = float(flat["t"])
    for key, value in flat.items():
        name = ALIASES.get(key, key)
        if name in PROPS and isinstance(value, (int, float)):
            out[name] = float(value)
    for key in ("gaze_deg", "gaze"):
        v = flat.get(key)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            out.setdefault("gaze_x_deg", float(v[0]))
            out.setdefault("gaze_y_deg", float(v[1]))
    return out


# ============================================================================== sidebar UI
if bpy is not None:

    class AEYE_PT_panel(bpy.types.Panel):
        bl_label = "Artificial Eye"
        bl_idname = "AEYE_PT_panel"
        bl_space_type = "VIEW_3D"
        bl_region_type = "UI"
        bl_category = "Artificial Eye"

        def draw(self, context):
            lay = self.layout
            ctrl = bpy.data.objects.get(CTRL_NAME)
            if ctrl is None:
                lay.label(text=f"'{CTRL_NAME}' not found")
                return
            col = lay.column(align=True)
            for n in PROPS:
                col.prop(ctrl, f'["{n}"]', text=n.replace("_", " "), slider=True)

            box = lay.box()
            box.label(text="Your Python script")
            box.prop(context.scene, "aeye_script_path", text="")
            box.operator("artificial_eye.run_script", icon="PLAY")

            box = lay.box()
            box.label(text="Replay eye_twin data")
            box.prop(context.scene, "aeye_data_path", text="")
            box.operator("artificial_eye.load_data", icon="IMPORT")

            box = lay.box()
            box.label(text="Live link (external Python)")
            box.prop(context.scene, "aeye_port")
            on = _LIVE.get("eye") is not None and _LIVE["eye"]._live is not None
            box.operator("artificial_eye.live", text="Stop live link" if on else "Start live link",
                         icon="PAUSE" if on else "PLAY")

            if ctrl.get("gaze_x_deg") is not None:
                box = lay.box()
                box.label(text="Muscle activation (schematic)")
                a = ArtificialEye(ctrl).activations()
                for side in ("R", "L"):
                    box.label(text=f"{side}: " + "  ".join(f"{m} {a[f'{m}_{side}']:.2f}" for m in ("MR", "LR", "SR", "IR", "SO", "IO")))

    _LIVE = {"eye": None}

    class AEYE_OT_run_script(bpy.types.Operator):
        bl_idname = "artificial_eye.run_script"
        bl_label = "Run script on the eye"

        def execute(self, context):
            path = context.scene.aeye_script_path
            if not path:
                self.report({"ERROR"}, "Choose a .py file first")
                return {"CANCELLED"}
            try:
                ArtificialEye().run_script(path)
            except Exception as exc:
                self.report({"ERROR"}, f"{type(exc).__name__}: {exc}")
                return {"CANCELLED"}
            self.report({"INFO"}, "Script finished")
            return {"FINISHED"}

    class AEYE_OT_load_data(bpy.types.Operator):
        bl_idname = "artificial_eye.load_data"
        bl_label = "Load CSV / JSON as animation"

        def execute(self, context):
            path = context.scene.aeye_data_path
            try:
                eye = ArtificialEye()
                n = eye.load_job(path) if path.lower().endswith(".json") else eye.load_csv(path)
            except Exception as exc:
                self.report({"ERROR"}, f"{type(exc).__name__}: {exc}")
                return {"CANCELLED"}
            self.report({"INFO"}, f"Loaded {n} rows. Press play.")
            return {"FINISHED"}

    class AEYE_OT_live(bpy.types.Operator):
        bl_idname = "artificial_eye.live"
        bl_label = "Toggle live link"

        def execute(self, context):
            eye = _LIVE["eye"] or ArtificialEye()
            _LIVE["eye"] = eye
            if eye._live:
                eye.stop_live_link()
                self.report({"INFO"}, "Live link stopped")
            else:
                try:
                    port = eye.start_live_link(port=context.scene.aeye_port)
                except OSError as exc:
                    self.report({"ERROR"}, f"Could not listen: {exc}")
                    return {"CANCELLED"}
                self.report({"INFO"}, f"Listening on 127.0.0.1:{port}")
            return {"FINISHED"}

    _CLASSES = (AEYE_PT_panel, AEYE_OT_run_script, AEYE_OT_load_data, AEYE_OT_live)

    def register():
        bpy.types.Scene.aeye_script_path = bpy.props.StringProperty(name="Script", subtype="FILE_PATH")
        bpy.types.Scene.aeye_data_path = bpy.props.StringProperty(name="Data", subtype="FILE_PATH")
        bpy.types.Scene.aeye_port = bpy.props.IntProperty(name="Port", default=9876, min=1024, max=65535)
        for c in _CLASSES:
            try:
                bpy.utils.register_class(c)
            except ValueError:
                pass  # already registered

    def unregister():
        for c in reversed(_CLASSES):
            try:
                bpy.utils.unregister_class(c)
            except RuntimeError:
                pass

    if __name__ == "__main__":
        register()
