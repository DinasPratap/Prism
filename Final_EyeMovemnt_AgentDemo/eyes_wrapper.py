"""
eyes_wrapper.py  --  deterministic Python control of the FaceRig eyes (runs inside Blender)

    from eyes_wrapper import eyes          # FaceRig.blend must be open / loaded
    eyes.move(h, v)                        # gaze offset in degrees: h > 0 = image right, v > 0 = up
    eyes.reset()                           # back to the 35 cm fixation point, 4 mm pupils
    eyes.pupil(5.5)                        # both pupils, diameter in mm (always synchronous)
    eyes.pupil_centers_px()                # ground-truth 2D pupil centres in the locked camera image
    eyes.render("frame.png")               # 1024x1024 PNG from the locked camera

GEOMETRY (all lengths in centimetres; 1 Blender unit = 1 cm)
  * M = midpoint between the two eyeball centres. The rig is built so M is the world origin.
  * The face looks towards -Y. The locked camera sits CAM_DIST = 40 cm in front of M, on the midline,
    looking along +Y, and cannot be moved (all its transform channels are locked).
  * The fixation target sits FIX_DIST = 35 cm in front of M. Each eye aims at it with a Damped Track
    constraint on its bone (eye.L / eye.R), so vergence is automatic.
  * move(h, v) puts the target at M + 35 cm * g(h, v), where g = (sin h cos v, -cos h cos v, sin v).
    Nothing random is involved: the same (h, v) always gives the same pose, and reset() always returns
    exactly to the neutral pose.
IMAGE COORDINATES: continuous pixels, origin at the image's top-left corner, x right, y down; pixel i covers [i, i+1).
"""
import math

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

FIX_DIST = 35.0
CAM_DIST = 40.0
PUPIL_DEFAULT_MM = 4.0
PUPIL_MIN_MM, PUPIL_MAX_MM = 1.5, 9.0


class Eyes:
    def __init__(self, scene=None):
        self.scene = scene or bpy.context.scene
        o = bpy.data.objects
        try:
            self.target = o["FIXATION_TARGET"]
            self.cam = o["Cam_Locked"]
            self.rig = o["FaceRig"]
            self.eyeball = {"L": o["Eyeball_L"], "R": o["Eyeball_R"]}
            self.pupil_obj = {"L": o["Pupil_L"], "R": o["Pupil_R"]}
        except KeyError as exc:
            raise RuntimeError(f"FaceRig object {exc} not found. Open FaceRig.blend first.") from None
        self.h = self.v = 0.0
        self.pupil_mm = PUPIL_DEFAULT_MM
        self.mid = (self._world(self.eyeball["L"]).translation + self._world(self.eyeball["R"]).translation) / 2
        self.reset()

    # ------------------------------------------------------------------ helpers
    def _update(self):
        bpy.context.view_layer.update()

    def _world(self, obj):
        dg = bpy.context.evaluated_depsgraph_get()
        return obj.evaluated_get(dg).matrix_world

    @staticmethod
    def gaze_direction(h_deg, v_deg):
        h, v = math.radians(h_deg), math.radians(v_deg)
        return Vector((math.sin(h) * math.cos(v), -math.cos(h) * math.cos(v), math.sin(v)))

    # ------------------------------------------------------------------ controls
    def move(self, h, v):
        """Aim both eyes at the fixation point displaced by (h, v) degrees. h > 0 = image right, v > 0 = up."""
        self.h, self.v = float(h), float(v)
        self.target.location = self.mid + FIX_DIST * self.gaze_direction(self.h, self.v)
        self._update()
        return self

    def pupil(self, mm):
        """Set both pupils (always synchronous) to a diameter in millimetres."""
        self.pupil_mm = max(PUPIL_MIN_MM, min(PUPIL_MAX_MM, float(mm)))
        r = self.pupil_mm / 20.0  # radius in cm
        for p in self.pupil_obj.values():
            p.scale = (r, 1.0, r)
        self._update()
        return self

    def reset(self):
        """Neutral pose: eyes fixate the point 35 cm straight ahead, pupils 4 mm."""
        self.move(0.0, 0.0)
        self.pupil(PUPIL_DEFAULT_MM)
        return self

    @staticmethod
    def circle_angle(k, n=10):
        """theta_k = 2*pi*k/n, measured CLOCKWISE from 12 o'clock as seen in the image."""
        return 2.0 * math.pi * k / n

    def move_on_circle(self, k, n=10, amplitude_deg=20.0):
        """Gaze offset for step k of n on a clockwise circle that starts at 12 o'clock (up)."""
        th = self.circle_angle(k, n)
        return self.move(amplitude_deg * math.sin(th), amplitude_deg * math.cos(th))

    # ------------------------------------------------------------------ ground truth
    def _resolution(self):
        r = self.scene.render
        return r.resolution_x * r.resolution_percentage / 100.0, r.resolution_y * r.resolution_percentage / 100.0

    def project(self, world_point):
        """World point -> continuous image pixels (x right, y down). Also returns depth in cm along the optical axis."""
        co = world_to_camera_view(self.scene, self.cam, Vector(world_point))
        w, h = self._resolution()
        return (co.x * w, (1.0 - co.y) * h), co.z

    def pupil_centers_px(self):
        """Ground-truth 2D pupil centres {'L': (x, y), 'R': (x, y)}: the pupil-disc centres projected through the camera."""
        return {s: self.project(self._world(p).translation)[0] for s, p in self.pupil_obj.items()}

    def pupil_diameter_px(self):
        """Projected pupil diameter in pixels (pinhole approximation: fx * diameter / depth)."""
        cal = self.calibration()
        out = {}
        for s, p in self.pupil_obj.items():
            _px, depth = self.project(self._world(p).translation)
            out[s] = cal["intrinsics"]["fx"] * (self.pupil_mm / 10.0) / depth
        return out

    def gaze_vectors(self):
        """Unit optical-axis vectors of each eye in world space."""
        return {s: (self._world(e).to_3x3() @ Vector((0, -1, 0))).normalized() for s, e in self.eyeball.items()}

    def eye_centers(self):
        return {s: self._world(e).translation.copy() for s, e in self.eyeball.items()}

    def gaze_error_deg(self):
        """Angle between each eye's optical axis and the line from the eye to the fixation target (should be ~0)."""
        t = self._world(self.target).translation
        out = {}
        for s, c in self.eye_centers().items():
            out[s] = math.degrees((t - c).normalized().angle(self.gaze_vectors()[s]))
        return out

    # ------------------------------------------------------------------ calibration
    def calibration(self):
        cam, sc = self.cam, self.scene
        w, h = self._resolution()
        cd = cam.data
        fx = cd.lens / cd.sensor_width * w
        mw = self._world(cam)
        return {
            "image_size_px": [int(w), int(h)],
            "intrinsics": {"fx": fx, "fy": fx, "cx": w / 2.0 + cd.shift_x * w, "cy": h / 2.0 - cd.shift_y * w,
                           "focal_length_mm": cd.lens, "sensor_width_mm": cd.sensor_width},
            "camera_to_world": [list(r) for r in mw],
            "camera_position_cm": list(mw.translation),
            "camera_distance_to_eye_midpoint_cm": (mw.translation - self.mid).length,
            "fixation_distance_cm": (self._world(self.target).translation - self.mid).length,
            "eye_midpoint_cm": list(self.mid),
            "eye_centers_cm": {s: list(c) for s, c in self.eye_centers().items()},
            "interocular_distance_cm": (self.eye_centers()["L"] - self.eye_centers()["R"]).length,
            "camera_locked": all(cam.lock_location) and all(cam.lock_rotation) and all(cam.lock_scale),
            "units": "centimetres; 1 Blender unit = 1 cm",
        }

    # ------------------------------------------------------------------ rendering
    def configure_render(self, engine="CYCLES", samples=48, size=1024, seed=0, denoise=False):
        sc = self.scene
        sc.render.resolution_x = sc.render.resolution_y = size
        sc.render.resolution_percentage = 100
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_mode = "RGB"
        sc.render.image_settings.color_depth = "8"
        sc.render.film_transparent = False
        if engine == "CYCLES":
            sc.render.engine = "CYCLES"
            sc.cycles.device = "CPU"
            sc.cycles.samples = samples
            sc.cycles.seed = seed           # fixed seed + no animated seed => reproducible noise pattern
            sc.cycles.use_animated_seed = False
            sc.cycles.use_denoising = bool(denoise)   # needs a Blender build with OpenImageDenoise (official builds have it)
            sc.cycles.max_bounces = 6
            sc.cycles.use_adaptive_sampling = True   # deterministic for a fixed seed; skips converged pixels
            sc.cycles.adaptive_threshold, sc.cycles.adaptive_min_samples = 0.02, 8
            sc.cycles.caustics_reflective = sc.cycles.caustics_refractive = False
            sc.cycles.sample_clamp_indirect = 3.0
        else:  # fast preview
            sc.render.engine = "BLENDER_WORKBENCH"
            sc.display.shading.light = "STUDIO"
            sc.display.shading.color_type = "TEXTURE"
        return self

    def render(self, path):
        self.scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        return path

    def render_mask(self, path, samples=4):
        """Alpha mask of the *visible* pupils (eyelids occlude, cornea refraction ignored), used to validate the projection."""
        sc, vl = self.scene, bpy.context.view_layer
        mat = bpy.data.materials.get("MASK_WHITE")
        if mat is None:
            mat = bpy.data.materials.new("MASK_WHITE")
            mat.use_nodes = True
            nt = mat.node_tree
            nt.nodes.clear()
            em = nt.nodes.new("ShaderNodeEmission")
            out = nt.nodes.new("ShaderNodeOutputMaterial")
            nt.links.new(em.outputs[0], out.inputs[0])
        saved_obj = {o: (o.is_holdout, o.hide_render) for o in sc.objects if o.type == "MESH"}
        saved_sc = (sc.render.engine, sc.render.film_transparent, sc.cycles.samples, sc.cycles.use_denoising,
                    sc.render.image_settings.color_mode, vl.material_override, sc.view_settings.view_transform)
        try:
            for o in saved_obj:
                if o.name.startswith("Pupil_"):
                    o.is_holdout = False
                elif o.name.startswith("Cornea_"):
                    o.hide_render = True
                else:
                    o.is_holdout = True
            sc.render.engine = "CYCLES"
            sc.render.film_transparent = True
            sc.cycles.samples, sc.cycles.use_denoising = samples, False
            sc.render.image_settings.color_mode = "RGBA"
            sc.view_settings.view_transform = "Standard"
            vl.material_override = mat
            self.render(path)
        finally:
            for o, (hold, hide) in saved_obj.items():
                o.is_holdout, o.hide_render = hold, hide
            (sc.render.engine, sc.render.film_transparent, sc.cycles.samples, sc.cycles.use_denoising,
             sc.render.image_settings.color_mode, vl.material_override, sc.view_settings.view_transform) = saved_sc
        return path

    # ------------------------------------------------------------------ state
    def state(self):
        c = self.pupil_centers_px()
        return {"h_deg": self.h, "v_deg": self.v, "pupil_mm": self.pupil_mm,
                "pupil_center_px": {s: list(p) for s, p in c.items()}}


try:
    eyes = Eyes()
except RuntimeError:       # imported outside the FaceRig scene (e.g. for a syntax check)
    eyes = None
