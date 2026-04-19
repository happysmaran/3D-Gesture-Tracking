bl_info = {
    "name": "3D Gesture Tracking",
    "author": "Smaran Vallabhaneni",
    "version": (1, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > 3D Gesture Tracking",
    "description": "3D viewport control based on hand gestures for easy/fun viewing.",
    "category": "3D View",
}

import bpy
import threading
import queue
import sys
import subprocess
import os
import platform
import shutil
from bpy.props import BoolProperty, IntProperty, FloatProperty
from mathutils import Vector, Quaternion

# Dependency installation

_install_lock = threading.Lock()
_deps_ok = False


def _deps_target_dir():
    config_dir = bpy.utils.user_resource('SCRIPTS', path="hand_control_deps", create=True)
    return config_dir


def _ensure_target_on_path(target):
    if target not in sys.path:
        sys.path.insert(0, target)


def _check_deps():
    try:
        import cv2
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        return True
    except ImportError:
        return False


def ensure_dependencies():
    global _deps_ok
    with _install_lock:
        if _deps_ok:
            return True, "Dependencies already satisfied."

        target = _deps_target_dir()
        _ensure_target_on_path(target)

        # Check first — avoids nuking the dir when packages are already installed and importable.
        if _check_deps():
            _deps_ok = True
            return True, "Dependencies already satisfied."

        # Only wipe-and-recreate when needed.
        if os.path.exists(target):
            try:
                shutil.rmtree(target)
            except Exception as e:
                print(f"[Hand Control] Warning: Could not fully clean deps dir: {e}")
        os.makedirs(target, exist_ok=True)
        _ensure_target_on_path(target)

        py = sys.executable
        packages = ["opencv-python"]

        if platform.system() == "Windows":
            packages.append("msvc-runtime")

        packages.append("protobuf==4.25.6")
        packages.append("mediapipe")

        for pkg in packages:
            try:
                print(f"[Hand Control] Installing {pkg} ...")
                subprocess.check_call(
                    [py, "-m", "pip", "install", "--quiet",
                     "--no-warn-script-location", "--target", target, pkg],
                    timeout=300,
                )
            except subprocess.CalledProcessError as exc:
                return False, f"Failed to install {pkg}: {exc}"
            except subprocess.TimeoutExpired:
                return False, f"Timed out installing {pkg}."

        import importlib
        importlib.invalidate_caches()

        if not _check_deps():
            return False, (
                "Packages installed but import still fails.\n"
                f"Target dir: {target}\n"
                "Restart Blender completely (run as Administrator on Windows)."
            )

        _deps_ok = True
        return True, "✅ Dependencies installed successfully."


# Smoothing, GestureClassifier, geometry helpers

class EMAFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self._value = None

    def update(self, v: Vector) -> Vector:
        if self._value is None:
            self._value = v.copy()
        else:
            self._value = self._value.lerp(v, self.alpha)
        return self._value.copy()

    def reset(self):
        self._value = None


class EMAScalar:
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self._value = None

    def update(self, v: float) -> float:
        if self._value is None:
            self._value = v
        else:
            self._value = self._value * (1.0 - self.alpha) + v * self.alpha
        return self._value

    def reset(self):
        self._value = None


_FINGER_DEFS = {
    'index': (8, 5, 6),
    'middle': (12, 9, 10),
    'ring': (16, 13, 14),
    'pinky': (20, 17, 18),
}

HYSTERESIS_MARGIN = 0.18
MIN_CONFIDENCE = 0.28


def _lv(lm, i):
    return Vector((lm[i].x, lm[i].y, lm[i].z))


def _finger_scores(lm):
    wrist = _lv(lm, 0)
    out = {}
    for name, (tip_i, mcp_i, pip_i) in _FINGER_DEFS.items():
        tip = _lv(lm, tip_i)
        mcp = _lv(lm, mcp_i)
        pip = _lv(lm, pip_i)
        d_mcp = (mcp - wrist).length
        curl = ((tip - wrist).length / d_mcp) if d_mcp > 1e-4 else 1.0
        past_pip = (tip - wrist).length > (pip - wrist).length
        out[name] = (curl, curl > 0.95 and past_pip)
    return out


def _pinch_dist(lm):
    return (_lv(lm, 4) - _lv(lm, 8)).length


def _score_orbit(lm, fs):
    # Orbit gestures:
    #   pan - needs index+middle both up
    #   zoom - needs ALL four or more fingers up
    #   orbit - needs ONLY index up, rest down
    idx_curl, _ = fs['index']
    mid_curl, _ = fs['middle']
    rng_curl, _ = fs['ring']
    pky_curl, _ = fs['pinky']

    # Index must be clearly extended
    idx_ext = min(1.0, max(0.0, (idx_curl - 0.80) / 0.35))
    if idx_ext < 0.5:
        return 0.0

    score = idx_ext

    # Middle must be clearly curled — this is the key separator from pan.
    # If middle is up at all, score collapses toward zero.
    mid_ext = min(1.0, max(0.0, (mid_curl - 0.70) / 0.35))
    score -= 0.7 * mid_ext

    # Ring and pinky curled is nice to have but less critical
    rng_ext = min(1.0, max(0.0, (rng_curl - 0.75) / 0.35))
    pky_ext = min(1.0, max(0.0, (pky_curl - 0.75) / 0.35))
    score -= 0.15 * rng_ext
    score -= 0.15 * pky_ext

    return max(0.0, score)


def _score_zoom(lm, fs):
    score = 0.0
    for name in ('index', 'middle', 'ring', 'pinky'):
        curl, _ = fs[name]
        score += min(1.0, max(0.0, (curl - 0.85) / 0.5)) * 0.25
    if score < 0.55:
        return 0.0
    pd = _pinch_dist(lm)
    pinch_pen = max(0.0, 1.0 - pd / 0.10)
    score *= (1.0 - pinch_pen * 0.9)
    return score


def _score_pan(lm, fs):
    idx_curl, _ = fs['index']
    mid_curl, _ = fs['middle']
    rng_curl, _ = fs['ring']
    pky_curl, _ = fs['pinky']
    up = (min(1.0, max(0.0, (idx_curl - 0.85) / 0.4)) * 0.5 +
          min(1.0, max(0.0, (mid_curl - 0.85) / 0.4)) * 0.5)
    down = (min(1.0, max(0.0, (0.90 - rng_curl) / 0.3)) * 0.5 +
            min(1.0, max(0.0, (0.90 - pky_curl) / 0.3)) * 0.5)
    score = (up + down) * 0.5
    pd = _pinch_dist(lm)
    pinch_pen = max(0.0, 1.0 - pd / 0.10)
    score *= (1.0 - pinch_pen * 0.8)
    return score


class GestureClassifier:
    def __init__(self):
        self._current = None
        self._ss = {'orbit': 0.0, 'zoom': 0.0, 'pan': 0.0}

    def classify(self, lm):
        fs = _finger_scores(lm)
        raw = {
            'orbit': _score_orbit(lm, fs),
            'zoom': _score_zoom(lm, fs),
            'pan': _score_pan(lm, fs),
        }
        alpha = 0.4
        for g in self._ss:
            self._ss[g] = self._ss[g] * (1.0 - alpha) + raw[g] * alpha

        best_g = max(self._ss, key=lambda g: self._ss[g])
        best_val = self._ss[best_g]

        if best_val < MIN_CONFIDENCE:
            if self._current and self._ss.get(self._current, 0) < 0.15:
                self._current = None
            return self._current

        if self._current is None:
            self._current = best_g
        elif best_g != self._current:
            if best_val - self._ss.get(self._current, 0.0) >= HYSTERESIS_MARGIN:
                self._current = best_g

        return self._current

    def reset(self):
        self._current = None
        self._ss = {'orbit': 0.0, 'zoom': 0.0, 'pan': 0.0}

    @property
    def smooth_scores(self):
        return dict(self._ss)


def _palm_z(lm):
    knuckles = [0, 5, 9, 13, 17]
    return -sum(lm[i].z for i in knuckles) / len(knuckles)


def _palm_center(lm):
    knuckles = [0, 5, 9, 13, 17]
    x = sum(lm[i].x for i in knuckles) / len(knuckles)
    y = sum(lm[i].y for i in knuckles) / len(knuckles)
    return x, y


# TrackingThread –  MediaPipe Tasks API imports

PREVIEW_IMAGE_NAME = "Hand Control Preview"

_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]


class TrackingThread(threading.Thread):
    def __init__(self, webcam_id, show_preview):
        super().__init__()
        self.webcam_id = webcam_id
        self.show_preview = show_preview
        self.stop_event = threading.Event()
        self.data_queue = queue.Queue(maxsize=1)
        self.daemon = True

        self._cv2 = None
        self._mp = None

        self._wrist_f = EMAFilter(alpha=0.3)
        self._thumb_f = EMAFilter(alpha=0.3)
        self._index_f = EMAFilter(alpha=0.3)
        self._palm_z_f = EMAScalar(alpha=0.2)
        self._palm_x_f = EMAScalar(alpha=0.3)
        self._palm_y_f = EMAScalar(alpha=0.3)
        self._gesture_clf = GestureClassifier()

    def _push(self, pkt):
        if self.data_queue.full():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                pass
        self.data_queue.put(pkt)

    def _make_pkt(self, wrist, thumb, index, gesture, palm_z, palm_x, palm_y,
                  rgba_bytes, w, h, dbg):
        return {
            'wrist': wrist, 'thumb': thumb, 'index': index,
            'gesture': gesture,
            'palm_z': palm_z, 'palm_x': palm_x, 'palm_y': palm_y,
            'preview': (rgba_bytes, w, h) if rgba_bytes is not None else None,
            'debug_frame': dbg,
        }

    def _reset_filters(self):
        for f in (self._wrist_f, self._thumb_f, self._index_f,
                  self._palm_z_f, self._palm_x_f, self._palm_y_f):
            f.reset()
        self._gesture_clf.reset()

    def _process(self, lm, rgb_frame, w, h):
        # Use instance attributes — no risk of NameError regardless of call depth.
        cv2 = self._cv2

        wrist = self._wrist_f.update(_lv(lm, 0))
        thumb = self._thumb_f.update(_lv(lm, 4))
        index = self._index_f.update(_lv(lm, 8))
        gesture = self._gesture_clf.classify(lm)
        palm_z = self._palm_z_f.update(_palm_z(lm))
        px, py = _palm_center(lm)
        palm_x = self._palm_x_f.update(px)
        palm_y = self._palm_y_f.update(py)

        rgba_bytes = None
        dbg = None

        if self.show_preview:
            for i in range(21):
                cv2.circle(rgb_frame, (int(lm[i].x * w), int(lm[i].y * h)), 4, (0, 255, 0), -1)
            for a, b in _HAND_CONNECTIONS:
                cv2.line(rgb_frame,
                         (int(lm[a].x * w), int(lm[a].y * h)),
                         (int(lm[b].x * w), int(lm[b].y * h)),
                         (0, 200, 255), 2)

            ss = self._gesture_clf.smooth_scores
            cv2.putText(rgb_frame, f"Gesture: {gesture or 'none'}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(rgb_frame, f"O:{ss['orbit']:.2f} Z:{ss['zoom']:.2f} N:{ss['pan']:.2f}",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            rgba = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2RGBA)
            rgba = cv2.flip(rgba, 0)
            rgba_bytes = rgba.tobytes()
            if platform.system() != "Darwin":
                dbg = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        return self._make_pkt(wrist, thumb, index, gesture, palm_z, palm_x, palm_y,
                              rgba_bytes, w, h, dbg)

    def run(self):
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            self._cv2 = cv2
            self._mp = mp

            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else \
                      cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_V4L2

            cap = cv2.VideoCapture(self.webcam_id, backend)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.webcam_id)
            if not cap.isOpened():
                print("[Hand Control] Could not open webcam.")
                return

            self._run_tasks_api(cap, mp_python, mp_vision)

            cap.release()
            if platform.system() != "Darwin":
                cv2.destroyAllWindows()

        except Exception as exc:
            import traceback
            print(f"[Hand Control] Tracking thread error: {exc}")
            traceback.print_exc()

    def _run_tasks_api(self, cap, mp_python, mp_vision):
        import tempfile, time, urllib.request

        cv2 = self._cv2
        mp = self._mp

        model_path = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
        if not os.path.exists(model_path):
            print("[Hand Control] Downloading hand landmarker model…")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("[Hand Control] Model downloaded.")

        _cb_lock = threading.Lock()
        _cb_result = [None]

        def _on_result(result, _img, _ts):
            with _cb_lock:
                _cb_result[0] = result

        BaseOptions = mp_python.BaseOptions
        options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            result_callback=_on_result,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        had_hand = False

        with mp_vision.HandLandmarker.create_from_options(options) as detector:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]

                ts = int(time.time() * 1000)
                # Keep a named reference so the Image isn't GC'd before the
                # async callback fires (detect_async holds a raw C++ pointer
                # into the buffer; dropping the object early → crash on macOS).
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                detector.detect_async(mp_img, ts)
                del mp_img

                with _cb_lock:
                    result = _cb_result[0]

                if result and result.hand_landmarks:
                    lm = result.hand_landmarks[0]
                    self._push(self._process(lm, rgb, w, h))
                    had_hand = True
                else:
                    if had_hand:
                        self._reset_filters()
                        had_hand = False
                    if self.show_preview:
                        rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
                        rgba = cv2.flip(rgba, 0)
                        self._push(self._make_pkt(None, None, None, None, None, None, None,
                                                 rgba.tobytes(), w, h, None))


# Viewport / Modal / Panel

_DELTA_DEAD = 0.003
_ORBIT_SCALE = 3.5
_ZOOM_SCALE = 3.0
_PAN_SCALE = 0.8


def _update_preview_image(rgba_bytes, width, height):
    img = bpy.data.images.get(PREVIEW_IMAGE_NAME)
    if img is None or img.size[0] != width or img.size[1] != height:
        if img:
            bpy.data.images.remove(img)
        img = bpy.data.images.new(PREVIEW_IMAGE_NAME, width=width, height=height, alpha=True)
        img.use_fake_user = True   # prevent Blender GC'ing it between redraws
        for name in ('Raw', 'Non-Color'):
            try:
                img.colorspace_settings.name = name
                break
            except TypeError:
                pass
    import numpy as np
    arr = np.frombuffer(rgba_bytes, dtype=np.uint8).astype(np.float32) / 255.0
    img.pixels.foreach_set(arr)
    img.update()
    try:
        img.gl_load()
    except Exception:
        pass

def _find_preview_window():
    """Return the Blender window we opened for the camera preview, or None."""
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                for space in area.spaces:
                    if (space.type == 'IMAGE_EDITOR' and
                            space.image and
                            space.image.name == PREVIEW_IMAGE_NAME):
                        return win
    return None


def _open_preview_window(context):
    """
    Open a new Blender window sized for the camera feed and switch it to the
    Image Editor showing our preview image.  Works on both Windows and macOS
    because it stays inside Blender's own windowing system.
    """
    if not bpy.data.images.get(PREVIEW_IMAGE_NAME):
        bpy.data.images.new(PREVIEW_IMAGE_NAME, width=640, height=480, alpha=True)

    bpy.ops.wm.window_new()

    new_win = context.window_manager.windows[-1]
    area = new_win.screen.areas[0]
    area.type = 'IMAGE_EDITOR'

    for space in area.spaces:
        if space.type == 'IMAGE_EDITOR':
            space.image = bpy.data.images[PREVIEW_IMAGE_NAME]
            space.show_region_header = False
            break

    return new_win


def _close_preview_window():
    """Close the dedicated preview window if it is still open."""
    win = _find_preview_window()
    if win:
        with bpy.context.temp_override(window=win):
            bpy.ops.wm.window_close()


class WM_OT_hand_modal(bpy.types.Operator):
    bl_idname = "wm.hand_control_modal"
    bl_label = "Hand Control Modal"
    _timer = None
    _thread = None
    _preview_win = None

    def _reset_refs(self):
        self._prev_wrist = self._prev_palm_z = self._prev_palm_x = self._prev_palm_y = None
        self._prev_gesture = None

    def modal(self, context, event):
        props = context.scene.hand_control_props
        if not props.enabled:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        try:
            pkt = self._thread.data_queue.get_nowait()
        except queue.Empty:
            return {'PASS_THROUGH'}

        if props.show_preview and pkt.get('preview'):
            rgba_bytes, pw, ph = pkt['preview']
            _update_preview_image(rgba_bytes, pw, ph)
            # Redraw the preview window if it is still alive.
            win = _find_preview_window()
            if win:
                for area in win.screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        area.tag_redraw()

        if pkt['wrist'] is None or not context.region_data:
            self._reset_refs()
            return {'PASS_THROUGH'}

        rv3d = context.region_data
        gesture = pkt['gesture']

        if gesture != self._prev_gesture:
            self._reset_refs()
            self._prev_gesture = gesture

        if gesture == 'orbit':
            wrist = pkt['wrist']
            if self._prev_wrist is None:
                self._prev_wrist = wrist
            else:
                dx = wrist.x - self._prev_wrist.x
                dy = wrist.y - self._prev_wrist.y
                self._prev_wrist = wrist
                if abs(dx) > _DELTA_DEAD or abs(dy) > _DELTA_DEAD:
                    dx *= _ORBIT_SCALE * props.sensitivity
                    dy *= _ORBIT_SCALE * props.sensitivity
                    q_yaw = Quaternion((0, 0, 1), -dx)
                    view_x = rv3d.view_rotation @ Vector((1, 0, 0))
                    q_pitch = Quaternion(view_x, dy)
                    rv3d.view_rotation = (q_yaw @ q_pitch @ rv3d.view_rotation).normalized()

        elif gesture == 'zoom':
            pz = pkt['palm_z']
            if pz is not None:
                if self._prev_palm_z is None:
                    self._prev_palm_z = pz
                else:
                    dz = pz - self._prev_palm_z
                    self._prev_palm_z = pz
                    if abs(dz) > _DELTA_DEAD:
                        rv3d.view_distance = max(
                            0.001, rv3d.view_distance * (1.0 - dz * _ZOOM_SCALE * props.sensitivity))

        elif gesture == 'pan':
            px, py = pkt['palm_x'], pkt['palm_y']
            if px is not None and py is not None:
                if self._prev_palm_x is None:
                    self._prev_palm_x, self._prev_palm_y = px, py
                else:
                    dx = px - self._prev_palm_x
                    dy = py - self._prev_palm_y
                    self._prev_palm_x, self._prev_palm_y = px, py
                    if abs(dx) > _DELTA_DEAD or abs(dy) > _DELTA_DEAD:
                        sc = _PAN_SCALE * props.sensitivity * rv3d.view_distance
                        view_right = rv3d.view_rotation @ Vector((1, 0, 0))
                        view_up = rv3d.view_rotation @ Vector((0, 1, 0))
                        rv3d.view_location += view_right * (-dx * sc) + view_up * (dy * sc)

        context.area.tag_redraw()
        return {'PASS_THROUGH'}

    def execute(self, context):
        props = context.scene.hand_control_props
        ok, msg = ensure_dependencies()
        if not ok:
            self.report({'ERROR'}, msg)
            props.enabled = False
            return {'CANCELLED'}
        self._reset_refs()
        self._thread = TrackingThread(props.webcam_id, props.show_preview)
        self._thread.start()
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.016, window=context.window)

        if props.show_preview:
            try:
                _open_preview_window(context)
            except Exception as e:
                print(f"[Hand Control] Could not open preview window: {e}")

        return {'RUNNING_MODAL'}

    def cancel(self, context):
        if self._thread:
            self._thread.stop_event.set()

        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

        try:
            _close_preview_window()
        except Exception as e:
            print(f"[Hand Control] Could not close preview window: {e}")

        img = bpy.data.images.get(PREVIEW_IMAGE_NAME)
        if img:
            bpy.data.images.remove(img)


class HAND_OT_toggle(bpy.types.Operator):
    bl_idname = "hand_control.toggle"
    bl_label = "Toggle Hand Tracking"

    def execute(self, context):
        props = context.scene.hand_control_props
        props.enabled = not props.enabled
        if props.enabled:
            bpy.ops.wm.hand_control_modal('INVOKE_DEFAULT')
        return {'FINISHED'}


class HAND_OT_install_deps(bpy.types.Operator):
    bl_idname = "hand_control.install_deps"
    bl_label = "Install Dependencies (takes a minute)"
    bl_description = "Install opencv + mediapipe + msvc-runtime"

    def execute(self, context):
        self.report({'INFO'}, "Installing dependencies (may take 30-90 seconds)…")
        ok, msg = ensure_dependencies()
        self.report({'INFO' if ok else 'ERROR'}, msg)
        return {'FINISHED'}


class VIEW3D_PT_hand_control(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "3D Gesture Tracking"
    bl_label = "3D Gesture Tracking"

    def draw(self, context):
        layout = self.layout
        props = context.scene.hand_control_props

        if not _check_deps():
            box = layout.box()
            box.label(text="Dependencies missing!", icon='ERROR')
            box.operator("hand_control.install_deps", icon='IMPORT')
            layout.separator()

        row = layout.row()
        row.scale_y = 1.5
        row.operator(
            "hand_control.toggle",
            text="STOP" if props.enabled else "START",
            icon='CANCEL' if props.enabled else 'PLAY',
            depress=props.enabled,
        )

        layout.separator()
        col = layout.column(align=True)
        col.prop(props, "webcam_id", text="Camera Index")
        col.prop(props, "sensitivity", text="Sensitivity")
        col.prop(props, "show_preview", text="Show Camera Preview")

        if props.show_preview:
            img = bpy.data.images.get(PREVIEW_IMAGE_NAME)
            layout.label(
                text="Preview window open" if img else "Preview opens in a new window on start",
                icon='WINDOW' if img else 'INFO')

        layout.separator()
        box = layout.box()
        box.label(text="Gestures:", icon='HAND')
        col = box.column(align=True)
        col.label(text="Point (index finger only) → Orbit")
        col.label(text="Open palm → Zoom")
        col.label(text="Index+Middle → Pan")


class HandControlProps(bpy.types.PropertyGroup):
    enabled: BoolProperty(name="Enabled", default=False)
    webcam_id: IntProperty(name="Webcam ID", default=0, min=0, max=9)
    sensitivity: FloatProperty(name="Sensitivity", default=1.0, min=0.1, max=5.0, soft_max=3.0)
    show_preview: BoolProperty(name="Show Preview", default=False)


_classes = (HandControlProps, HAND_OT_toggle, HAND_OT_install_deps,
            WM_OT_hand_modal, VIEW3D_PT_hand_control)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.hand_control_props = bpy.props.PointerProperty(type=HandControlProps)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.hand_control_props


if __name__ == "__main__":
    register()