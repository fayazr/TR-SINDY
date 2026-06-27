"""Turbulence Realm SINDy — Mobile Edition (Kivy)

A touch-friendly mobile app for video-based fluid-flow analysis.
Runs on Android (via buildozer/python-for-android) and desktop.

Features:
  - Load video from file picker
  - Draw ROI with touch
  - Optical flow processing (Farneback via cv2)
  - SINDy model fitting (pure-numpy STLSQ)
  - Velocity field visualization (Kivy canvas)
  - Discovered equation display
"""

from __future__ import annotations

import os
import threading

import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.graphics import Color, Line, Rectangle, Ellipse, Mesh
from kivy.lang import Builder
from kivy.properties import BooleanProperty, NumericProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.modalview import ModalView
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider
from kivy.uix.spinner import Spinner
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget

from .flow_lite import process_video
from .sindy_lite import fit_sindy, predict_sindy, compute_gradient

KV = """
#:kivy 2.0

<TRSindyMobile>:
    orientation: 'vertical'
    canvas.before:
        Color:
            rgba: 0.965, 0.953, 0.925, 1
        Rectangle:
            pos: self.pos
            size: self.size

    BoxLayout:
        size_hint_y: None
        height: '56dp'
        canvas.before:
            Color:
                rgba: 0.937, 0.918, 0.878, 1
            Rectangle:
                pos: self.pos
                size: self.size
        Label:
            text: 'Turbulence Realm — SINDy'
            font_size: '18sp'
            bold: True
            color: 0.651, 0.486, 0.165, 1
            size_hint_x: 0.6
        Label:
            id: status_label
            text: 'Ready'
            font_size: '12sp'
            color: 0.353, 0.376, 0.420, 1
            size_hint_x: 0.4
            halign: 'right'

    TabbedPanel:
        id: main_tabs
        do_default_tab: False
        tab_pos: 'top'

        TabbedPanelItem:
            text: 'Setup'
            SetupTab:
                id: setup_tab

        TabbedPanelItem:
            text: 'Flow'
            FlowTab:
                id: flow_tab

        TabbedPanelItem:
            text: 'SINDy'
            SINDyTab:
                id: sindy_tab

        TabbedPanelItem:
            text: 'Viz'
            VizTab:
                id: viz_tab


<SetupTab>:
    orientation: 'vertical'
    padding: '12dp'
    spacing: '8dp'

    BoxLayout:
        size_hint_y: None
        height: '48dp'
        Button:
            text: 'Browse Video…'
            on_press: root.browse_video()
            background_color: 0.651, 0.486, 0.165, 1
        Label:
            id: video_label
            text: 'No video selected'
            color: 0.353, 0.376, 0.420, 1
            font_size: '12sp'

    BoxLayout:
        size_hint_y: None
        height: '48dp'
        Label:
            text: 'Calib (px):'
            size_hint_x: 0.3
            color: 0.353, 0.376, 0.420, 1
        TextInput:
            id: calib_px
            text: '200'
            input_filter: 'float'
            multiline: False
        Label:
            text: 'Calib (m):'
            size_hint_x: 0.3
            color: 0.353, 0.376, 0.420, 1
        TextInput:
            id: calib_m
            text: '0.1'
            input_filter: 'float'
            multiline: False

    BoxLayout:
        size_hint_y: None
        height: '48dp'
        Label:
            text: 'Backend:'
            size_hint_x: 0.3
            color: 0.353, 0.376, 0.420, 1
        Spinner:
            id: backend_spinner
            text: 'farneback'
            values: ['farneback', 'lucas_kanade']
            size_hint_x: 0.7

    BoxLayout:
        size_hint_y: None
        height: '48dp'
        Label:
            text: 'Library degree:'
            size_hint_x: 0.4
            color: 0.353, 0.376, 0.420, 1
        Spinner:
            id: degree_spinner
            text: '3'
            values: ['1', '2', '3', '4']
            size_hint_x: 0.3
        Label:
            text: 'Threshold:'
            size_hint_x: 0.3
            color: 0.353, 0.376, 0.420, 1

    Slider:
        id: threshold_slider
        min: 0.001
        max: 1.0
        value: 0.07
        step: 0.001
        size_hint_y: None
        height: '32dp'
    Label:
        id: threshold_label
        text: '0.070'
        color: 0.651, 0.486, 0.165, 1
        font_size: '12sp'
        size_hint_y: None
        height: '20dp'

    Label:
        text: 'Draw ROI on the Flow tab after loading video'
        color: 0.353, 0.376, 0.420, 1
        font_size: '11sp'
        size_hint_y: None
        height: '24dp'

    Widget:
        size_hint_y: None
        height: '8dp'

    Button:
        text: '① Process Optical Flow'
        size_hint_y: None
        height: '56dp'
        font_size: '16sp'
        background_color: 0.651, 0.486, 0.165, 1
        on_press: root.run_optical_flow()
        disabled: not app.video_path

    Button:
        text: '② Run SINDy'
        size_hint_y: None
        height: '56dp'
        font_size: '16sp'
        background_color: 0.478, 0.353, 0.114, 1
        on_press: root.run_sindy()
        disabled: not app.flow_result

    ProgressBar:
        id: progress_bar
        size_hint_y: None
        height: '8dp'
        max: 100
        value: 0


<FlowTab>:
    orientation: 'vertical'
    canvas.before:
        Color:
            rgba: 0.937, 0.918, 0.878, 1
        Rectangle:
            pos: self.pos
            size: self.size
    Label:
        text: 'Touch and drag to draw ROI rectangle'
        color: 0.353, 0.376, 0.420, 1
        font_size: '11sp'
        size_hint_y: None
        height: '24dp'
    ROIDrawingWidget:
        id: roi_widget
        size_hint: 1, 1


<SINDyTab>:
    orientation: 'vertical'
    padding: '12dp'
    spacing: '8dp'
    ScrollView:
        Label:
            id: equation_label
            text: 'Run SINDy to see discovered equations'
            markup: True
            font_size: '14sp'
            color: 0.651, 0.486, 0.165, 1
            size_hint_y: None
            height: self.texture_size[1]
            text_size: self.width, None
            halign: 'left'
            valign: 'top'
    BoxLayout:
        size_hint_y: None
        height: '48dp'
        Label:
            id: sindy_info
            text: ''
            color: 0.353, 0.376, 0.420, 1
            font_size: '11sp'


<VizTab>:
    orientation: 'vertical'
    canvas.before:
        Color:
            rgba: 0.937, 0.918, 0.878, 1
        Rectangle:
            pos: self.pos
            size: self.size
    Spinner:
        id: viz_mode
        text: 'quiver'
        values: ['quiver', 'magnitude', 'vorticity']
        size_hint_y: None
        height: '40dp'
        on_text: root.update_viz()
    FlowCanvasWidget:
        id: flow_canvas
        size_hint: 1, 1
"""


# ---------------------------------------------------------------------
#  Custom widgets
# ---------------------------------------------------------------------
class ROIDrawingWidget(Widget):
    """Widget that displays the first video frame and lets user draw ROI."""

    frame_texture = ObjectProperty(None)
    roi = ObjectProperty(None)  # (x0, y0, x1, y1) in normalized coords

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._touch_start = None
        self._touch_end = None
        self.bind(size=self._update_rect, pos=self._update_rect)

    def set_frame(self, frame_array):
        """Set the background frame from a numpy array (H, W, 3) uint8."""
        if frame_array is None:
            self.frame_texture = None
            self.canvas.clear()
            return
        # Convert numpy array to Kivy texture
        from kivy.graphics.texture import Texture
        h, w = frame_array.shape[:2]
        buf = frame_array.tobytes()
        tex = Texture.create(size=(w, h), colorfmt='rgb')
        tex.blit_buffer(buf, colorfmt='rgb')
        tex.flip_vertical()
        self.frame_texture = tex
        self.canvas.clear()
        with self.canvas:
            Color(1, 1, 1, 1)
            Rectangle(texture=tex, pos=self.pos, size=self.size)

    def _update_rect(self, *args):
        if self.frame_texture:
            self.canvas.clear()
            with self.canvas:
                Color(1, 1, 1, 1)
                Rectangle(texture=self.frame_texture, pos=self.pos,
                          size=self.size)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self._touch_start = (touch.x, touch.y)
            self._touch_end = (touch.x, touch.y)

    def on_touch_move(self, touch):
        if self._touch_start and self.collide_point(*touch.pos):
            self._touch_end = (touch.x, touch.y)
            self._draw_roi()

    def on_touch_up(self, touch):
        if self._touch_start and self._touch_end:
            x0 = min(self._touch_start[0], self._touch_end[0])
            y0 = min(self._touch_start[1], self._touch_end[1])
            x1 = max(self._touch_start[0], self._touch_end[0])
            y1 = max(self._touch_start[1], self._touch_end[1])
            # Convert to normalized coords relative to widget
            self.roi = (
                (x0 - self.x) / self.width,
                (y0 - self.y) / self.height,
                (x1 - self.x) / self.width,
                (y1 - self.y) / self.height,
            )
            self._draw_roi()

    def _draw_roi(self):
        self._update_rect()
        if self._touch_start and self._touch_end:
            with self.canvas:
                Color(0.651, 0.486, 0.165, 0.8)
                x0 = min(self._touch_start[0], self._touch_end[0])
                y0 = min(self._touch_start[1], self._touch_end[1])
                x1 = max(self._touch_start[0], self._touch_end[0])
                y1 = max(self._touch_start[1], self._touch_end[1])
                Line(rectangle=(x0, y0, x1 - x0, y1 - y0), width=2)

    def get_roi_pixels(self, frame_w, frame_h):
        """Get ROI in pixel coordinates of the original frame."""
        if self.roi is None:
            return (0, 0, frame_w, frame_h)
        nx0, ny0, nx1, ny1 = self.roi
        # Flip Y (Kivy Y is bottom-up, image Y is top-down)
        return (
            int(nx0 * frame_w),
            int((1 - ny1) * frame_h),
            int(nx1 * frame_w),
            int((1 - ny0) * frame_h),
        )


class FlowCanvasWidget(Widget):
    """Widget that renders velocity field visualization on Kivy canvas."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.u = None
        self.v = None
        self.mode = "quiver"
        self.bind(size=self._redraw, pos=self._redraw)

    def set_data(self, u, v, mode="quiver"):
        self.u = u
        self.v = v
        self.mode = mode
        self._redraw()

    def _redraw(self, *args):
        self.canvas.clear()
        if self.u is None or self.v is None:
            with self.canvas:
                Color(0.290, 0.310, 0.353, 1)
                Rectangle(pos=self.pos, size=self.size)
            return
        u, v = self.u, self.v
        h, w = u.shape
        with self.canvas:
            if self.mode == "magnitude":
                mag = np.sqrt(u**2 + v**2)
                if mag.max() > 0:
                    norm = (mag / mag.max() * 255).astype(np.uint8)
                else:
                    norm = np.zeros((h, w), np.uint8)
                # Draw as colored rectangles (downsampled)
                step = max(1, h // 100)
                for y in range(0, h, step):
                    for x in range(0, w, step):
                        val = norm[y, x] / 255.0
                        # Blue → cyan → yellow → red
                        if val < 0.33:
                            r, g, b = 0, val * 3, 1
                        elif val < 0.66:
                            r, g, b = 0, 1, (1 - (val - 0.33) * 3)
                        else:
                            r, g, b = (val - 0.66) * 3, 1, 0
                        Color(r, g, b, 0.9)
                        px = self.x + (x / w) * self.width
                        py = self.y + self.height - ((y + step) / h) * self.height
                        Rectangle(pos=(px, py),
                                  size=(self.width / w * step + 1,
                                        self.height / h * step + 1))
            elif self.mode == "vorticity":
                if h > 2 and w > 2:
                    dvdx = np.gradient(v, axis=1)
                    dudy = np.gradient(u, axis=0)
                    vort = dvdx - dudy
                    vmax = max(abs(vort.min()), abs(vort.max()), 1e-10)
                    vort_norm = vort / vmax
                else:
                    vort_norm = np.zeros((h, w))
                step = max(1, h // 100)
                for y in range(0, h, step):
                    for x in range(0, w, step):
                        val = vort_norm[y, x]
                        if val > 0:
                            Color(1, 0.3, 0.3, min(abs(val), 1) * 0.9)
                        else:
                            Color(0.3, 0.3, 1, min(abs(val), 1) * 0.9)
                        px = self.x + (x / w) * self.width
                        py = self.y + self.height - ((y + step) / h) * self.height
                        Rectangle(pos=(px, py),
                                  size=(self.width / w * step + 1,
                                        self.height / h * step + 1))
            else:  # quiver
                step = max(2, min(h, w) // 25)
                mag = np.sqrt(u**2 + v**2)
                max_mag = mag.max() if mag.max() > 0 else 1
                for y in range(step // 2, h, step):
                    for x in range(step // 2, w, step):
                        du = u[y, x] / max_mag
                        dv = v[y, x] / max_mag
                        px = self.x + (x / w) * self.width
                        py = self.y + self.height - (y / h) * self.height
                        arrow_len = step * 1.5
                        ex = px + du * arrow_len
                        ey = py - dv * arrow_len
                        # Color by magnitude (gold gradient)
                        m = mag[y, x] / max_mag
                        Color(0.651 + m * 0.125, 0.486 + m * 0.114, 0.165 + m * 0.062, 0.9)
                        Line(points=[px, py, ex, ey], width=1.5)
                        # Arrowhead dot
                        Color(1, 1, 1, 0.8)
                        Ellipse(pos=(ex - 2, ey - 2), size=(4, 4))


# ---------------------------------------------------------------------
#  Tab content widgets
# ---------------------------------------------------------------------
class SetupTab(BoxLayout):
    def browse_video(self):
        content = BoxLayout(orientation='vertical')
        fc = FileChooserListView(filters=['*.mp4', '*.avi', '*.mov', '*.mkv'])
        content.add_widget(fc)
        btn_box = BoxLayout(size_hint_y=None, height='48dp')
        ok_btn = Button(text='Select')
        cancel_btn = Button(text='Cancel')
        btn_box.add_widget(ok_btn)
        btn_box.add_widget(cancel_btn)
        content.add_widget(btn_box)
        popup = Popup(title='Select Video', content=content, size_hint=(0.9, 0.9))
        ok_btn.bind(on_press=lambda x: self._on_video_selected(fc.selection, popup))
        cancel_btn.bind(on_press=popup.dismiss)
        popup.open()

    def _on_video_selected(self, selection, popup):
        if selection:
            app = App.get_running_app()
            app.video_path = selection[0]
            self.ids.video_label.text = os.path.basename(selection[0])
            # Load first frame for ROI drawing
            self._load_first_frame(selection[0])
        popup.dismiss()

    def _load_first_frame(self, path):
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                # Resize for display
                h, w = frame.shape[:2]
                max_dim = 800
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                app = App.get_running_app()
                app.first_frame = frame
                app.frame_w, app.frame_h = w, h
                # Show on Flow tab
                flow_tab = self.ids.flow_tab if hasattr(self.ids, 'flow_tab') else None
                # Access via app root
                root = App.get_running_app().root
                flow_tab = root.ids.flow_tab
                flow_tab.ids.roi_widget.set_frame(frame)
        except Exception as e:
            print(f"Error loading frame: {e}")

    def run_optical_flow(self):
        app = App.get_running_app()
        if not app.video_path:
            return
        # Get ROI
        root = app.root
        roi_widget = root.ids.flow_tab.ids.roi_widget
        roi = roi_widget.get_roi_pixels(app.frame_w, app.frame_h)
        if roi is None:
            roi = (0, 0, app.frame_w, app.frame_h)
        # Calibration
        calib_px = float(self.ids.calib_px.text or "200")
        calib_m = float(self.ids.calib_m.text or "0.1")
        mpp = calib_m / calib_px
        backend = self.ids.backend_spinner.text
        # Run in thread
        threading.Thread(target=self._flow_thread, args=(
            app.video_path, roi, mpp, backend), daemon=True).start()

    def _flow_thread(self, path, roi, mpp, backend):
        app = App.get_running_app()
        Clock.schedule_once(lambda dt: self._set_status("Processing optical flow…"))
        try:
            result = process_video(path, roi, mpp, backend=backend,
                                   max_frames=60,
                                   progress_cb=self._progress_cb)
            app.flow_result = result
            Clock.schedule_once(lambda dt: self._on_flow_done(result))
        except Exception as e:
            Clock.schedule_once(lambda dt: self._show_error(str(e)))

    def _progress_cb(self, i, n, stage):
        pct = int(i / max(n, 1) * 100)
        Clock.schedule_once(lambda dt: self._set_progress(pct))

    def _set_progress(self, pct):
        self.ids.progress_bar.value = pct

    def _set_status(self, msg):
        app = App.get_running_app()
        root = app.root
        root.ids.status_label.text = msg

    def _on_flow_done(self, result):
        self.ids.progress_bar.value = 100
        self._set_status(f"Flow done: {result['n_frames']} frames")
        # Show first frame velocity on Viz tab
        app = App.get_running_app()
        root = app.root
        viz_tab = root.ids.viz_tab
        if result["u"].shape[0] > 0:
            viz_tab.ids.flow_canvas.set_data(
                result["u"][0], result["v"][0], "quiver")

    def _show_error(self, msg):
        self._set_status(f"Error: {msg}")
        popup = Popup(title='Error', content=Label(text=msg),
                      size_hint=(0.8, 0.4))
        popup.open()

    def run_sindy(self):
        app = App.get_running_app()
        if not app.flow_result:
            return
        threading.Thread(target=self._sindy_thread, daemon=True).start()

    def _sindy_thread(self):
        app = App.get_running_app()
        result = app.flow_result
        Clock.schedule_once(lambda dt: self._set_status("Fitting SINDy…"))
        try:
            u = result["u"]
            v = result["v"]
            # Spatially average to get a mean flow trajectory
            u_mean = u.mean(axis=(1, 2))
            v_mean = v.mean(axis=(1, 2))
            X = np.column_stack([u_mean, v_mean])
            dt = result["dt"]
            Xdot = compute_gradient(X, dt)
            degree = int(self.ids.degree_spinner.text)
            threshold = self.ids.threshold_slider.value
            model = fit_sindy(X, Xdot, degree=degree, threshold=threshold)
            app.sindy_model = model
            Clock.schedule_once(lambda dt: self._on_sindy_done(model))
        except Exception as e:
            Clock.schedule_once(lambda dt: self._show_error(str(e)))

    def _on_sindy_done(self, model):
        self._set_status(f"SINDy done: {model['n_terms']} terms")
        root = App.get_running_app().root
        sindy_tab = root.ids.sindy_tab
        eq_text = "\n".join(model["equations"])
        sindy_tab.ids.equation_label.text = eq_text
        sindy_tab.ids.sindy_info.text = (
            f"Terms: {model['n_terms']} | "
            f"Degree: {model['degree']} | "
            f"Threshold: {model['threshold']:.3f}")


class FlowTab(BoxLayout):
    pass


class SINDyTab(BoxLayout):
    pass


class VizTab(BoxLayout):
    def update_viz(self):
        app = App.get_running_app()
        if app.flow_result and app.flow_result["u"].shape[0] > 0:
            mode = self.ids.viz_mode.text
            self.ids.flow_canvas.set_data(
                app.flow_result["u"][0], app.flow_result["v"][0], mode)


class TRSindyMobile(BoxLayout):
    pass


# ---------------------------------------------------------------------
#  Main App
# ---------------------------------------------------------------------
class TRSindyApp(App):
    video_path = StringProperty("")
    first_frame = ObjectProperty(None)
    frame_w = NumericProperty(0)
    frame_h = NumericProperty(0)
    flow_result = ObjectProperty(None)
    sindy_model = ObjectProperty(None)

    def build(self):
        Builder.load_string(KV)
        return TRSindyMobile()

    def on_stop(self):
        pass


def main():
    """Entry point for the mobile app."""
    TRSindyApp().run()


if __name__ == "__main__":
    main()
