from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .export import ExportJob
from .params import (
    PRESETS,
    OrbitCamera,
    PaintParams,
    camera_from_orbit,
    preset_params,
    render_hash,
)
from .preview import PreviewWorker
from .renderer import PaintRenderer


class PreviewCanvas(QWidget):
    camera_dragged = pyqtSignal(float, float)
    camera_wheel = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(560, 330)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)
        self._pixmap: Optional[QPixmap] = None
        self._last_mouse: Optional[QPoint] = None
        self._placeholder = "正在生成程序化漆面预览…"

    def set_image(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        contiguous = np.ascontiguousarray(image, dtype=np.uint8)
        qimage = QImage(
            contiguous.data,
            width,
            height,
            3 * width,
            QImage.Format_RGB888,
        ).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self.update()

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0b0e13"))
        if self._pixmap is None:
            painter.setPen(QColor("#7f8da3"))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
            painter.end()
            return

        target = self.rect()
        scaled = self._pixmap.scaled(
            target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        x = (target.width() - scaled.width()) // 2
        y = (target.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.setPen(QColor(225, 235, 245, 125))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(
            target.adjusted(14, 10, -14, -10),
            Qt.AlignLeft | Qt.AlignBottom,
            "拖拽旋转   ·   滚轮缩放",
        )
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._last_mouse = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if self._last_mouse is None:
            return
        delta = event.pos() - self._last_mouse
        self._last_mouse = event.pos()
        if delta.x() or delta.y():
            self.camera_dragged.emit(float(delta.x()), float(delta.y()))

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._last_mouse = None
            self.unsetCursor()

    def wheelEvent(self, event) -> None:
        self.camera_wheel.emit(int(event.angleDelta().y()))


class SliderRow(QWidget):
    value_changed = pyqtSignal(int)

    def __init__(
        self,
        title: str,
        minimum: int,
        maximum: int,
        value: int,
        divisor: float = 1.0,
        decimals: int = 0,
        suffix: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._divisor = float(divisor)
        self._decimals = int(decimals)
        self._suffix = suffix
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        self.title_label = QLabel(title)
        self.title_label.setFixedWidth(82)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(minimum), int(maximum))
        self.value_label = QLabel()
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setFixedWidth(56)
        layout.addWidget(self.title_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._on_value_changed)
        self.set_value(int(value))

    def _format(self, raw: int) -> str:
        value = raw / self._divisor
        return "{:.{d}f}{s}".format(
            value, d=self._decimals, s=self._suffix
        )

    def _on_value_changed(self, raw: int) -> None:
        self.value_label.setText(self._format(raw))
        self.value_changed.emit(int(raw))

    def set_value(self, raw: int) -> None:
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(int(raw))
        self.slider.blockSignals(blocked)
        self.value_label.setText(self._format(self.slider.value()))

    def value(self) -> int:
        return int(self.slider.value())


class ParameterPanel(QWidget):
    params_changed = pyqtSignal()
    preset_selected = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._syncing = False
        self._paint_type = "metallic"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        preset_group = QGroupBox("漆面预设")
        preset_layout = QVBoxLayout(preset_group)
        self.preset_combo = QComboBox()
        for name in PRESETS:
            self.preset_combo.addItem(name)
        self.preset_combo.addItem("自定义")
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        layout.addWidget(preset_group)

        color_group = QGroupBox("底色 RGB（sRGB）")
        color_layout = QVBoxLayout(color_group)
        self.red = SliderRow("R", 0, 255, 0)
        self.green = SliderRow("G", 0, 255, 0)
        self.blue = SliderRow("B", 0, 255, 0)
        color_layout.addWidget(self.red)
        color_layout.addWidget(self.green)
        color_layout.addWidget(self.blue)
        layout.addWidget(color_group)

        coat_group = QGroupBox("分层漆面参数")
        coat_layout = QVBoxLayout(coat_group)
        self.metallic = SliderRow("金属度", 0, 100, 0, suffix="%")
        self.roughness = SliderRow("粗糙度", 1, 100, 25, suffix="%")
        self.thickness = SliderRow("清漆厚度", 0, 400, 100, 100.0, 2, " μm")
        self.density = SliderRow("颗粒密度", 0, 100, 0, suffix="%")
        self.pearl_shift = SliderRow("珠光色相", -100, 100, 0, 100.0, 2)
        self.exposure = SliderRow("曝光", 5, 400, 130, 100.0, 2)
        for row in (
            self.metallic,
            self.roughness,
            self.thickness,
            self.density,
            self.pearl_shift,
            self.exposure,
        ):
            coat_layout.addWidget(row)
        layout.addWidget(coat_group)

        grain_group = QGroupBox("确定性颗粒")
        grain_layout = QFormLayout(grain_group)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2147483647)
        self.seed_spin.setValue(211)
        grain_layout.addRow("随机种子", self.seed_spin)
        grain_note = QLabel("1024 × 1024 法线扰动图\n参数变化时按需重建并缓存")
        grain_note.setWordWrap(True)
        grain_note.setStyleSheet("color:#8593a8;")
        grain_layout.addRow(grain_note)
        layout.addWidget(grain_group)

        output_group = QGroupBox("评审出图")
        output_layout = QVBoxLayout(output_group)
        self.output_label = QLabel()
        self.output_label.setWordWrap(True)
        self.output_label.setStyleSheet("color:#8593a8;")
        self.choose_button = QPushButton("选择输出目录")
        self.export_button = QPushButton("导出 1920 × 1080 PNG")
        self.cancel_button = QPushButton("取消在途出图")
        self.cancel_button.setEnabled(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.export_status = QLabel("就绪")
        self.export_status.setWordWrap(True)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.choose_button)
        output_layout.addWidget(self.export_button)
        output_layout.addWidget(self.cancel_button)
        output_layout.addWidget(self.progress_bar)
        output_layout.addWidget(self.export_status)
        layout.addWidget(output_group)
        layout.addStretch(1)

        for row in (
            self.red,
            self.green,
            self.blue,
            self.metallic,
            self.roughness,
            self.thickness,
            self.density,
            self.pearl_shift,
            self.exposure,
        ):
            row.value_changed.connect(self._on_any_change)
        self.seed_spin.valueChanged.connect(self._on_any_change)

    def _on_preset_changed(self, name: str) -> None:
        if self._syncing or name == "自定义":
            return
        self.preset_selected.emit(name)

    def _on_any_change(self, _value=None) -> None:
        if not self._syncing:
            self.params_changed.emit()

    def set_params(self, params: PaintParams, preset_name: str) -> None:
        self._syncing = True
        try:
            self._paint_type = params.paint_type
            self.red.set_value(int(round(params.base_color[0] * 255)))
            self.green.set_value(int(round(params.base_color[1] * 255)))
            self.blue.set_value(int(round(params.base_color[2] * 255)))
            self.metallic.set_value(int(round(params.metallic * 100)))
            self.roughness.set_value(int(round(params.roughness * 100)))
            self.thickness.set_value(int(round(params.clearcoat_thickness * 100)))
            self.density.set_value(int(round(params.grain_density * 100)))
            self.pearl_shift.set_value(int(round(params.pearl_shift * 100)))
            self.exposure.set_value(int(round(params.exposure * 100)))
            self.seed_spin.setValue(int(params.grain_seed))
            self.show_preset(preset_name)
        finally:
            self._syncing = False

    def show_preset(self, name: str) -> None:
        previous = self._syncing
        self._syncing = True
        index = self.preset_combo.findText(name)
        if index < 0:
            self.preset_combo.addItem(name)
            index = self.preset_combo.findText(name)
        self.preset_combo.setCurrentIndex(index)
        self._syncing = previous

    def make_params(self) -> PaintParams:
        color = (
            self.red.value() / 255.0,
            self.green.value() / 255.0,
            self.blue.value() / 255.0,
        )
        return PaintParams(
            base_color=color,
            metallic=self.metallic.value() / 100.0,
            roughness=self.roughness.value() / 100.0,
            clearcoat_thickness=self.thickness.value() / 100.0,
            grain_density=self.density.value() / 100.0,
            pearl_shift=self.pearl_shift.value() / 100.0,
            exposure=self.exposure.value() / 100.0,
            grain_seed=self.seed_spin.value(),
            paint_type=self._paint_type,
        ).normalized()

    def set_export_running(self, running: bool) -> None:
        self.export_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.choose_button.setEnabled(not running)
        self.progress_bar.setVisible(running)
        if running:
            self.progress_bar.setValue(0)

    def set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.preset_combo,
            self.red,
            self.green,
            self.blue,
            self.metallic,
            self.roughness,
            self.thickness,
            self.density,
            self.pearl_shift,
            self.exposure,
            self.seed_spin,
        ):
            widget.setEnabled(enabled)


class PaintReviewWindow(QMainWindow):
    PREVIEW_WIDTH = 640
    PREVIEW_HEIGHT = 360

    def __init__(self) -> None:
        super().__init__()
        self.params = preset_params("金属漆")
        self.preset_name = "金属漆"
        self.camera = OrbitCamera()
        self.output_dir = Path(__file__).resolve().parent.parent / "output_review"
        self.renderer = PaintRenderer(subdivisions_u=96, subdivisions_v=96)
        self._preview_token = 0
        self._preview_ms: Optional[float] = None
        self._export_thread: Optional[ExportJob] = None
        self._pending_preset: Optional[str] = None
        self._closing = False
        self._last_log_path = ""

        self.setWindowTitle("漆面分层评审渲染器")
        self.resize(1360, 820)
        self.setMinimumSize(1040, 660)
        self._build_ui()
        self.panel.set_params(self.params, self.preset_name)
        self._update_output_label()
        self._update_title()
        self._update_hash_label()

        self.preview_worker = PreviewWorker(self.renderer, self)
        self.preview_worker.preview_ready.connect(self._on_preview_ready)
        self.preview_worker.failed.connect(self._on_preview_failed)
        self.preview_worker.start()
        self._request_preview()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.canvas = PreviewCanvas()
        self.canvas.camera_dragged.connect(self._on_camera_dragged)
        self.canvas.camera_wheel.connect(self._on_camera_wheel)
        root.addWidget(self.canvas, 1)

        self.panel = ParameterPanel()
        self.panel.params_changed.connect(self._on_parameter_changed)
        self.panel.preset_selected.connect(self._on_preset_selected)
        self.panel.choose_button.clicked.connect(self._choose_output_dir)
        self.panel.export_button.clicked.connect(self._start_export)
        self.panel.cancel_button.clicked.connect(self._cancel_export)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        scroll.setMinimumWidth(370)
        scroll.setMaximumWidth(405)
        scroll.setStyleSheet(
            "QScrollArea { border-left: 1px solid #252d39; background:#151b24; }"
        )
        root.addWidget(scroll)
        self.setCentralWidget(central)

        status = QStatusBar()
        status.setSizeGripEnabled(False)
        self.setStatusBar(status)
        self.status_label = QLabel("就绪")
        self.hash_label = QLabel()
        self.hash_label.setStyleSheet("color:#8d9bb0;")
        status.addWidget(self.status_label, 1)
        status.addPermanentWidget(self.hash_label)

    def _update_title(self) -> None:
        timing = "--" if self._preview_ms is None else "{:.0f}".format(self._preview_ms)
        self.setWindowTitle(
            "漆面分层评审 - {} | 预览 {} ms".format(self.preset_name, timing)
        )

    def _update_hash_label(self) -> None:
        material = self.params.material_hash()[:12]
        render = render_hash(
            self.params, self.camera, 1920, 1080
        )[:12]
        self.hash_label.setText("材料 {}  |  出图 {}".format(material, render))

    def _update_output_label(self) -> None:
        self.panel.output_label.setText("输出与日志目录：\n{}".format(self.output_dir))

    def _request_preview(self) -> None:
        if self.renderer.closed:
            return
        self._preview_token += 1
        self.canvas.set_placeholder("参数已变更，正在重绘…")
        self.preview_worker.submit(
            self._preview_token,
            self.params,
            self.camera,
            self.PREVIEW_WIDTH,
            self.PREVIEW_HEIGHT,
        )

    def _on_preview_ready(
        self, token: int, image: np.ndarray, stats: dict
    ) -> None:
        if token != self._preview_token:
            return
        self.canvas.set_image(image)
        self._preview_ms = float(stats["render_ms"])
        self._update_title()
        self._update_hash_label()
        if self._export_thread is None or not self._export_thread.isRunning():
            self.status_label.setText(
                "预览 {}×{} | 覆盖 {} px | {} 三角形 | 环境预过滤 {} 级".format(
                    stats["width"],
                    stats["height"],
                    stats["covered_pixels"],
                    stats["triangle_count"],
                    stats["env_levels"],
                )
            )

    def _on_preview_failed(self, message: str) -> None:
        self.status_label.setText("预览失败：{}".format(message))

    def _on_camera_dragged(self, dx: float, dy: float) -> None:
        self.camera = camera_from_orbit(
            yaw_deg=self.camera.yaw_deg - dx * 0.32,
            elevation_deg=self.camera.elevation_deg - dy * 0.26,
            distance=self.camera.distance,
            fov_deg=self.camera.fov_deg,
        )
        self._request_preview()
        self._update_hash_label()

    def _on_camera_wheel(self, delta: int) -> None:
        factor = math.exp(-float(delta) / 120.0 * 0.13)
        self.camera = camera_from_orbit(
            yaw_deg=self.camera.yaw_deg,
            elevation_deg=self.camera.elevation_deg,
            distance=self.camera.distance * factor,
            fov_deg=self.camera.fov_deg,
        )
        self._request_preview()
        self._update_hash_label()

    def _on_parameter_changed(self) -> None:
        self.params = self.panel.make_params()
        self.preset_name = "自定义"
        self.panel.show_preset("自定义")
        self._update_hash_label()
        self._request_preview()

    def _on_preset_selected(self, name: str) -> None:
        if name == self.preset_name or name == "自定义":
            return
        if self._export_thread is not None and self._export_thread.isRunning():
            self._pending_preset = name
            self.panel.set_controls_enabled(False)
            self._cancel_export("切换预设：正在取消在途出图并等待工作线程结束…")
            return
        self._apply_preset(name, notice=None)

    def _apply_preset(self, name: str, notice: Optional[str]) -> None:
        self._stop_preview_and_release_renderer()
        self.renderer = PaintRenderer(subdivisions_u=96, subdivisions_v=96)
        self.preview_worker = PreviewWorker(self.renderer, self)
        self.preview_worker.preview_ready.connect(self._on_preview_ready)
        self.preview_worker.failed.connect(self._on_preview_failed)
        self.preview_worker.start()
        self.params = preset_params(name)
        self.preset_name = name
        self.panel.set_params(self.params, name)
        self.panel.set_controls_enabled(True)
        self._preview_ms = None
        self._update_title()
        self._update_hash_label()
        if notice:
            self.status_label.setText(notice)
        self._request_preview()

    def _stop_preview_and_release_renderer(self) -> None:
        worker = getattr(self, "preview_worker", None)
        if worker is not None:
            worker.stop_and_wait()
            worker.deleteLater()
            self.preview_worker = None
        if not self.renderer.closed:
            self.renderer.close()

    def _choose_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "选择评审图与日志输出目录", str(self.output_dir)
        )
        if selected:
            self.output_dir = Path(selected)
            self._update_output_label()

    def _start_export(self) -> None:
        if self._export_thread is not None and self._export_thread.isRunning():
            return
        if self._pending_preset is not None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._last_log_path = str(self.output_dir / "render_log.jsonl")
        thread = ExportJob(
            self.renderer,
            self.params,
            self.camera,
            self.output_dir,
            self.preset_name,
            width=1920,
            height=1080,
            parent=self,
        )
        thread.progress.connect(self._on_export_progress)
        thread.completed.connect(self._on_export_completed)
        thread.cancelled.connect(self._on_export_cancelled)
        thread.failed.connect(self._on_export_failed)
        thread.finished.connect(self._on_export_thread_finished)
        self._export_thread = thread
        self.panel.set_export_running(True)
        self.panel.export_status.setStyleSheet("color:#8fbfff;")
        self.panel.export_status.setText("准备离屏渲染 1920 × 1080…")
        self.status_label.setText("离屏出图进行中，界面与预览保持可用")
        thread.start()

    def _cancel_export(self, reason: Optional[str] = None) -> None:
        thread = self._export_thread
        if thread is None or not thread.isRunning():
            return
        thread.request_cancel()
        self.panel.cancel_button.setEnabled(False)
        message = reason or "正在取消出图…"
        self.panel.export_status.setStyleSheet("color:#ffb86b;")
        self.panel.export_status.setText(message)
        self.status_label.setText(message)

    def _on_export_progress(self, value: int, message: str) -> None:
        self.panel.progress_bar.setValue(max(0, min(100, value)))
        self.panel.export_status.setText(message)

    def _on_export_completed(self, record: dict) -> None:
        self.panel.progress_bar.setValue(100)
        self.panel.export_status.setStyleSheet("color:#7de2a8;")
        self.panel.export_status.setText(
            "出图完成：{}\n渲染 {:.0f} ms | 日志 {}".format(
                record["filename"],
                float(record["stats"]["render_ms"]),
                Path(self._last_log_path).name,
            )
        )
        self.status_label.setText(
            "出图完成并写入日志 | 参数哈希 {}".format(record["params_hash"][:16])
        )
        self._last_log_path = str(Path(record["output_path"]).parent / "render_log.jsonl")
        QApplication.alert(self)

    def _on_export_cancelled(self) -> None:
        self.panel.export_status.setStyleSheet("color:#ffb86b;")
        self.panel.export_status.setText("出图已取消；工作线程已结束")
        self.status_label.setText("在途出图已取消，未保留后台线程")

    def _on_export_failed(self, message: str) -> None:
        self.panel.export_status.setStyleSheet("color:#ff7b7b;")
        self.panel.export_status.setText("出图失败：{}".format(message))
        self.status_label.setText("出图失败：{}".format(message))

    def _on_export_thread_finished(self) -> None:
        thread = self._export_thread
        if thread is not None:
            thread.wait()
            self._export_thread = None
            thread.deleteLater()
        self.panel.set_export_running(False)
        self.panel.progress_bar.setVisible(False)
        self.panel.cancel_button.setEnabled(False)
        self.panel.export_button.setEnabled(True)

        if self._closing:
            self._release_all_resources()
            self._closing = False
            self.close()
            return
        if self._pending_preset is not None:
            target = self._pending_preset
            self._pending_preset = None
            self._apply_preset(
                target,
                notice="在途出图已取消或完成；工作线程结束后已切换至“{}”".format(
                    target
                ),
            )

    def closeEvent(self, event) -> None:
        thread = self._export_thread
        if thread is not None and thread.isRunning():
            self._closing = True
            self._pending_preset = None
            self._cancel_export("关闭窗口：正在取消在途出图并等待工作线程结束…")
            event.ignore()
            return
        if thread is not None:
            thread.wait()
            self._export_thread = None
            thread.deleteLater()
        self._release_all_resources()
        event.accept()

    def _release_all_resources(self) -> None:
        self._stop_preview_and_release_renderer()


def create_window() -> PaintReviewWindow:
    return PaintReviewWindow()

