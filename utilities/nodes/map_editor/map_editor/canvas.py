"""Qt map canvas with image-space tools and world-space planner overlays."""
from copy import deepcopy
import math

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

from .overlays import LAYERS, plan_layers


class MapCanvas(QGraphicsView):
    changed = Signal()
    position = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.image_item = self.scene().addPixmap(QPixmap())
        self.setBackgroundBrush(QColor('#30343b'))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)
        self.document = None
        self.tool = 'Brush'
        self.material = 0
        self.brush_size = 12
        self.filled = True
        self.visible_layers = set(LAYERS)
        self.overlay_items = []
        self.start = None
        self.last = None
        self.buffer = None
        self.polygon = []
        self.pan_position = None
        self.preview = self.scene().addPath(QPainterPath(), QPen(QColor('#d946ef'), 1))
        self.preview.setZValue(20)

    def set_document(self, document):
        self.cancel_gesture()
        self.document = document
        self.refresh()
        self.fit_map()

    def set_tool(self, tool):
        self.cancel_gesture()
        self.tool = tool
        self.setCursor(Qt.CursorShape.OpenHandCursor if tool == 'Pan' else Qt.CursorShape.CrossCursor)

    def cancel_gesture(self):
        self.start = self.last = self.buffer = self.pan_position = None
        self.polygon = []
        self.preview.setPath(QPainterPath())
        if self.document is not None:
            self.refresh_image()

    def refresh_image(self, pixels=None):
        if self.document is None:
            return
        pixels = self.document.pixels if pixels is None else pixels
        h, w = pixels.shape
        image = QImage(pixels.data, w, h, pixels.strides[0], QImage.Format.Format_Grayscale8).copy()
        self.image_item.setPixmap(QPixmap.fromImage(image))
        self.setSceneRect(QRectF(0, 0, w, h))

    def refresh(self):
        self.refresh_image()
        self.refresh_overlays()

    def fit_map(self):
        if self.document is not None:
            self.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)

    def refresh_overlays(self):
        for item in self.overlay_items:
            self.scene().removeItem(item)
        self.overlay_items = []
        doc = self.document
        if doc is None:
            return
        if doc.plan_current:
            for label, points in plan_layers(doc.plan).items():
                if label not in self.visible_layers:
                    continue
                pixels = doc.world_to_pixel(points)
                pen = QPen(QColor(LAYERS[label]))
                pen.setWidthF(2)
                pen.setCosmetic(True)
                path = QPainterPath()
                if label == 'Track Boundaries':
                    # Bounds are serialized as individual spheres, right side then left side.
                    # Draw samples to avoid an artificial segment joining the two walls.
                    pen.setWidthF(1)
                    radius = max(0.5, 0.025 / float(doc.metadata['resolution']))
                    for x, y in pixels:
                        path.addEllipse(QPointF(float(x), float(y)), radius, radius)
                else:
                    path.moveTo(*map(float, pixels[0]))
                    for point in pixels[1:]:
                        path.lineTo(*map(float, point))
                    path.closeSubpath()
                item = self.scene().addPath(path, pen)
                item.setZValue(10)
                self.overlay_items.append(item)
        if 'initial_pose' in doc.metadata:
            x, y, yaw = doc.metadata['initial_pose']
            length = 0.6
            coords = doc.world_to_pixel([[x, y], [x + length * math.cos(yaw), y + length * math.sin(yaw)]])
            path = QPainterPath(QPointF(*coords[0]))
            path.lineTo(QPointF(*coords[1]))
            path.addEllipse(QPointF(*coords[0]), 2, 2)
            pen = QPen(QColor('#d946ef'), 3)
            pen.setCosmetic(True)
            item = self.scene().addPath(path, pen)
            item.setZValue(12)
            self.overlay_items.append(item)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        scale = self.transform().m11() * factor
        if 0.01 <= scale <= 100:
            self.scale(factor, factor)
        event.accept()

    def _point(self, event):
        pos = self.mapToScene(event.position().toPoint())
        h, w = self.document.pixels.shape
        return (max(0, min(w - 1, round(pos.x()))), max(0, min(h - 1, round(pos.y()))))

    def mousePressEvent(self, event):
        if self.document is None:
            return
        if event.button() == Qt.MouseButton.MiddleButton or (
                event.button() == Qt.MouseButton.LeftButton and self.tool == 'Pan'):
            self.pan_position = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            if self.tool == 'Polygon':
                self.finish_polygon()
            else:
                self.cancel_gesture()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self.image_item.boundingRect().contains(self.mapToScene(event.position().toPoint())):
            return
        point = self._point(event)
        if self.tool == 'Polygon':
            self.polygon.append(point)
            self._polygon_preview(point)
            return
        if self.tool == 'Remove component':
            occupancy = (255 - self.document.pixels.astype(np.float32)) / 255.
            mask = (occupancy > self.document.metadata['occupied_thresh']).astype(np.uint8)
            if mask[point[1], point[0]]:
                cv2.floodFill(mask, None, point, 2, flags=8)
                pixels = self.document.pixels.copy()
                pixels[mask == 2] = 255
                if self.document.commit(pixels):
                    self._committed()
            return
        self.start = self.last = point
        if self.tool in ('Brush', 'Eraser'):
            self.buffer = self.document.pixels.copy()
            self._stroke(point, point)
            self.refresh_image(self.buffer)

    def _stroke(self, start, end):
        color = 255 if self.tool == 'Eraser' else self.material
        cv2.line(self.buffer, start, end, color, self.brush_size, cv2.LINE_8)

    def mouseMoveEvent(self, event):
        if self.document is None:
            return
        if self.pan_position is not None:
            delta = event.position() - self.pan_position
            self.pan_position = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - round(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - round(delta.y()))
            return
        point = self._point(event)
        world = self.document.pixel_to_world(point)
        self.position.emit(f'Pixel {point[0]}, {point[1]}  |  Map {world[0]:.3f}, {world[1]:.3f} m')
        if self.polygon:
            self._polygon_preview(point)
        if self.start is None:
            return
        if self.buffer is not None:
            self._stroke(self.last, point)
            self.last = point
            self.refresh_image(self.buffer)
        else:
            pixels = self.document.pixels.copy()
            self._shape(pixels, point)
            self.refresh_image(pixels)

    def _shape(self, pixels, end):
        if self.tool == 'Line':
            cv2.line(pixels, self.start, end, self.material, self.brush_size)
        elif self.tool == 'Rectangle':
            cv2.rectangle(pixels, self.start, end, self.material, -1 if self.filled else self.brush_size)
        elif self.tool == 'Initial pose':
            path = QPainterPath(QPointF(*self.start))
            path.lineTo(QPointF(*end))
            self.preview.setPath(path)

    def mouseReleaseEvent(self, event):
        if self.pan_position is not None:
            self.pan_position = None
            self.setCursor(Qt.CursorShape.OpenHandCursor if self.tool == 'Pan' else Qt.CursorShape.CrossCursor)
            return
        if self.start is None or event.button() != Qt.MouseButton.LeftButton:
            return
        end = self._point(event)
        if self.tool == 'Initial pose':
            coords = self.document.pixel_to_world([self.start, end])
            delta = coords[1] - coords[0]
            yaw = math.atan2(delta[1], delta[0])
            metadata = deepcopy(self.document.metadata)
            metadata['initial_pose'] = [float(coords[0, 0]), float(coords[0, 1]), yaw]
            changed = self.document.commit(metadata=metadata)
        else:
            pixels = self.buffer if self.buffer is not None else self.document.pixels.copy()
            if self.buffer is not None:
                self._stroke(self.last, end)
            else:
                self._shape(pixels, end)
            changed = self.document.commit(pixels)
        self.cancel_gesture()
        if changed:
            self._committed()

    def _polygon_preview(self, point):
        path = QPainterPath(QPointF(*self.polygon[0]))
        for vertex in self.polygon[1:]:
            path.lineTo(QPointF(*vertex))
        path.lineTo(QPointF(*point))
        path.lineTo(QPointF(*self.polygon[0]))
        self.preview.setPath(path)

    def mouseDoubleClickEvent(self, event):
        if self.tool == 'Polygon' and event.button() == Qt.MouseButton.LeftButton:
            self.finish_polygon()
        else:
            super().mouseDoubleClickEvent(event)

    def finish_polygon(self):
        if len(self.polygon) >= 3:
            pixels = self.document.pixels.copy()
            vertices = np.asarray(self.polygon, np.int32)
            if self.filled:
                cv2.fillPoly(pixels, [vertices], self.material)
            else:
                cv2.polylines(pixels, [vertices], True, self.material, self.brush_size)
            changed = self.document.commit(pixels)
            self.cancel_gesture()
            if changed:
                self._committed()
        else:
            self.cancel_gesture()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancel_gesture()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.polygon:
            self.finish_polygon()
        else:
            super().keyPressEvent(event)

    def _committed(self):
        self.refresh()
        self.changed.emit()
