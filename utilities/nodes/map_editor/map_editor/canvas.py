"""Qt map canvas with image-space tools and world-space planner overlays."""
from copy import deepcopy
import math

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

from .overlays import LAYERS, plan_layers
from .sectors import COLORS


class MapCanvas(QGraphicsView):
    changed = Signal()
    position = Signal(str)
    sector_selected = Signal(int)
    sector_add = Signal(int)
    sector_delete = Signal(int)
    sector_move = Signal(int, int)

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
        self.sector_kind = 'speed'
        self.sector_action = 0
        self.sector_speed_preview = True
        self.sector_selected_index = 0
        self.sector_drag = None
        self.sectors_paused = False
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
        self.refresh_overlays()

    def cancel_gesture(self):
        self.start = self.last = self.buffer = self.pan_position = None
        self.sector_drag = None
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
            if self.tool == 'Sectors':
                self._sector_overlays()
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

    def _sector_overlays(self):
        doc = self.document
        kind = self.sector_kind
        if doc.sector_issues(kind, check_binding=False):
            return
        config = doc.sectors[kind]
        pixels = doc.world_to_pixel(plan_layers(doc.plan)['Global Raceline'])
        unconfirmed = bool(doc.sector_issues(kind))
        for i, section in enumerate(config.sections):
            color = COLORS[i % len(COLORS)] if kind == 'speed' else (
                '#35df85' if section['ot_flag'] else '#87929d')
            path = QPainterPath(QPointF(*pixels[section['start']]))
            for point in pixels[section['start'] + 1:min(section['end'] + 2, len(pixels))]:
                path.lineTo(QPointF(*point))
            pen = QPen(QColor(color), 8 if i == self.sector_selected_index else 6)
            pen.setCosmetic(True)
            item = self.scene().addPath(path, pen)
            item.setZValue(14)
            item.setOpacity(.65 if unconfirmed else 1.)
            self.overlay_items.append(item)
        if kind == 'speed' and self.sector_speed_preview:
            factors = config.speed_factors(len(pixels))
            speeds = np.asarray([p['vx_mps'] for p in doc.plan['global_traj_wpnts_iqp']['wpnts']]) * factors
            maximum = max(float(np.max([p['vx_mps'] for p in
                                       doc.plan['global_traj_wpnts_iqp']['wpnts']])), .001)
            # Group speed colors into paths to keep large tracks responsive.
            bins = np.minimum(31, np.maximum(0, (speeds / maximum * 31).astype(int)))
            for level in np.unique(bins[:-1]):
                path = QPainterPath()
                for index in np.flatnonzero(bins[:-1] == level):
                    path.moveTo(QPointF(*pixels[index]))
                    path.lineTo(QPointF(*pixels[index + 1]))
                color = QColor.fromHsvF((1 - float(level) / 31) * .66, .9, 1.)
                pen = QPen(color, 3)
                pen.setCosmetic(True)
                item = self.scene().addPath(path, pen)
                item.setZValue(15)
                self.overlay_items.append(item)
        for i, section in enumerate(config.sections):
            point = pixels[section['start']]
            pen = QPen(QColor('#ffffff' if i == self.sector_selected_index else '#202830'), 2)
            pen.setCosmetic(True)
            item = self.scene().addEllipse(-8, -8, 16, 16, pen, QColor('#f6be45'))
            item.setPos(float(point[0]), float(point[1]))
            item.setFlag(item.GraphicsItemFlag.ItemIgnoresTransformations)
            item.setZValue(16)
            item.setToolTip(f"Section {i}, start waypoint {section['start']}" + (' (fixed)' if i == 0 else ''))
            self.overlay_items.append(item)
            label = self.scene().addSimpleText(str(i))
            label.setBrush(QColor('#202830'))
            label.setFlag(label.GraphicsItemFlag.ItemIgnoresTransformations)
            label.setPos(float(point[0]), float(point[1]))
            label.setTransform(QTransform.fromTranslate(-label.boundingRect().width() / 2,
                                                         -label.boundingRect().height() / 2))
            label.setZValue(17)
            self.overlay_items.append(label)

    def _sector_hit(self, event, boundaries=False):
        doc = self.document
        if not doc.plan_current or doc.sector_issues(self.sector_kind, check_binding=False):
            return None
        pixels = doc.world_to_pixel(plan_layers(doc.plan)['Global Raceline'])
        indices = ([section['start'] for section in doc.sectors[self.sector_kind].sections]
                   if boundaries else np.arange(len(pixels)))
        pos = self.mapToScene(event.position().toPoint())
        distances = np.linalg.norm(pixels[indices] - [pos.x(), pos.y()], axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] * self.transform().m11() > 14:
            return None
        return nearest if boundaries else int(indices[nearest])

    def _sector_press(self, event):
        if self.sectors_paused or not self.document.plan_current or self.document.sector_issues(self.sector_kind):
            return
        boundary = self._sector_hit(event, boundaries=True)
        if event.button() == Qt.MouseButton.RightButton or self.sector_action == 2:
            if boundary is not None and boundary > 0:
                self.sector_delete.emit(boundary)
            return
        index = self._sector_hit(event)
        if index is None:
            return
        if self.sector_action == 1:
            self.sector_add.emit(index)
        else:
            section = (boundary if boundary is not None else
                       self.document.sectors[self.sector_kind].section_at(index))
            self.sector_selected.emit(section)
            if boundary is not None and boundary > 0:
                self.sector_drag = (boundary, index)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        scale = self.transform().m11() * factor
        if 0.01 <= scale <= 100:
            self.scale(factor, factor)
            if self.tool == 'Sectors':
                self.refresh_overlays()
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
        if self.tool == 'Sectors':
            if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                self._sector_press(event)
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
        if self.tool == 'Sectors':
            index = self._sector_hit(event)
            if index is not None:
                text = f'Global Waypoint {index}'
                if self.sector_kind == 'speed':
                    factors = self.document.sectors['speed'].speed_factors(
                        len(self.document.plan['global_traj_wpnts_iqp']['wpnts']))
                    speed = self.document.plan['global_traj_wpnts_iqp']['wpnts'][index]['vx_mps']
                    text += f' | {speed:.3f} → {speed * factors[index]:.3f} m/s (×{factors[index]:.3f})'
                self.position.emit(text)
                if self.sector_drag is not None:
                    section = self.sector_drag[0]
                    self.sector_drag = (section, index)
                    pixels = self.document.world_to_pixel(plan_layers(self.document.plan)['Global Raceline'])
                    radius = 9 / max(self.transform().m11(), .01)
                    path = QPainterPath()
                    path.addEllipse(QPointF(*pixels[index]), radius, radius)
                    self.preview.setPath(path)
            return
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
        if self.tool == 'Sectors':
            if self.sector_drag is not None and event.button() == Qt.MouseButton.LeftButton:
                section, index = self.sector_drag
                nearest = self._sector_hit(event)
                if nearest is not None:
                    index = nearest
                self.cancel_gesture()
                if not self.sectors_paused:
                    self.sector_move.emit(section, index)
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
