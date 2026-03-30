#!/usr/bin/env python3
"""Shared workflow stepper shell for multi-step dialogs."""

from __future__ import annotations

try:
    from PyQt6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )
except Exception:
    from PyQt5.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )


class WorkflowShell(QWidget):
    """Reusable stepper layout with a left workflow rail and right content area."""

    def __init__(self, step_titles, on_step_selected=None, parent=None):
        super().__init__(parent)
        self._step_titles = [str(title) for title in step_titles]
        self._on_step_selected = on_step_selected
        self._step_buttons = []

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        self.stepper_frame = QFrame()
        stepper_layout = QVBoxLayout(self.stepper_frame)
        stepper_layout.setContentsMargins(6, 6, 6, 6)
        stepper_layout.setSpacing(8)
        stepper_layout.addWidget(QLabel("Workflow"))

        for idx, title in enumerate(self._step_titles):
            button = QPushButton(f"{idx + 1}. {title}")
            button.setObjectName("workflowStepButton")
            button.setCheckable(True)
            button.setMinimumHeight(36)
            button.clicked.connect(lambda _checked=False, i=idx: self._handle_step_clicked(i))
            stepper_layout.addWidget(button)
            self._step_buttons.append(button)
        stepper_layout.addStretch(1)
        root_layout.addWidget(self.stepper_frame, 0)

        self.right_layout = QVBoxLayout()
        self.step_stack = QStackedWidget()
        self.right_layout.addWidget(self.step_stack, 1)
        root_layout.addLayout(self.right_layout, 1)

    @property
    def step_buttons(self):
        return self._step_buttons

    @property
    def step_titles(self):
        return tuple(self._step_titles)

    def step_count(self) -> int:
        return len(self._step_titles)

    def max_step_index(self) -> int:
        return max(0, len(self._step_titles) - 1)

    def add_step(self, page: QWidget) -> None:
        self.step_stack.addWidget(page)

    def set_current_step(self, step_index: int) -> int:
        if not self._step_titles:
            return 0
        step = max(0, min(int(step_index), self.max_step_index()))
        self.step_stack.setCurrentIndex(step)
        for idx, button in enumerate(self._step_buttons):
            is_current = idx == step
            button.setChecked(is_current)
            prefix = "▶ " if is_current else ""
            button.setText(f"{prefix}{idx + 1}. {self._step_titles[idx]}")
        return step

    def _handle_step_clicked(self, step_index: int) -> None:
        if self._on_step_selected is not None:
            self._on_step_selected(step_index)
            return
        self.set_current_step(step_index)
