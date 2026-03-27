#!/usr/bin/env python3
import atexit
import faulthandler
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from PyQt6.QtCore import Qt, QSize, QTimer, qInstallMessageHandler
    from PyQt6.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QDialog,
        QFrame,
        QGridLayout,
        QFileDialog,
        QCheckBox,
        QGroupBox,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QComboBox,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QSplashScreen,
        QSplitter,
        QStackedWidget,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    from PyQt6.QtGui import QAction, QIcon, QFontMetrics, QPixmap, QColor
    QT_LIB = 6
except ImportError:
    from PyQt5.QtCore import Qt, QSize, QTimer, qInstallMessageHandler
    from PyQt5.QtWidgets import (
        QAction,
        QApplication,
        QAbstractItemView,
        QDialog,
        QFrame,
        QGridLayout,
        QFileDialog,
        QCheckBox,
        QGroupBox,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QComboBox,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QSplashScreen,
        QSplitter,
        QStackedWidget,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    from PyQt5.QtGui import QIcon, QFontMetrics, QPixmap, QColor
    QT_LIB = 5

_DIAGNOSTIC_LOG_HANDLE = None
_QT_MESSAGE_HANDLER = None


def _diagnostic_log_path() -> Path:
    raw = str(os.getenv("MRSI_VIEWER_DIAGNOSTICS_LOG", "")).strip()
    if raw:
        return Path(raw).expanduser()
    return Path("/tmp/mrsi_viewer_diagnostics.log")


def _write_diagnostic_line(text: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {text}"
    try:
        sys.__stderr__.write(line + "\n")
        sys.__stderr__.flush()
    except Exception:
        pass
    global _DIAGNOSTIC_LOG_HANDLE
    if _DIAGNOSTIC_LOG_HANDLE is not None:
        try:
            _DIAGNOSTIC_LOG_HANDLE.write(line + "\n")
            _DIAGNOSTIC_LOG_HANDLE.flush()
        except Exception:
            pass


def _install_runtime_diagnostics() -> Path:
    global _DIAGNOSTIC_LOG_HANDLE, _QT_MESSAGE_HANDLER

    log_path = _diagnostic_log_path()
    os.environ.setdefault("MRSI_VIEWER_DIAGNOSTICS_LOG", str(log_path))
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        _DIAGNOSTIC_LOG_HANDLE = open(log_path, "a", encoding="utf-8", buffering=1)
    except Exception:
        _DIAGNOSTIC_LOG_HANDLE = None

    _write_diagnostic_line(
        f"=== Launch pid={os.getpid()} argv={sys.argv!r} qt_lib={QT_LIB} python={sys.version.split()[0]} ==="
    )

    try:
        faulthandler.enable(file=_DIAGNOSTIC_LOG_HANDLE or sys.__stderr__, all_threads=True)
        _write_diagnostic_line("faulthandler enabled")
    except Exception as exc:
        _write_diagnostic_line(f"failed to enable faulthandler: {exc}")

    default_excepthook = sys.excepthook

    def _diagnostic_excepthook(exc_type, exc_value, exc_tb):
        _write_diagnostic_line(
            "Unhandled Python exception:\n" + "".join(traceback.format_exception(exc_type, exc_value, exc_tb)).rstrip()
        )
        try:
            default_excepthook(exc_type, exc_value, exc_tb)
        except Exception:
            pass

    sys.excepthook = _diagnostic_excepthook

    try:
        def _qt_message_handler(msg_type, context, message):
            file_name = getattr(context, "file", "") if context is not None else ""
            line_no = getattr(context, "line", 0) if context is not None else 0
            function_name = getattr(context, "function", "") if context is not None else ""
            mode_name = getattr(msg_type, "name", str(msg_type))
            _write_diagnostic_line(
                f"[QT {mode_name}] {message} ({file_name}:{line_no} {function_name})"
            )

        _QT_MESSAGE_HANDLER = _qt_message_handler
        qInstallMessageHandler(_QT_MESSAGE_HANDLER)
        _write_diagnostic_line("Qt message handler installed")
    except Exception as exc:
        _write_diagnostic_line(f"failed to install Qt message handler: {exc}")

    def _on_exit():
        _write_diagnostic_line("Process exiting")

    atexit.register(_on_exit)
    _write_diagnostic_line(f"diagnostic log path: {log_path}")
    return log_path

try:
    if QT_LIB == 6:
        from PyQt6.QtGui import QPainter
        from PyQt6.QtSvg import QSvgRenderer
    else:
        from PyQt5.QtGui import QPainter
        from PyQt5.QtSvg import QSvgRenderer
except Exception:
    QPainter = None
    QSvgRenderer = None

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
import matplotlib.transforms as mtransforms
try:
    from matplotlib.widgets import RectangleSelector
except Exception:
    RectangleSelector = None

from mrsitoolbox.graphplot.simmatrix import SimMatrixPlot
from mrsitoolbox.graphplot.colorbar import ColorBar
from mrsitoolbox.graphplot import colorbar as colorbar_module
from mrsitoolbox.connectomics.nettools import NetTools

nettools = NetTools()
# Ensure Qt can locate platform plugins when installed via pip wheels.
if QT_LIB == 6:
    try:
        import PyQt6

        qt_plugins = Path(PyQt6.__file__).resolve().parent / "Qt6" / "plugins"
        os.environ.setdefault("QT_PLUGIN_PATH", str(qt_plugins))
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(qt_plugins / "platforms"))
    except Exception:
        pass
else:
    try:
        import PyQt5

        qt5_base = Path(PyQt5.__file__).resolve().parent
        qt_plugins = qt5_base / "Qt5" / "plugins"
        if not qt_plugins.exists():
            qt_plugins = qt5_base / "Qt" / "plugins"
        os.environ.setdefault("QT_PLUGIN_PATH", str(qt_plugins))
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(qt_plugins / "platforms"))
    except Exception:
        pass

DEFAULT_COLORMAP = "plasma"
DEFAULT_GRADIENT_COLORMAP = "spectrum_fsl"


def _first_env_value(*names: str) -> str:
    for name in names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return ""


_MATLAB_FROM_ENV = _first_env_value(
    "MRSI_MATLAB_CMD",
    "MATLAB_CMD",
    "MATLAB_EXECUTABLE",
)
_MATLAB_FROM_PATH = shutil.which("matlab")
_MATLAB_DEFAULT_CANDIDATE = _MATLAB_FROM_ENV or _MATLAB_FROM_PATH or ""
if _MATLAB_DEFAULT_CANDIDATE:
    _matlab_path = Path(_MATLAB_DEFAULT_CANDIDATE).expanduser()
    DEFAULT_MATLAB_CMD = str(_matlab_path.resolve()) if _matlab_path.exists() else str(_matlab_path)
else:
    DEFAULT_MATLAB_CMD = ""

_NBS_FROM_ENV = _first_env_value(
    "MRSI_NBS_PATH",
    "MATLAB_NBS_PATH",
    "NBS_PATH",
)
if _NBS_FROM_ENV:
    _nbs_path = Path(_NBS_FROM_ENV).expanduser()
    DEFAULT_MATLAB_NBS_PATH = str(_nbs_path.resolve()) if _nbs_path.exists() else str(_nbs_path)
else:
    DEFAULT_MATLAB_NBS_PATH = ""
_CONFIG_HOME = Path(os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config"))).expanduser()
CONFIG_PATH = _CONFIG_HOME / "mrsi_viewer" / "connectome_viewer.json"
COLORMAPS = [
    "plasma",
    "viridis",
    "magma",
    "inferno",
    "cividis",
    "turbo",
    "jet",
]
COLORBAR_BARS = [
    "blueblackred",
    "redblackblue",
    "mask_alpha",
    "bluewhitered",
    "redwhiteblue",
    "wg",
    "bbo",
    "bo",
    "obb",
    "bwo",
    "bwg",
    "bwTo",
    "owb",
    "spectrum_fsl",
    "random9",
]

PARCEL_LABEL_KEYS = ("parcel_labels_group", "parcel_labels_group.npy")
PARCEL_NAME_KEYS = ("parcel_names_group", "parcel_names_group.npy")
CMAPS_DIR = Path(colorbar_module.__file__).with_name("cmaps")
ROOTDIR = Path(__file__).resolve().parent
DEFAULT_PARCELLATION_DIR = ROOTDIR / "data"
APP_ICON_CANDIDATES = (
    ROOTDIR / "icons" / "conviewer.png",
    ROOTDIR / "icons" / "conviewer_2.png",
)

if QT_LIB == 6:
    Qt.Horizontal = Qt.Orientation.Horizontal
    Qt.Vertical = Qt.Orientation.Vertical
    Qt.Checked = Qt.CheckState.Checked
    Qt.Unchecked = Qt.CheckState.Unchecked
    Qt.ElideRight = Qt.TextElideMode.ElideRight


def _user_role():
    return getattr(Qt, "UserRole", getattr(Qt.ItemDataRole, "UserRole"))


USER_ROLE = _user_role()


def _connectivity_file_metadata(path: Path) -> dict:
    base_name = Path(path).name
    results = {}
    patterns = {
        "atlas": r"atlas-([^_]+)",
        "scale": r"scale(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, base_name)
        results[key] = match.group(1) if match else None
    if results["scale"] is not None:
        results["scale"] = int(results["scale"])
    return results


def _connectivity_atlas_tag(path: Path) -> str:
    meta = _connectivity_file_metadata(Path(path))
    atlas = str(meta.get("atlas") or "").strip()
    scale = meta.get("scale")
    if atlas and scale is not None and f"scale{scale}" not in atlas:
        return f"{atlas}_scale{scale}"
    if atlas:
        return atlas
    return "unknown"


def _connectivity_modalities_from_path(path: Path):
    found = []
    seen = set()
    texts = (Path(path).name.lower(), str(path).lower())
    for text in texts:
        for match in re.findall(r"connectivity[_-]([a-z0-9]+)", text):
            modality = str(match).strip().lower()
            if not modality or modality in seen:
                continue
            seen.add(modality)
            found.append(modality)
    if found:
        return found

    lowered = str(path).lower().replace("\\", "/")
    for modality in ("mrsi", "func", "dwi"):
        if f"/{modality}/" in lowered and modality not in seen:
            seen.add(modality)
            found.append(modality)
    return found


def _infer_modality_from_path(path: Path) -> str:
    modalities = _connectivity_modalities_from_path(path)
    if modalities:
        return modalities[0]
    return "connectivity"


def _dialog_accepted_code():
    accepted = getattr(QDialog, "Accepted", None)
    if accepted is not None:
        return accepted
    dialog_code = getattr(QDialog, "DialogCode", None)
    if dialog_code is not None and hasattr(dialog_code, "Accepted"):
        return dialog_code.Accepted
    return 1


def _is_enabled_flag():
    return getattr(Qt, "ItemIsEnabled", getattr(Qt.ItemFlag, "ItemIsEnabled"))


def _is_user_checkable_flag():
    return getattr(Qt, "ItemIsUserCheckable", getattr(Qt.ItemFlag, "ItemIsUserCheckable"))


def _is_selectable_flag():
    return getattr(Qt, "ItemIsSelectable", getattr(Qt.ItemFlag, "ItemIsSelectable"))


def _is_editable_flag():
    return getattr(Qt, "ItemIsEditable", getattr(Qt.ItemFlag, "ItemIsEditable"))


def _align_right_flag():
    return getattr(Qt, "AlignRight", getattr(Qt.AlignmentFlag, "AlignRight"))


def _qheader_resize_mode(name: str):
    resize_mode = getattr(QHeaderView, "ResizeMode", None)
    if resize_mode is not None and hasattr(resize_mode, name):
        return getattr(resize_mode, name)
    return getattr(QHeaderView, name)


def _set_header_resize_mode(header, section: int, mode_name: str) -> None:
    header.setSectionResizeMode(section, _qheader_resize_mode(mode_name))


def _load_app_icon() -> QIcon:
    for icon_path in APP_ICON_CANDIDATES:
        if not icon_path.exists():
            continue
        icon = QIcon(str(icon_path))
        if not icon.isNull():
            return icon
    return QIcon()


def _is_valid_matrix_shape(shape) -> bool:
    if len(shape) == 2:
        return shape[0] == shape[1]
    if len(shape) == 3:
        a, b, c = shape
        return (a == b != c) or (a == c != b) or (b == c != a)
    return False


def _average_to_square(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 2:
        return matrix
    if matrix.ndim != 3:
        return matrix
    a, b, c = matrix.shape
    if a == b != c:
        return matrix.mean(axis=2)
    if a == c != b:
        return matrix.mean(axis=1)
    if b == c != a:
        return matrix.mean(axis=0)
    return matrix


def _load_matrix_from_npz(path: Path, key: str, average: bool = True) -> np.ndarray:
    with np.load(path, allow_pickle=True) as npz:
        if key not in npz:
            raise KeyError(f"Key '{key}' not found")
        matrix = npz[key]
    if average and matrix.ndim == 3:
        matrix = _average_to_square(matrix)
    return matrix


def _load_parcel_metadata(path: Path):
    labels = None
    names = None
    with np.load(path, allow_pickle=True) as npz:
        for key in PARCEL_LABEL_KEYS:
            if key in npz:
                labels = npz[key]
                break
        for key in PARCEL_NAME_KEYS:
            if key in npz:
                names = npz[key]
                break
    if labels is None:
        labels_path = path.with_name("parcel_labels_group.npy")
        if labels_path.exists():
            labels = np.load(labels_path, allow_pickle=True)
    if names is None:
        names_path = path.with_name("parcel_names_group.npy")
        if names_path.exists():
            names = np.load(names_path, allow_pickle=True)
    return labels, names


def _to_string_list(values):
    if values is None:
        return None
    try:
        flat = values.tolist()
    except Exception:
        flat = values
    if isinstance(flat, list):
        return [str(v) for v in flat]
    return [str(flat)]


def _load_covars_info(path: Path):
    try:
        with np.load(path, allow_pickle=True) as npz:
            if "covars" not in npz:
                return None
            covars = npz["covars"]
    except Exception:
        return None
    df = None
    if pd is not None:
        try:
            df = pd.DataFrame.from_records(covars)
        except Exception:
            df = None
    columns = list(covars.dtype.names) if getattr(covars.dtype, "names", None) else []
    return {"data": covars, "df": df, "columns": columns}


def _load_group_value(path: Path, index: int):
    try:
        with np.load(path, allow_pickle=True) as npz:
            if "group" not in npz:
                return None
            group_data = npz["group"]
    except Exception:
        return None
    if np.isscalar(group_data):
        return str(group_data)
    group_arr = np.asarray(group_data)
    if group_arr.ndim == 0:
        return str(group_arr.item())
    if index < 0 or index >= len(group_arr):
        return None
    return str(group_arr[index])


def _covars_columns(info):
    if info is None:
        return []
    df = info.get("df")
    if df is not None:
        return list(df.columns)
    return info.get("columns", [])


def _covars_series(info, name):
    if info is None:
        return None
    df = info.get("df")
    if df is not None:
        return df[name].to_numpy()
    data = info.get("data")
    if data is None or data.dtype.names is None:
        return None
    return data[name]


def _covars_to_rows(info):
    if info is None:
        return [], []

    df = info.get("df")
    if df is not None:
        columns = [str(col) for col in df.columns]
        rows = []
        for row in df.to_dict(orient="records"):
            rows.append([_display_text(row.get(col)) for col in columns])
        return columns, rows

    data = info.get("data")
    if data is None:
        return [], []
    arr = np.asarray(data)

    if getattr(arr.dtype, "names", None):
        columns = [str(col) for col in arr.dtype.names]
        rows = []
        for record in arr:
            rows.append([_display_text(record[col]) for col in columns])
        return columns, rows

    if arr.ndim == 2:
        columns = [f"col_{idx}" for idx in range(arr.shape[1])]
        rows = []
        for row in arr:
            rows.append([_display_text(value) for value in row])
        return columns, rows

    return [], []


def _decode_strings(values):
    return np.array(
        [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in values],
        dtype=object,
    )


def _series_is_numeric(values) -> bool:
    try:
        np.asarray(values, dtype=float)
        return True
    except Exception:
        return False


def _discover_custom_cmaps():
    if not CMAPS_DIR.exists():
        return []
    return sorted({path.stem for path in CMAPS_DIR.glob("*.cmap")})


def _apply_rotation(ax, matrix: np.ndarray, degrees: float) -> None:
    if not ax.images:
        return
    im = ax.images[-1]
    height, width = matrix.shape[:2]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    transform = mtransforms.Affine2D().rotate_deg_around(cx, cy, degrees)
    im.set_transform(transform + ax.transData)

    corners = np.array(
        [
            [-0.5, -0.5],
            [width - 0.5, -0.5],
            [width - 0.5, height - 0.5],
            [-0.5, height - 0.5],
        ]
    )
    theta = np.deg2rad(degrees)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    rotated = (corners - np.array([cx, cy])) @ rotation.T + np.array([cx, cy])
    xmin, ymin = rotated.min(axis=0)
    xmax, ymax = rotated.max(axis=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)


def _remove_axes_border(ax) -> None:
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _stack_axis(shape):
    if len(shape) != 3:
        return None
    a, b, c = shape
    if a == b != c:
        return 2
    if a == c != b:
        return 1
    if b == c != a:
        return 0
    return None


def _select_stack_slice(matrix: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return matrix[index, :, :]
    if axis == 1:
        return matrix[:, index, :]
    return matrix[:, :, index]


def _strip_prefix(value: str, prefix: str) -> str:
    if value.startswith(prefix):
        return value[len(prefix) :]
    return value


def _parse_participant_id(value: str):
    text = str(value)
    if "-" in text:
        parts = text.split("-")
        if parts[0] in {"sub", "subject", "participant"}:
            return None, parts[-1]
        return parts[0], parts[-1]
    if "_" in text:
        parts = text.split("_")
        if parts[0] in {"sub", "subject", "participant"}:
            return None, parts[-1]
        return parts[0], parts[-1]
    return None, text


def _normalize_subject_token(value: str) -> str:
    token = str(value).strip()
    if token.lower().startswith("sub-"):
        token = token[4:]
    return token


def _normalize_session_token(value: str) -> str:
    token = str(value).strip()
    if token.lower().startswith("ses-"):
        token = token[4:]
    upper = token.upper()
    if upper.startswith("T") and len(upper) > 1:
        return f"V{upper[1:]}"
    if token.isdigit():
        return f"V{token}"
    return token


def _display_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("latin-1", errors="ignore")
    try:
        if isinstance(value, np.generic):
            return str(value.item())
    except Exception:
        pass
    return str(value)


def _column_is_numeric(values):
    has_value = False
    for value in values:
        text = _display_text(value).strip()
        if text == "":
            continue
        has_value = True
        try:
            float(text)
        except Exception:
            return False
    return has_value


def _get_valid_keys(path: Path):
    keys = []
    try:
        with np.load(path, allow_pickle=True) as npz:
            for key in npz.files:
                if key == "covars" or key in PARCEL_LABEL_KEYS or key in PARCEL_NAME_KEYS:
                    continue
                try:
                    shape = npz[key].shape
                except Exception:
                    continue
                if _is_valid_matrix_shape(shape):
                    keys.append(key)
    except Exception:
        return []
    return keys


def _format_value(value) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def _coerce_label_indices(labels, expected_len: int):
    if labels is None:
        return None
    values = np.asarray(labels).reshape(-1)
    if values.size != expected_len:
        return None
    out = []
    for raw in values:
        value = raw.item() if isinstance(raw, np.generic) else raw
        if isinstance(value, (bytes, bytearray)):
            value = value.decode(errors="ignore")
        try:
            number = float(value)
        except Exception:
            return None
        if not np.isfinite(number):
            return None
        rounded = int(round(number))
        if not np.isclose(number, rounded):
            return None
        out.append(rounded)
    return out


def _flatten_display_vector(values):
    try:
        array = np.asarray(values)
    except Exception:
        return None
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    elif array.ndim != 1:
        return None
    return [_display_text(value) for value in array.tolist()]


class HistogramDialog(QDialog):
    def __init__(self, entries, entry_ids, titles, parent=None) -> None:
        super().__init__(parent)
        self.entries = entries
        self.entry_ids = list(entry_ids)
        self.titles = dict(titles)
        self._cache = {}
        self.setWindowTitle("Histogram Viewer")
        self._build_ui()
        self._update_plot()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)

        self.list_widget = QListWidget()
        for entry_id in self.entry_ids:
            entry = self.entries.get(entry_id)
            if entry is None:
                continue
            label = self.titles.get(entry_id, entry.get("label", "Matrix"))
            item = QListWidgetItem(label)
            item.setData(USER_ROLE, entry_id)
            item.setCheckState(Qt.Checked)
            self.list_widget.addItem(item)
        self.list_widget.itemChanged.connect(self._update_plot)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout.addWidget(self.list_widget, 0)
        layout.addWidget(self.canvas, 1)

    def _matrix_for_entry(self, entry):
        if entry.get("kind") == "derived":
            return entry.get("matrix"), entry.get("selected_key")
        key = entry.get("selected_key")
        if not key:
            valid_keys = _get_valid_keys(entry["path"])
            if not valid_keys:
                raise KeyError("No valid matrix key")
            key = valid_keys[0]
            entry["selected_key"] = key
        matrix = _load_matrix_from_npz(entry["path"], key, average=True)
        return matrix, key

    def _load_upper_triangle(self, entry_id: str):
        if entry_id in self._cache:
            return self._cache[entry_id]
        entry = self.entries.get(entry_id)
        if entry is None:
            raise KeyError("Entry not found")
        matrix, _ = self._matrix_for_entry(entry)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            values = matrix.ravel()
        else:
            idx = np.triu_indices_from(matrix, k=1)
            values = matrix[idx]
        self._cache[entry_id] = values
        return values

    def _update_plot(self) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        plotted = False
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() != Qt.Checked:
                continue
            entry_id = item.data(USER_ROLE)
            try:
                values = self._load_upper_triangle(entry_id)
            except Exception:
                continue
            label = self.titles.get(entry_id, item.text())
            ax.hist(values, bins=50, alpha=0.6, histtype="step", label=label)
            plotted = True
        if plotted:
            ax.set_title("Upper Triangle Histogram")
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)
        else:
            ax.set_title("No matrices selected")
            ax.set_xticks([])
            ax.set_yticks([])
        self.canvas.draw_idle()


class LabelInfoDialog(QDialog):
    def __init__(self, source_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._source_path = Path(source_path)
        self._vector_options = self._load_vector_options()
        self.setWindowTitle(f"Label Info - {self._source_path.name}")
        self.resize(860, 620)
        self._build_ui()
        self._populate_key_selectors()
        self._update_table()

    def _load_vector_options(self):
        options = {}
        try:
            with np.load(self._source_path, allow_pickle=True) as npz:
                for key in npz.files:
                    values = _flatten_display_vector(npz[key])
                    if values is not None:
                        options[str(key)] = values
        except Exception:
            return {}
        return options

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        info_label = QLabel(
            "Inspect parcel label indices and names from the selected NPZ. "
            "If the standard keys are missing, choose alternative 1D arrays from the dropdowns."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)

        form.addWidget(QLabel("Label indices key"), 0, 0)
        self.indices_combo = QComboBox()
        self.indices_combo.currentIndexChanged.connect(self._update_table)
        form.addWidget(self.indices_combo, 0, 1)

        form.addWidget(QLabel("Label names key"), 0, 2)
        self.names_combo = QComboBox()
        self.names_combo.currentIndexChanged.connect(self._update_table)
        form.addWidget(self.names_combo, 0, 3)

        layout.addLayout(form)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Row", "Label Index", "Label Name"])
        self.table.setAlternatingRowColors(True)
        if QT_LIB == 6:
            self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        else:
            self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
            _set_header_resize_mode(header, 0, "ResizeToContents")
            _set_header_resize_mode(header, 1, "ResizeToContents")
            _set_header_resize_mode(header, 2, "Stretch")
        layout.addWidget(self.table, 1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_row.addWidget(close_button)
        layout.addLayout(close_row)

    def _preferred_key(self, choices, preferred_keys):
        for key in preferred_keys:
            if key in choices:
                return key
        return ""

    def _populate_key_selectors(self) -> None:
        options = [""] + sorted(self._vector_options.keys())
        self.indices_combo.blockSignals(True)
        self.names_combo.blockSignals(True)
        self.indices_combo.clear()
        self.names_combo.clear()
        self.indices_combo.addItem("(None)", "")
        self.names_combo.addItem("(None)", "")
        for key in sorted(self._vector_options.keys()):
            self.indices_combo.addItem(key, key)
            self.names_combo.addItem(key, key)

        preferred_indices = self._preferred_key(options, PARCEL_LABEL_KEYS)
        preferred_names = self._preferred_key(options, PARCEL_NAME_KEYS)
        if preferred_indices:
            self.indices_combo.setCurrentText(preferred_indices)
        elif self.indices_combo.count() > 1:
            self.indices_combo.setCurrentIndex(1)
        if preferred_names:
            self.names_combo.setCurrentText(preferred_names)
        elif self.names_combo.count() > 1:
            fallback_index = 2 if self.names_combo.count() > 2 else 1
            self.names_combo.setCurrentIndex(fallback_index)
        self.indices_combo.blockSignals(False)
        self.names_combo.blockSignals(False)

    def _selected_vector(self, combo: QComboBox):
        key = combo.currentData()
        if not key:
            return None
        return list(self._vector_options.get(str(key), []))

    def _update_table(self) -> None:
        indices = self._selected_vector(self.indices_combo) or []
        names = self._selected_vector(self.names_combo) or []
        row_count = max(len(indices), len(names))
        self.table.setRowCount(row_count)
        for row in range(row_count):
            values = [
                str(row),
                indices[row] if row < len(indices) else "",
                names[row] if row < len(names) else "",
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(str(text))
                item.setFlags(item.flags() & ~_is_editable_flag())
                self.table.setItem(row, col, item)

        missing = []
        if self._preferred_key(self._vector_options, PARCEL_LABEL_KEYS) == "":
            missing.append("parcel label indices")
        if self._preferred_key(self._vector_options, PARCEL_NAME_KEYS) == "":
            missing.append("parcel label names")
        if missing:
            missing_text = ", ".join(missing)
            self.status_label.setText(
                f"Standard keys not found for {missing_text}. Choose alternative arrays from the dropdowns."
            )
        else:
            self.status_label.setText("Loaded standard parcel label keys from the NPZ file.")


class ParticipantsInfoDialog(QDialog):
    def __init__(self, source_path: Path, covars_info, parent=None) -> None:
        super().__init__(parent)
        self._source_path = Path(source_path)
        self._columns, self._rows = _covars_to_rows(covars_info)
        self.setWindowTitle(f"Participants - {self._source_path.name}")
        self.resize(1080, 760)
        self._build_ui()
        self._populate_table()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        summary = QLabel(
            f"{self._source_path.name} | rows: {len(self._rows)} | covariates: {len(self._columns)}"
        )
        summary.setWordWrap(False)
        layout.addWidget(summary)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        if QT_LIB == 6:
            self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        else:
            self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_row.addWidget(close_button)
        layout.addLayout(close_row)

    def _populate_table(self) -> None:
        self.table.clear()
        self.table.setColumnCount(len(self._columns))
        self.table.setHorizontalHeaderLabels(self._columns)
        self.table.setRowCount(len(self._rows))

        editable_flag = _is_editable_flag()
        for row_idx, row_values in enumerate(self._rows):
            for col_idx, value in enumerate(row_values):
                item = QTableWidgetItem(_display_text(value))
                item.setFlags(item.flags() & ~editable_flag)
                self.table.setItem(row_idx, col_idx, item)

        header = self.table.horizontalHeader()
        if header is not None and self._columns:
            for col_idx in range(max(len(self._columns) - 1, 0)):
                _set_header_resize_mode(header, col_idx, "ResizeToContents")
            _set_header_resize_mode(header, len(self._columns) - 1, "Stretch")

        if not self._columns:
            self.status_label.setText("No covariate columns were found in this NPZ file.")
        else:
            self.status_label.setText("Loaded participant covariates from NPZ `covars`.")


class PreferencesDialog(QDialog):
    def __init__(
        self,
        *,
        theme_name: str,
        matrix_cmap: str,
        gradient_cmap: str,
        matlab_cmd: str,
        matlab_nbs_path: str,
        results_dir: str,
        bids_dir: str,
        atlas_dir: str,
        colormap_names,
        dialog_title: str = "Preferences",
        require_results_dir: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._require_results_dir = bool(require_results_dir)
        self.setWindowTitle(dialog_title)
        self.resize(860, 520)
        self._colormap_names = list(colormap_names or [])
        self._build_ui(
            theme_name=theme_name,
            matrix_cmap=matrix_cmap,
            gradient_cmap=gradient_cmap,
            matlab_cmd=matlab_cmd,
            matlab_nbs_path=matlab_nbs_path,
            results_dir=results_dir,
            bids_dir=bids_dir,
            atlas_dir=atlas_dir,
        )

    def _build_ui(
        self,
        *,
        theme_name: str,
        matrix_cmap: str,
        gradient_cmap: str,
        matlab_cmd: str,
        matlab_nbs_path: str,
        results_dir: str,
        bids_dir: str,
        atlas_dir: str,
    ) -> None:
        layout = QVBoxLayout(self)
        info_label = QLabel(
            "Configure application defaults. Results folder is required and will be used as the default output root."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)

        row = 0
        form.addWidget(QLabel("Theme"), row, 0)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "Teya", "Donald"])
        if self.theme_combo.findText(theme_name) >= 0:
            self.theme_combo.setCurrentText(theme_name)
        else:
            self.theme_combo.setCurrentText("Dark")
        form.addWidget(self.theme_combo, row, 1, 1, 3)

        row += 1
        form.addWidget(QLabel("Default matrix color map"), row, 0)
        self.matrix_cmap_combo = QComboBox()
        self.matrix_cmap_combo.addItems(self._colormap_names)
        if self.matrix_cmap_combo.findText(matrix_cmap) >= 0:
            self.matrix_cmap_combo.setCurrentText(matrix_cmap)
        elif self.matrix_cmap_combo.count() > 0:
            self.matrix_cmap_combo.setCurrentIndex(0)
        form.addWidget(self.matrix_cmap_combo, row, 1, 1, 3)

        row += 1
        form.addWidget(QLabel("Default gradients color map"), row, 0)
        self.gradients_cmap_combo = QComboBox()
        self.gradients_cmap_combo.addItems(self._colormap_names)
        if self.gradients_cmap_combo.findText(gradient_cmap) >= 0:
            self.gradients_cmap_combo.setCurrentText(gradient_cmap)
        elif self.gradients_cmap_combo.count() > 0:
            self.gradients_cmap_combo.setCurrentIndex(0)
        form.addWidget(self.gradients_cmap_combo, row, 1, 1, 3)

        row += 1
        form.addWidget(QLabel("MATLAB executable"), row, 0)
        self.matlab_cmd_edit = QLineEdit(str(matlab_cmd or DEFAULT_MATLAB_CMD))
        form.addWidget(self.matlab_cmd_edit, row, 1, 1, 2)
        matlab_browse_button = QPushButton("Browse")
        matlab_browse_button.clicked.connect(self._browse_matlab_cmd)
        form.addWidget(matlab_browse_button, row, 3)

        row += 1
        form.addWidget(QLabel("NBS path"), row, 0)
        self.nbs_path_edit = QLineEdit(str(matlab_nbs_path or DEFAULT_MATLAB_NBS_PATH))
        form.addWidget(self.nbs_path_edit, row, 1, 1, 2)
        nbs_browse_button = QPushButton("Browse")
        nbs_browse_button.clicked.connect(self._browse_nbs_path)
        form.addWidget(nbs_browse_button, row, 3)

        row += 1
        form.addWidget(QLabel("Results folder"), row, 0)
        self.results_dir_edit = QLineEdit(str(results_dir or ""))
        self.results_dir_edit.setPlaceholderText("Required output folder")
        form.addWidget(self.results_dir_edit, row, 1, 1, 2)
        results_browse_button = QPushButton("Browse")
        results_browse_button.clicked.connect(self._browse_results_dir)
        form.addWidget(results_browse_button, row, 3)

        row += 1
        form.addWidget(QLabel("BIDS folder"), row, 0)
        self.bids_dir_edit = QLineEdit(str(bids_dir or ""))
        self.bids_dir_edit.setPlaceholderText("Optional default folder when opening matrices")
        form.addWidget(self.bids_dir_edit, row, 1, 1, 2)
        bids_browse_button = QPushButton("Browse")
        bids_browse_button.clicked.connect(self._browse_bids_dir)
        form.addWidget(bids_browse_button, row, 3)

        row += 1
        form.addWidget(QLabel("Atlas folder"), row, 0)
        self.atlas_dir_edit = QLineEdit(str(atlas_dir or ""))
        self.atlas_dir_edit.setPlaceholderText("Optional default folder for parcellation / atlas files")
        form.addWidget(self.atlas_dir_edit, row, 1, 1, 2)
        atlas_browse_button = QPushButton("Browse")
        atlas_browse_button.clicked.connect(self._browse_atlas_dir)
        form.addWidget(atlas_browse_button, row, 3)

        layout.addLayout(form)
        layout.addStretch(1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_row.addWidget(self.cancel_button)
        self.save_button = QPushButton("Save Configuration")
        self.save_button.clicked.connect(self.accept)
        button_row.addWidget(self.save_button)
        layout.addLayout(button_row)

    @staticmethod
    def _browse_directory(parent, title: str, current: str) -> str:
        start_dir = str(Path(current).expanduser()) if current else str(Path.home())
        return QFileDialog.getExistingDirectory(parent, title, start_dir)

    def _browse_matlab_cmd(self) -> None:
        current = self.matlab_cmd_edit.text().strip()
        start_dir = str(Path(current).parent) if current else str(Path.home())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select MATLAB executable",
            start_dir,
            "All files (*)",
        )
        if selected:
            self.matlab_cmd_edit.setText(str(Path(selected).resolve()))

    def _browse_nbs_path(self) -> None:
        current = self.nbs_path_edit.text().strip()
        start_dir = str(Path(current).expanduser()) if current else str(Path.home())
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select NBS directory",
            start_dir,
        )
        if selected:
            self.nbs_path_edit.setText(str(Path(selected).resolve()))

    def _browse_results_dir(self) -> None:
        selected = self._browse_directory(
            self,
            "Select results folder",
            self.results_dir_edit.text().strip(),
        )
        if selected:
            self.results_dir_edit.setText(str(Path(selected).resolve()))

    def _browse_bids_dir(self) -> None:
        selected = self._browse_directory(
            self,
            "Select BIDS folder",
            self.bids_dir_edit.text().strip(),
        )
        if selected:
            self.bids_dir_edit.setText(str(Path(selected).resolve()))

    def _browse_atlas_dir(self) -> None:
        selected = self._browse_directory(
            self,
            "Select atlas folder",
            self.atlas_dir_edit.text().strip(),
        )
        if selected:
            self.atlas_dir_edit.setText(str(Path(selected).resolve()))

    def accept(self) -> None:
        results_dir = self.results_dir_edit.text().strip()
        if self._require_results_dir and not results_dir:
            QMessageBox.warning(self, "Results Folder Required", "Select a results folder before continuing.")
            return
        if results_dir:
            try:
                results_path = Path(results_dir).expanduser()
                results_path.mkdir(parents=True, exist_ok=True)
                self.results_dir_edit.setText(str(results_path.resolve()))
            except Exception as exc:
                QMessageBox.warning(self, "Invalid Results Folder", f"Failed to create results folder:\n{exc}")
                return
        super().accept()

    def values(self):
        return {
            "theme": self.theme_combo.currentText().strip(),
            "matrix_colormap": self.matrix_cmap_combo.currentText().strip(),
            "gradient_colormap": self.gradients_cmap_combo.currentText().strip(),
            "matlab_cmd": self.matlab_cmd_edit.text().strip(),
            "matlab_nbs_path": self.nbs_path_edit.text().strip(),
            "results_dir": self.results_dir_edit.text().strip(),
            "bids_dir": self.bids_dir_edit.text().strip(),
            "atlas_dir": self.atlas_dir_edit.text().strip(),
        }


class ExportGridDialog(QDialog):
    _FILTERS = "PDF (*.pdf);;SVG (*.svg);;PNG (*.png)"

    def __init__(
        self,
        *,
        default_path: str,
        default_columns: int,
        rotate: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._selected_filter = "PDF (*.pdf)"
        self.setWindowTitle("Export Grid")
        self.resize(720, 220)
        self._build_ui(
            default_path=default_path,
            default_columns=default_columns,
            rotate=rotate,
        )

    def _build_ui(self, *, default_path: str, default_columns: int, rotate: bool) -> None:
        layout = QVBoxLayout(self)

        info_label = QLabel(
            "Choose the output file, grid layout, and optional 45 degree rotation for the exported workspace matrices."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)

        form.addWidget(QLabel("Output path"), 0, 0)
        self.output_path_edit = QLineEdit(str(default_path or ""))
        self.output_path_edit.setPlaceholderText("Choose PDF, SVG, or PNG output")
        form.addWidget(self.output_path_edit, 0, 1, 1, 2)
        self.output_browse_button = QPushButton("Browse")
        self.output_browse_button.clicked.connect(self._browse_output_path)
        form.addWidget(self.output_browse_button, 0, 3)

        form.addWidget(QLabel("Columns"), 1, 0)
        self.columns_spin = QSpinBox()
        self.columns_spin.setRange(1, 12)
        self.columns_spin.setValue(max(1, min(int(default_columns), 12)))
        form.addWidget(self.columns_spin, 1, 1)

        self.rotate_check = QCheckBox("Rotate 45 deg")
        self.rotate_check.setObjectName("greenSquareIndicator")
        self.rotate_check.setChecked(bool(rotate))
        form.addWidget(self.rotate_check, 1, 2, 1, 2)

        layout.addLayout(form)
        layout.addStretch(1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_row.addWidget(self.cancel_button)
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.accept)
        button_row.addWidget(self.export_button)
        layout.addLayout(button_row)

    def _browse_output_path(self) -> None:
        current = self.output_path_edit.text().strip()
        if current:
            start_path = Path(current).expanduser()
        else:
            start_path = Path.home() / "connectome_grid.pdf"
        selected, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Choose export path",
            str(start_path),
            self._FILTERS,
        )
        if not selected:
            return
        self.output_path_edit.setText(str(Path(selected).expanduser()))
        if selected_filter:
            self._selected_filter = selected_filter

    def set_theme(self, theme_name: str) -> None:
        theme = str(theme_name or "Dark").strip().title()
        if theme == "Dark":
            unchecked_bg = "#1f2430"
            unchecked_border = "#94a3b8"
        elif theme == "Teya":
            unchecked_bg = "#ffe6f1"
            unchecked_border = "#0b7f7a"
        elif theme == "Donald":
            unchecked_bg = "#c96a04"
            unchecked_border = "#ffd19e"
        else:
            unchecked_bg = "#ffffff"
            unchecked_border = "#94a3b8"
        self.rotate_check.setStyleSheet(
            "QCheckBox#greenSquareIndicator::indicator {"
            "width: 16px;"
            "height: 16px;"
            "border-radius: 2px;"
            "}"
            f"QCheckBox#greenSquareIndicator::indicator:unchecked {{ background: {unchecked_bg}; border: 2px solid {unchecked_border}; }}"
            "QCheckBox#greenSquareIndicator::indicator:checked { background: #16a34a; border: 2px solid #15803d; }"
        )

    def accept(self) -> None:
        output_raw = self.output_path_edit.text().strip()
        if not output_raw:
            QMessageBox.warning(self, "Output Path Required", "Choose an export path before continuing.")
            return
        try:
            output_path = Path(output_raw).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path_edit.setText(str(output_path))
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Output Path", f"Failed to prepare output folder:\n{exc}")
            return
        super().accept()

    def values(self):
        return {
            "output_path": self.output_path_edit.text().strip(),
            "columns": int(self.columns_spin.value()),
            "rotate": bool(self.rotate_check.isChecked()),
            "selected_filter": self._selected_filter,
        }


class BatchMatrixImportDialog(QDialog):
    _MODALITY_OPTIONS = (
        ("All", ""),
        ("dwi", "connectivity_dwi"),
        ("mrsi", "connectivity_mrsi"),
        ("func", "connectivity_func"),
    )
    _STEP_TITLES = ("Select Matrices", "Selection", "Setup", "Multimodal", "Run")
    _HARMONIZE_TYPE_OPTIONS = ("categorical", "continuous")

    def __init__(self, folder_path: Path, candidate_paths, export_callback=None, parent=None) -> None:
        super().__init__(parent)
        self._folder_path = Path(folder_path)
        self._candidate_paths = [Path(path) for path in candidate_paths]
        self._export_callback = export_callback
        self._atlas_options = self._detected_atlas_options()
        self._requested_action = "add"
        self._selection_widget = None
        self._stack_prepare_widget = None
        self._current_step = 0
        self._multimodal_excluded_pairs = set()
        self._multimodal_table_refreshing = False
        self._multimodal_duplicate_entries = []
        self._workflow_log_expanded = False
        self._optional_steps_refresh_pending = False
        self.setWindowTitle("Add Batch")
        self.resize(1200, 800)
        self._build_ui()
        self._populate_files()
        self._apply_filters()
        self.set_theme(getattr(parent, "_theme_name", "Dark"))
        self._go_to_step(0)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.workflow_splitter = QSplitter(Qt.Vertical)

        content_widget = QWidget()
        content_row = QHBoxLayout(content_widget)
        content_row.setContentsMargins(0, 0, 0, 0)

        stepper_frame = QFrame()
        stepper_layout = QVBoxLayout(stepper_frame)
        stepper_layout.setContentsMargins(6, 6, 6, 6)
        stepper_layout.setSpacing(8)
        stepper_title = QLabel("Workflow")
        stepper_layout.addWidget(stepper_title)
        self._step_buttons = []
        for idx, title in enumerate(self._STEP_TITLES):
            button = QPushButton(f"{idx + 1}. {title}")
            button.setObjectName("workflowStepButton")
            button.setCheckable(True)
            button.setMinimumHeight(36)
            button.clicked.connect(lambda _checked=False, i=idx: self._go_to_step(i))
            stepper_layout.addWidget(button)
            self._step_buttons.append(button)
        for button in self._step_buttons[1:]:
            button.setEnabled(False)
        stepper_layout.addStretch(1)
        content_row.addWidget(stepper_frame, 0)

        self.step_stack = QStackedWidget()
        content_row.addWidget(self.step_stack, 1)
        self.workflow_splitter.addWidget(content_widget)

        terminal_group = QGroupBox("Integrated Terminal")
        terminal_layout = QVBoxLayout(terminal_group)
        self.workflow_terminal = QPlainTextEdit()
        self.workflow_terminal.setReadOnly(True)
        self.workflow_terminal.setPlaceholderText("Stack progress will appear here.")
        self.workflow_terminal.setMinimumHeight(150)
        if QT_LIB == 6:
            self.workflow_terminal.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        else:
            self.workflow_terminal.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.workflow_terminal.setStyleSheet(
            "QPlainTextEdit {"
            " background-color: #000000;"
            " color: #b8f7c6;"
            " border: 1px solid #404040;"
            " border-radius: 4px;"
            " selection-background-color: #2d6cdf;"
            " font-family: 'DejaVu Sans Mono', 'Courier New', monospace;"
            " font-size: 10.5pt;"
            "}"
        )
        terminal_layout.addWidget(self.workflow_terminal, 1)
        self.workflow_splitter.addWidget(terminal_group)
        self.workflow_splitter.setStretchFactor(0, 4)
        self.workflow_splitter.setStretchFactor(1, 1)
        self.workflow_splitter.setSizes([620, 180])
        layout.addWidget(self.workflow_splitter, 1)

        self.workflow_log_toggle_button = QPushButton("Show log ▾")
        self.workflow_log_toggle_button.clicked.connect(self._toggle_workflow_log_drawer)
        self.workflow_log_toggle_button.setMaximumWidth(140)
        layout.addWidget(self.workflow_log_toggle_button, 0, _align_right_flag())
        self.workflow_log_drawer = terminal_group
        self._set_workflow_log_drawer_expanded(False)

        selection_page = QWidget()
        selection_layout = QVBoxLayout(selection_page)

        self.folder_label = QLabel(f"Folder: {self._folder_path}")
        self.folder_label.setWordWrap(True)
        selection_layout.addWidget(self.folder_label)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter"))
        self.filter_value_edit = QLineEdit("")
        self.filter_value_edit.setPlaceholderText("Substring match, e.g. scale3 or sub-001")
        self.filter_value_edit.textChanged.connect(self._apply_filters)
        self.filter_value_edit.returnPressed.connect(self._apply_filters)
        filter_row.addWidget(self.filter_value_edit, 1)

        filter_row.addWidget(QLabel("Modality"))
        self.modality_combo = QComboBox()
        for label, token in self._MODALITY_OPTIONS:
            self.modality_combo.addItem(label, token)
        self.modality_combo.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.modality_combo)

        self.filter_button = QPushButton("Filter")
        self.filter_button.clicked.connect(self._apply_filters)
        filter_row.addWidget(self.filter_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self._reset_filters)
        filter_row.addWidget(self.reset_button)
        selection_layout.addLayout(filter_row)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        selection_layout.addWidget(self.summary_label)

        self.file_list = QListWidget()
        if QT_LIB == 6:
            self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        else:
            self.file_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.file_list.itemChanged.connect(self._on_item_changed)
        selection_layout.addWidget(self.file_list, 1)

        select_row = QHBoxLayout()
        self.select_visible_button = QPushButton("Select Visible")
        self.select_visible_button.clicked.connect(lambda: self._set_visible_checked(True))
        select_row.addWidget(self.select_visible_button)
        self.clear_visible_button = QPushButton("Clear Visible")
        self.clear_visible_button.clicked.connect(lambda: self._set_visible_checked(False))
        select_row.addWidget(self.clear_visible_button)
        select_row.addStretch(1)
        selection_layout.addLayout(select_row)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        actions.addWidget(self.cancel_button)
        self.stack_button = QPushButton("Stack")
        self.stack_button.clicked.connect(self._open_stack_prepare)
        actions.addWidget(self.stack_button)
        self.add_selected_button = QPushButton("Add Selected")
        self.add_selected_button.clicked.connect(self._accept_if_selection)
        actions.addWidget(self.add_selected_button)
        selection_layout.addLayout(actions)

        self.selection_page = QWidget()
        self.selection_tab_layout = QVBoxLayout(self.selection_page)
        self.selection_placeholder = QLabel(
            "Step 2 appears here after you select matrices in Step 1 and click 'Stack'."
        )
        self.selection_placeholder.setWordWrap(True)
        self.selection_tab_layout.addWidget(self.selection_placeholder)
        self.selection_tab_layout.addStretch(1)

        self.setup_page = QWidget()
        self.setup_tab_layout = QVBoxLayout(self.setup_page)
        self.setup_placeholder = QLabel(
            "Step 3 appears here after you open the Selection step."
        )
        self.setup_placeholder.setWordWrap(True)
        self.setup_tab_layout.addWidget(self.setup_placeholder)
        self.setup_tab_layout.addStretch(1)

        multimodal_page = QWidget()
        multimodal_layout = QVBoxLayout(multimodal_page)
        self.multimodal_enable_check = QCheckBox("Enable multimodal stack")
        self.multimodal_enable_check.toggled.connect(self._update_multimodal_controls)
        multimodal_layout.addWidget(self.multimodal_enable_check)
        self.multimodal_align_check = QCheckBox("Align cross-modality matrices")
        self.multimodal_align_check.setChecked(False)
        multimodal_layout.addWidget(self.multimodal_align_check)
        reference_row = QHBoxLayout()
        reference_row.addWidget(QLabel("Reference modality"))
        self.reference_modality_combo = QComboBox()
        self.reference_modality_combo.currentIndexChanged.connect(self._refresh_multimodal_table)
        self.reference_modality_combo.currentIndexChanged.connect(self._refresh_run_table)
        reference_row.addWidget(self.reference_modality_combo, 1)
        multimodal_layout.addLayout(reference_row)
        self.multimodal_summary_label = QLabel("")
        self.multimodal_summary_label.setWordWrap(True)
        multimodal_layout.addWidget(self.multimodal_summary_label)
        self.multimodal_table = QTableWidget()
        self.multimodal_table.setAlternatingRowColors(True)
        self.multimodal_table.itemChanged.connect(self._on_multimodal_table_item_changed)
        if QT_LIB == 6:
            self.multimodal_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.multimodal_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        else:
            self.multimodal_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.multimodal_table.setSelectionMode(QAbstractItemView.SingleSelection)
        multimodal_layout.addWidget(self.multimodal_table, 1)
        self.multimodal_optional_label = QLabel(
            "Optional step. Continue to Run without configuring multimodal stacking."
        )
        self.multimodal_optional_label.setWordWrap(True)
        multimodal_layout.addWidget(self.multimodal_optional_label)

        run_page = QWidget()
        run_layout = QVBoxLayout(run_page)
        self.run_info_label = QLabel(
            "Final step. Review the subject/session pairs that will be stacked, then run processing."
        )
        self.run_info_label.setWordWrap(True)
        run_layout.addWidget(self.run_info_label)
        self.run_summary_label = QLabel(
            "The stack preview will appear here after Selection and Setup are ready."
        )
        self.run_summary_label.setWordWrap(True)
        run_layout.addWidget(self.run_summary_label)
        self.run_table = QTableWidget()
        self.run_table.setAlternatingRowColors(True)
        if QT_LIB == 6:
            self.run_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.run_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        else:
            self.run_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.run_table.setSelectionMode(QAbstractItemView.SingleSelection)
        run_layout.addWidget(self.run_table, 1)
        run_actions = QHBoxLayout()
        run_actions.addStretch(1)
        self.run_import_button = QPushButton("Import to workspace")
        self.run_import_button.setEnabled(False)
        self.run_import_button.clicked.connect(self._trigger_workflow_import)
        run_actions.addWidget(self.run_import_button)
        self.run_close_button = QPushButton("Close")
        self.run_close_button.clicked.connect(self._request_close)
        run_actions.addWidget(self.run_close_button)
        self.run_process_button = QPushButton("Process")
        self.run_process_button.setEnabled(False)
        self.run_process_button.clicked.connect(self._trigger_workflow_process)
        run_actions.addWidget(self.run_process_button)
        run_layout.addLayout(run_actions)

        self.step_stack.addWidget(selection_page)
        self.step_stack.addWidget(self.selection_page)
        self.step_stack.addWidget(self.setup_page)
        self.step_stack.addWidget(multimodal_page)
        self.step_stack.addWidget(run_page)

    def _populate_files(self) -> None:
        self.file_list.blockSignals(True)
        self.file_list.clear()
        flags = _is_enabled_flag() | _is_user_checkable_flag()
        for path in self._candidate_paths:
            item = QListWidgetItem(self._relative_label(path))
            item.setToolTip(str(path))
            item.setData(USER_ROLE, str(path))
            item.setFlags(item.flags() | flags)
            item.setCheckState(Qt.Unchecked)
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)

    def _relative_label(self, path: Path) -> str:
        try:
            return path.relative_to(self._folder_path).as_posix()
        except Exception:
            return path.name

    def _detected_atlas_options(self):
        atlases = {_connectivity_atlas_tag(path) for path in self._candidate_paths}
        return sorted(atlases, key=lambda value: (str(value).lower() == "unknown", str(value).lower()))

    def _filter_tokens(self):
        values = [token.strip().lower() for token in self.filter_value_edit.text().split(",")]
        return [token for token in values if token]

    def _matches_current_filters(self, path: Path) -> bool:
        name = self._relative_label(path).lower()
        modality_token = str(self.modality_combo.currentData() or "").strip().lower()
        if modality_token and modality_token not in name:
            return False
        for token in self._filter_tokens():
            if token not in name:
                return False
        return True

    def _checked_count(self) -> int:
        total = 0
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item is not None and item.checkState() == Qt.Checked:
                total += 1
        return total

    def _on_item_changed(self, _item) -> None:
        self._apply_filters()
        if self._selection_widget is not None:
            self._selection_widget.set_selected_paths(self.selected_paths())
        if self._stack_prepare_widget is not None:
            self._stack_prepare_widget.set_selected_paths(self.selected_paths())
            self._refresh_optional_steps()

    def _apply_filters(self) -> None:
        visible_count = 0
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item is None:
                continue
            raw_path = item.data(USER_ROLE)
            path = Path(str(raw_path))
            is_visible = self._matches_current_filters(path)
            item.setHidden(not is_visible)
            if is_visible:
                visible_count += 1
        total = self.file_list.count()
        selected = self._checked_count()
        self.summary_label.setText(
            f"Showing {visible_count} of {total} connectivity matrices. Selected: {selected}."
        )

    def _reset_filters(self) -> None:
        self.filter_value_edit.clear()
        self.modality_combo.setCurrentIndex(0)
        self._apply_filters()

    def _set_visible_checked(self, checked: bool) -> None:
        target_state = Qt.Checked if checked else Qt.Unchecked
        self.file_list.blockSignals(True)
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item is None or item.isHidden():
                continue
            item.setCheckState(target_state)
        self.file_list.blockSignals(False)
        self._apply_filters()

    def selected_paths(self):
        paths = []
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item is None or item.checkState() != Qt.Checked:
                continue
            raw_path = item.data(USER_ROLE)
            if raw_path:
                paths.append(str(raw_path))
        return paths

    def requested_action(self) -> str:
        return str(self._requested_action or "add")

    def _stack_processing(self) -> bool:
        if self._stack_prepare_widget is None:
            return False
        if hasattr(self._stack_prepare_widget, "is_processing"):
            try:
                return bool(self._stack_prepare_widget.is_processing())
            except Exception:
                return False
        return False

    def _teardown_embedded_workflow_widgets(self) -> None:
        if self._stack_prepare_widget is not None:
            _write_diagnostic_line("Tearing down embedded stack_prepare_widget")
            if hasattr(self._stack_prepare_widget, "prepare_for_close"):
                try:
                    self._stack_prepare_widget.prepare_for_close()
                except Exception:
                    pass
            try:
                self._stack_prepare_widget.hide()
            except Exception:
                pass
            self._stack_prepare_widget = None
        if self._selection_widget is not None:
            _write_diagnostic_line("Tearing down embedded selection_widget")
            if hasattr(self._selection_widget, "prepare_for_close"):
                try:
                    self._selection_widget.prepare_for_close()
                except Exception:
                    pass
            try:
                self._selection_widget.hide()
            except Exception:
                pass
            self._selection_widget = None

    def _request_close(self) -> None:
        if self._stack_processing():
            QMessageBox.information(
                self,
                "Processing Active",
                "Wait for the stack process to finish before closing this window.",
            )
            return
        self.reject()

    def reject(self) -> None:
        if self._stack_processing():
            QMessageBox.information(
                self,
                "Processing Active",
                "Wait for the stack process to finish before closing this window.",
            )
            return
        super().reject()

    def accept(self) -> None:
        super().accept()

    def _append_workflow_terminal(self, text) -> None:
        if not hasattr(self, "workflow_terminal") or self.workflow_terminal is None:
            return
        cleaned = str(text or "").rstrip("\n")
        if not cleaned:
            return
        self.workflow_terminal.appendPlainText(cleaned)
        scroll = self.workflow_terminal.verticalScrollBar()
        if scroll is not None:
            scroll.setValue(scroll.maximum())

    def _set_workflow_log_drawer_expanded(self, expanded: bool) -> None:
        self._workflow_log_expanded = bool(expanded)
        if hasattr(self, "workflow_log_drawer") and self.workflow_log_drawer is not None:
            self.workflow_log_drawer.setVisible(self._workflow_log_expanded)
        if hasattr(self, "workflow_log_toggle_button") and self.workflow_log_toggle_button is not None:
            self.workflow_log_toggle_button.setText(
                "Hide log ▴" if self._workflow_log_expanded else "Show log ▾"
            )
        if self._workflow_log_expanded and hasattr(self, "workflow_splitter"):
            self.workflow_splitter.setSizes([620, 180])

    def _toggle_workflow_log_drawer(self) -> None:
        self._set_workflow_log_drawer_expanded(not self._workflow_log_expanded)

    def _go_to_step(self, step_index: int) -> None:
        step = max(0, min(int(step_index), len(self._STEP_TITLES) - 1))
        if step > 0 and not self._step_buttons[step].isEnabled():
            step = 0
        self._current_step = step
        self.step_stack.setCurrentIndex(step)
        for idx, button in enumerate(self._step_buttons):
            is_current = idx == step
            button.setChecked(is_current)
            prefix = "▶ " if is_current else ""
            button.setText(f"{prefix}{idx + 1}. {self._STEP_TITLES[idx]}")
        if step == 4:
            self._refresh_run_table()
        self._sync_workflow_process_button()

    def _run_preview_visible(self) -> bool:
        if not hasattr(self, "step_stack") or self.step_stack is None:
            return False
        try:
            return int(self.step_stack.currentIndex()) == 4
        except Exception:
            return False

    def _group_from_path(self, path: Path) -> str:
        parts = list(Path(path).parts)
        if "derivatives" in parts:
            idx = parts.index("derivatives")
            if idx > 0:
                return str(parts[idx - 1])
        return str(self._folder_path.name)

    def _path_subject_session(self, path: Path):
        text = Path(path).name
        sub_match = re.search(r"sub-([^_]+)", text)
        ses_match = re.search(r"ses-([^_]+)", text)
        sub = _normalize_subject_token(sub_match.group(1) if sub_match else "")
        ses = _normalize_session_token(ses_match.group(1) if ses_match else "")
        return self._group_from_path(path), sub, ses

    def _path_subject_session_pair(self, path: Path):
        _group, sub, ses = self._path_subject_session(path)
        return sub, ses

    def _covar_row_subject_session(self, covar_row, default_group: str, col_map=None):
        if col_map is None:
            covars_df = self._filtered_covars_df()
            if covars_df is None:
                return default_group, "", ""
            col_map = {str(col).lower(): col for col in covars_df.columns}
        group_col = col_map.get("group")
        sub_col = (
            col_map.get("participant_id")
            or col_map.get("subject_id")
            or col_map.get("subject")
            or col_map.get("sub")
            or col_map.get("id")
        )
        ses_col = col_map.get("session_id") or col_map.get("session") or col_map.get("ses")
        participant_value = covar_row.get(sub_col, "") if sub_col is not None else ""
        session_value = covar_row.get(ses_col, "") if ses_col is not None else ""
        covar_group = str(covar_row.get(group_col, "")).strip() if group_col is not None else ""
        parsed_group, parsed_sub = _parse_participant_id(str(participant_value))
        group = covar_group or parsed_group or default_group
        sub = _normalize_subject_token(str(parsed_sub or participant_value).strip())
        ses = _normalize_session_token(str(session_value).strip())
        return group, sub, ses

    def _allowed_pair_keys_from_covars(self):
        covars_df = self._filtered_covars_df()
        if covars_df is None:
            return None
        if len(covars_df) == 0:
            return set()
        default_group = self._group_from_path(Path(self.selected_paths()[0])) if self.selected_paths() else ""
        col_map = {str(col).lower(): col for col in covars_df.columns}
        keys = set()
        for _, covar_row in covars_df.iterrows():
            _group, sub, ses = self._covar_row_subject_session(covar_row, default_group, col_map=col_map)
            keys.add((sub, ses))
        return keys

    def _pair_matches_path(self, path: Path, sub: str, ses: str) -> bool:
        pair = self._path_subject_session_pair(path)
        return pair == (_normalize_subject_token(sub), _normalize_session_token(ses))

    def _paths_for_subject_session(self, paths, sub: str, ses: str):
        return [Path(path) for path in paths if self._pair_matches_path(Path(path), sub, ses)]

    def _modality_counts_for_paths(self, paths, modalities):
        counts = {modality: 0 for modality in modalities}
        for path in paths:
            modality = _infer_modality_from_path(Path(path))
            if modality in counts:
                counts[modality] += 1
        return counts

    def _modality_paths_for_paths(self, paths, modalities):
        matches = {modality: [] for modality in modalities}
        for path in paths:
            resolved = str(Path(path))
            modality = _infer_modality_from_path(Path(path))
            if modality in matches:
                matches[modality].append(resolved)
        return matches

    def effective_selected_paths(self):
        selected = [Path(path) for path in self.selected_paths()]
        allowed_pairs = self._allowed_pair_keys_from_covars()
        out = []
        for path in selected:
            pair_key = self._path_subject_session_pair(path)
            if allowed_pairs is not None and pair_key not in allowed_pairs:
                continue
            if pair_key in self._multimodal_excluded_pairs:
                continue
            out.append(str(path))
        return out

    def _filtered_covars_df(self):
        if self._selection_widget is not None and hasattr(self._selection_widget, "selected_covars_df"):
            try:
                return self._selection_widget.selected_covars_df()
            except Exception:
                return None
        if self._stack_prepare_widget is None:
            return None
        if not hasattr(self._stack_prepare_widget, "selected_covars_df"):
            return None
        try:
            return self._stack_prepare_widget.selected_covars_df()
        except Exception:
            return None

    def _selection_suffix(self):
        if self._selection_widget is not None and hasattr(self._selection_widget, "current_selection_suffix"):
            try:
                return self._selection_widget.current_selection_suffix()
            except Exception:
                return ("all", "all")
        return ("all", "all")

    def _harmonization_columns(self):
        if self._selection_widget is not None and hasattr(self._selection_widget, "covars_columns"):
            try:
                return [str(col) for col in self._selection_widget.covars_columns()]
            except Exception:
                return []
        covars_df = self._filtered_covars_df()
        if covars_df is None:
            return []
        return [str(col) for col in covars_df.columns]

    def _harm_default_batch_col(self) -> str:
        if self._selection_widget is not None and hasattr(self._selection_widget, "default_batch_col"):
            try:
                return str(self._selection_widget.default_batch_col() or "")
            except Exception:
                return ""
        columns = self._harmonization_columns()
        preferred = {"scanner", "site", "batch"}
        for name in columns:
            if name.strip().lower() in preferred:
                return name
        return columns[0] if columns else ""

    def _harm_covariate_type(self, covar_name: str) -> str:
        if self._selection_widget is not None and hasattr(self._selection_widget, "covariate_type"):
            try:
                return str(self._selection_widget.covariate_type(covar_name) or "categorical")
            except Exception:
                return "categorical"
        covars_df = self._filtered_covars_df()
        if covars_df is None or covar_name not in covars_df.columns:
            return "categorical"
        values = covars_df[covar_name].tolist()
        return "continuous" if _column_is_numeric(values) else "categorical"

    def _harm_is_id_like(self, covar_name: str) -> bool:
        if self._selection_widget is not None and hasattr(self._selection_widget, "is_id_like"):
            try:
                return bool(self._selection_widget.is_id_like(covar_name))
            except Exception:
                return False
        return str(covar_name).strip().lower() in {
            "participant_id",
            "subject_id",
            "session_id",
            "id",
            "sub",
            "ses",
        }

    def _selected_modalities(self):
        if self._stack_prepare_widget is not None and hasattr(self._stack_prepare_widget, "detected_modalities"):
            try:
                modalities = list(self._stack_prepare_widget.detected_modalities())
            except Exception:
                modalities = []
        else:
            modalities = []
        if not modalities:
            modalities = sorted({_infer_modality_from_path(Path(path)) for path in self.selected_paths() if str(path).strip()})
        return [item for item in modalities if item]

    def _ordered_modalities(self):
        modalities = self._selected_modalities()
        reference = self.reference_modality_combo.currentText().strip()
        if reference and reference in modalities:
            return [reference] + [item for item in modalities if item != reference]
        return modalities

    def _multimodal_process_config(self):
        if not self.multimodal_enable_check.isChecked():
            return None
        effective_paths = [Path(path) for path in self.effective_selected_paths()]
        ordered_modalities = []
        for modality in self._ordered_modalities():
            if any(_infer_modality_from_path(path) == modality for path in effective_paths):
                ordered_modalities.append(modality)
        if len(ordered_modalities) <= 1:
            return None
        return {
            "enabled": True,
            "ordered_modalities": ordered_modalities,
            "align_cross_modality": bool(self.multimodal_align_check.isChecked()),
        }

    def _build_pair_table_rows(self, paths, modalities=None, include_empty_covar_rows=False):
        selected_paths = [Path(path) for path in (paths or []) if str(path).strip()]
        ordered_modalities = [str(item).strip() for item in (modalities or self._ordered_modalities()) if str(item).strip()]
        if not ordered_modalities:
            ordered_modalities = sorted(
                {
                    _infer_modality_from_path(path)
                    for path in selected_paths
                    if str(_infer_modality_from_path(path)).strip()
                }
            )

        rows = []
        extra_columns = []
        covars_df = self._filtered_covars_df()
        if covars_df is not None and len(covars_df) > 0:
            covar_columns = [str(col) for col in covars_df.columns]
            col_map = {str(col).lower(): col for col in covars_df.columns}
            group_col = col_map.get("group")
            sub_col = (
                col_map.get("participant_id")
                or col_map.get("subject_id")
                or col_map.get("subject")
                or col_map.get("sub")
                or col_map.get("id")
            )
            ses_col = col_map.get("session_id") or col_map.get("session") or col_map.get("ses")
            hidden_cols = {group_col, sub_col, ses_col}
            extra_columns = [col for col in covar_columns if col not in hidden_cols and col is not None]
            default_group = self._group_from_path(selected_paths[0]) if selected_paths else ""
            for _, covar_row in covars_df.iterrows():
                group, sub, ses = self._covar_row_subject_session(covar_row, default_group, col_map=col_map)
                key = (sub, ses)
                matched_paths = self._paths_for_subject_session(selected_paths, sub, ses)
                if not matched_paths and not include_empty_covar_rows:
                    continue
                counts = self._modality_counts_for_paths(matched_paths, ordered_modalities)
                matched_by_modality = self._modality_paths_for_paths(matched_paths, ordered_modalities)
                row = {
                    "group": group,
                    "sub": sub,
                    "ses": ses,
                    "_key": key,
                    "_counts": counts,
                    "_paths_by_modality": matched_by_modality,
                    "_match_count": len(matched_paths),
                }
                for modality in ordered_modalities:
                    row[modality] = counts.get(modality, 0)
                for column in extra_columns:
                    row[column] = str(covar_row[column])
                rows.append(row)
        else:
            pair_rows = {}
            for path in selected_paths:
                group, sub, ses = self._path_subject_session(path)
                key = (sub, ses)
                if key not in pair_rows:
                    pair_rows[key] = {
                        "group": group,
                        "sub": sub,
                        "ses": ses,
                        "_key": key,
                    }
            for key in sorted(pair_rows.keys()):
                row = dict(pair_rows[key])
                matched_paths = self._paths_for_subject_session(selected_paths, row["sub"], row["ses"])
                counts = self._modality_counts_for_paths(matched_paths, ordered_modalities)
                matched_by_modality = self._modality_paths_for_paths(matched_paths, ordered_modalities)
                row["_counts"] = counts
                row["_paths_by_modality"] = matched_by_modality
                row["_match_count"] = len(matched_paths)
                for modality in ordered_modalities:
                    row[modality] = counts.get(modality, 0)
                rows.append(row)
        return rows, ordered_modalities, extra_columns

    def _populate_pair_table(self, table, rows, modalities, extra_columns, include_exclude_column=False):
        table.blockSignals(True)
        table.clear()
        headers = ["group", "sub", "ses"] + list(modalities) + list(extra_columns)
        if include_exclude_column:
            headers = ["Exclude"] + headers
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))

        duplicate_entries = []
        editable_flag = getattr(Qt, "ItemIsEditable", getattr(Qt.ItemFlag, "ItemIsEditable"))
        align_center = getattr(Qt, "AlignCenter", getattr(Qt.AlignmentFlag, "AlignCenter"))
        data_offset = 0
        if include_exclude_column:
            data_offset = 1

        for row_idx, row_data in enumerate(rows):
            pair_key = row_data.get("_key")
            if include_exclude_column:
                exclude_item = QTableWidgetItem("")
                exclude_item.setData(USER_ROLE, pair_key)
                exclude_item.setFlags((exclude_item.flags() | _is_user_checkable_flag()) & ~editable_flag)
                exclude_item.setCheckState(
                    Qt.Checked if pair_key in self._multimodal_excluded_pairs else Qt.Unchecked
                )
                table.setItem(row_idx, 0, exclude_item)

            for col_idx, header in enumerate(headers[data_offset:], start=data_offset):
                if header in modalities:
                    count = int(row_data.get(header, 0) or 0)
                    modality_paths = list(row_data.get("_paths_by_modality", {}).get(header, []))
                    if count <= 0:
                        item = QTableWidgetItem("✗")
                        item.setToolTip("Missing")
                        item.setForeground(QColor("#dc2626"))
                    elif count == 1:
                        item = QTableWidgetItem("✓")
                        item.setToolTip(modality_paths[0] if modality_paths else "Present")
                        item.setForeground(QColor("#16a34a"))
                    else:
                        item = QTableWidgetItem(f"o {count}")
                        item.setToolTip(
                            "\n".join(modality_paths)
                            if modality_paths
                            else f"{count} matching NPZ files detected"
                        )
                        item.setForeground(QColor("#ca8a04"))
                        duplicate_entries.append(
                            {
                                "pair": pair_key,
                                "modality": header,
                                "count": count,
                            }
                        )
                    item.setTextAlignment(int(align_center))
                else:
                    item = QTableWidgetItem(str(row_data.get(header, "")))
                item.setFlags(item.flags() & ~editable_flag)
                table.setItem(row_idx, col_idx, item)

        header = table.horizontalHeader()
        fixed_headers = {"group", "sub", "ses", *modalities}
        if include_exclude_column:
            fixed_headers.add("Exclude")
        if QT_LIB == 6:
            for col_idx, header_name in enumerate(headers):
                mode = (
                    QHeaderView.ResizeMode.Stretch
                    if header_name not in fixed_headers
                    else QHeaderView.ResizeMode.ResizeToContents
                )
                header.setSectionResizeMode(col_idx, mode)
        else:
            for col_idx, header_name in enumerate(headers):
                mode = QHeaderView.Stretch if header_name not in fixed_headers else QHeaderView.ResizeToContents
                header.setSectionResizeMode(col_idx, mode)
        table.blockSignals(False)
        return duplicate_entries

    def _refresh_run_table(self):
        if not hasattr(self, "run_table") or self.run_table is None:
            return

        ready = self._stack_prepare_widget is not None and self._selection_widget is not None
        if not ready:
            self.run_table.clear()
            self.run_table.setRowCount(0)
            self.run_table.setColumnCount(0)
            self.run_summary_label.setText(
                "The stack preview will appear here after Selection and Setup are ready."
            )
            return

        effective_paths = [Path(path) for path in self.effective_selected_paths()]
        rows, modalities, extra_columns = self._build_pair_table_rows(
            effective_paths,
            modalities=self._ordered_modalities(),
            include_empty_covar_rows=False,
        )
        duplicate_entries = self._populate_pair_table(
            self.run_table,
            rows,
            modalities,
            extra_columns,
            include_exclude_column=False,
        )
        duplicate_pair_count = len({tuple(entry["pair"]) for entry in duplicate_entries})
        validation_error = self.multimodal_validation_error()
        if not effective_paths:
            self.run_summary_label.setText(
                "No NPZ files remain after the current covariate and multimodal filters."
            )
            return
        if validation_error:
            self.run_summary_label.setText(
                f"{len(rows)} subject/session pairs will be stacked from {len(effective_paths)} NPZ file(s). "
                f"Processing is blocked: {validation_error}"
            )
            return
        if duplicate_pair_count:
            self.run_summary_label.setText(
                f"{len(rows)} subject/session pairs will be stacked from {len(effective_paths)} NPZ file(s). "
                f"Duplicate NPZ files remain for {duplicate_pair_count} sub/ses pair(s)."
            )
            return
        self.run_summary_label.setText(
            f"{len(rows)} subject/session pairs will be stacked from {len(effective_paths)} NPZ file(s)."
        )

    def _update_multimodal_controls(self):
        modalities = self._selected_modalities()
        has_multiple = len(modalities) > 1
        enabled = has_multiple and self.multimodal_enable_check.isChecked()
        self.reference_modality_combo.setEnabled(enabled)
        self.multimodal_align_check.setEnabled(enabled)
        self.multimodal_table.setVisible(enabled)
        if not has_multiple:
            self.multimodal_enable_check.setChecked(False)
            self.multimodal_enable_check.setEnabled(False)
            self.multimodal_align_check.setChecked(False)
            self.multimodal_align_check.setEnabled(False)
            self.multimodal_summary_label.setText(
                "Multimodal stacking becomes available when more than one modality is detected in the selected NPZ files."
            )
        else:
            self.multimodal_enable_check.setEnabled(True)
            self.multimodal_summary_label.setText(
                "Optional step. Enable multimodal stacking to review subject/session coverage across modalities."
            )
        self.reference_modality_combo.setVisible(has_multiple)
        self._refresh_multimodal_table()
        if self._stack_prepare_widget is not None and hasattr(self._stack_prepare_widget, "refresh_process_state"):
            self._stack_prepare_widget.refresh_process_state()
        if self._run_preview_visible():
            self._refresh_run_table()

    def _refresh_optional_steps(self):
        modalities = self._selected_modalities()
        current_reference = self.reference_modality_combo.currentText().strip()
        self.reference_modality_combo.blockSignals(True)
        self.reference_modality_combo.clear()
        self.reference_modality_combo.addItems(modalities)
        if current_reference and current_reference in modalities:
            self.reference_modality_combo.setCurrentText(current_reference)
        self.reference_modality_combo.blockSignals(False)
        ready = self._stack_prepare_widget is not None and self._selection_widget is not None
        for idx in (1, 2, 3, 4):
            self._step_buttons[idx].setEnabled(ready)
        self._update_multimodal_controls()
        if self._run_preview_visible():
            self._refresh_run_table()
        self._sync_workflow_process_button()

    def _schedule_refresh_optional_steps(self):
        if self._optional_steps_refresh_pending:
            return
        self._optional_steps_refresh_pending = True

        def _run_refresh():
            self._optional_steps_refresh_pending = False
            self._refresh_optional_steps()

        try:
            QTimer.singleShot(0, _run_refresh)
        except Exception:
            self._optional_steps_refresh_pending = False
            self._refresh_optional_steps()

    def _refresh_multimodal_table(self):
        self._multimodal_table_refreshing = True
        self._multimodal_duplicate_entries = []
        if not self.multimodal_enable_check.isChecked():
            self.multimodal_table.setRowCount(0)
            self.multimodal_table.setColumnCount(0)
            self._multimodal_table_refreshing = False
            return

        selected_paths = [Path(path) for path in self.selected_paths()]
        modalities = self._ordered_modalities()
        if len(modalities) <= 1:
            self.multimodal_table.setRowCount(0)
            self.multimodal_table.setColumnCount(0)
            self._multimodal_table_refreshing = False
            return

        rows, modalities, extra_columns = self._build_pair_table_rows(
            selected_paths,
            modalities=modalities,
            include_empty_covar_rows=True,
        )
        self._multimodal_duplicate_entries = self._populate_pair_table(
            self.multimodal_table,
            rows,
            modalities,
            extra_columns,
            include_exclude_column=True,
        )
        excluded_count = len([row for row in rows if row.get("_key") in self._multimodal_excluded_pairs])
        duplicate_pair_count = len({tuple(entry["pair"]) for entry in self._multimodal_duplicate_entries})
        if duplicate_pair_count:
            self.multimodal_summary_label.setText(
                "Optional step. "
                f"{len(rows)} subject/session pairs shown. Excluded: {excluded_count}. "
                f"Warning: duplicate NPZ files detected for {duplicate_pair_count} sub/ses pair(s). "
                "Processing is blocked until duplicates are removed or excluded."
            )
        else:
            self.multimodal_summary_label.setText(
                f"Optional step. {len(rows)} subject/session pairs shown. Excluded: {excluded_count}."
            )
        self._multimodal_table_refreshing = False

    def _on_multimodal_table_item_changed(self, item):
        if item is None or self._multimodal_table_refreshing or item.column() != 0:
            return
        pair_key = item.data(USER_ROLE)
        if not pair_key:
            return
        if item.checkState() == Qt.Checked:
            self._multimodal_excluded_pairs.add(tuple(pair_key))
        else:
            self._multimodal_excluded_pairs.discard(tuple(pair_key))
        self._refresh_multimodal_table()
        if self._run_preview_visible():
            self._refresh_run_table()
        if self._stack_prepare_widget is not None and hasattr(self._stack_prepare_widget, "refresh_process_state"):
            self._stack_prepare_widget.refresh_process_state()

    def multimodal_validation_error(self):
        if not self.multimodal_enable_check.isChecked():
            return ""
        duplicate_pair_count = len({tuple(entry["pair"]) for entry in self._multimodal_duplicate_entries})
        if duplicate_pair_count:
            return (
                f"Multiple NPZ files were found for {duplicate_pair_count} sub/ses pair(s) in the Multimodal step. "
                "Remove or exclude the duplicates before processing."
            )
        return ""

    def _sync_workflow_process_button(self):
        can_process = False
        can_import = False
        processing = self._stack_processing()
        if self._stack_prepare_widget is not None:
            if hasattr(self._stack_prepare_widget, "can_process"):
                try:
                    can_process = bool(self._stack_prepare_widget.can_process())
                except Exception:
                    can_process = False
            elif hasattr(self._stack_prepare_widget, "process_button"):
                try:
                    can_process = bool(self._stack_prepare_widget.process_button.isEnabled())
                except Exception:
                    can_process = False
            if hasattr(self._stack_prepare_widget, "can_import_last_output"):
                try:
                    can_import = bool(self._stack_prepare_widget.can_import_last_output())
                except Exception:
                    can_import = False
            elif hasattr(self._stack_prepare_widget, "import_button"):
                try:
                    can_import = bool(self._stack_prepare_widget.import_button.isEnabled())
                except Exception:
                    can_import = False
        if hasattr(self, "run_import_button") and self.run_import_button is not None:
            self.run_import_button.setEnabled(can_import)
        if hasattr(self, "run_process_button") and self.run_process_button is not None:
            self.run_process_button.setEnabled(can_process)
        if hasattr(self, "run_close_button") and self.run_close_button is not None:
            self.run_close_button.setEnabled(not processing)

    def _trigger_workflow_process(self):
        if self._stack_prepare_widget is None:
            return
        if hasattr(self._stack_prepare_widget, "trigger_process"):
            self._stack_prepare_widget.trigger_process()
        else:
            self._stack_prepare_widget._process()
        self._sync_workflow_process_button()

    def _trigger_workflow_import(self):
        if self._stack_prepare_widget is None:
            return
        if hasattr(self._stack_prepare_widget, "trigger_import_last_output"):
            self._stack_prepare_widget.trigger_import_last_output()
        else:
            self._stack_prepare_widget._import_last_output()
        self._sync_workflow_process_button()

    def set_theme(self, theme_name: str) -> None:
        theme = str(theme_name or "Dark").strip().title()
        if theme not in {"Light", "Dark", "Teya", "Donald"}:
            theme = "Dark"
        if theme == "Dark":
            style = (
                "QWidget { background: #1f2430; color: #e5e7eb; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QListWidget, QTableWidget { "
                "background: #2a3140; color: #e5e7eb; border: 1px solid #556070; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #344054; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #2563eb; border: 2px solid #60a5fa; color: #ffffff; font-weight: 600; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QListWidget::item:selected, QTableWidget::item:selected { background: #3b82f6; color: #ffffff; }"
            )
        elif theme == "Teya":
            style = (
                "QWidget { background: #ffd0e5; color: #0b7f7a; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QListWidget, QTableWidget { "
                "background: #ffe6f1; color: #0b7f7a; border: 1px solid #1db8b2; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #ffd9ea; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #2ecfc9; border: 2px solid #0b7f7a; color: #073f3c; font-weight: 700; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QListWidget::item:selected, QTableWidget::item:selected { background: #2ecfc9; color: #073f3c; }"
            )
        elif theme == "Donald":
            style = (
                "QWidget { background: #d97706; color: #ffffff; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QListWidget, QTableWidget { "
                "background: #c96a04; color: #ffffff; border: 1px solid #f3a451; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #c76b06; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #b85f00; border: 2px solid #ffd19e; color: #ffffff; font-weight: 700; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QListWidget::item:selected, QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
            )
        else:
            style = (
                "QWidget { background: #f4f6f9; color: #1f2937; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QListWidget, QTableWidget { "
                "background: #ffffff; color: #1f2937; border: 1px solid #c9d0da; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #edf2f7; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #2563eb; border: 2px solid #1d4ed8; color: #ffffff; font-weight: 600; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QListWidget::item:selected, QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
            )
        self.setStyleSheet(style)

    def _accept_if_selection(self) -> None:
        if not self.selected_paths():
            QMessageBox.information(self, "No Files Selected", "Select at least one matrix to add.")
            return
        self._requested_action = "add"
        self.accept()

    def _open_stack_prepare(self) -> None:
        selected = self.selected_paths()
        if not selected:
            QMessageBox.information(self, "No Files Selected", "Select at least one matrix to stack.")
            return
        selected_atlases = sorted({_connectivity_atlas_tag(Path(path)) for path in selected})
        if len(selected_atlases) > 1:
            QMessageBox.warning(
                self,
                "Mixed Atlases",
                "Stack requires a single atlas selection.\n\n"
                f"Detected atlases: {', '.join(selected_atlases)}\n\n"
                "Use the text filter or a narrower modality selection before stacking.",
            )
            return
        try:
            from window.stack_prepare import CovarsSelectionWidget, StackPrepareDialog
        except Exception:
            try:
                from mrsi_viewer.window.stack_prepare import CovarsSelectionWidget, StackPrepareDialog
            except Exception as exc:
                QMessageBox.warning(self, "Stack Unavailable", f"Failed to open stack step: {exc}")
                return

        if self._selection_widget is None:
            self._selection_widget = CovarsSelectionWidget(
                selected_paths=selected,
                parent=self.selection_page,
            )
            _write_diagnostic_line("Created embedded selection_widget with parent=selection_page")
            try:
                self._selection_widget.destroyed.connect(
                    lambda *_args: _write_diagnostic_line("embedded selection_widget destroyed")
                )
            except Exception:
                pass
            self._selection_widget.configuration_changed.connect(self._schedule_refresh_optional_steps)
            self.selection_tab_layout.removeWidget(self.selection_placeholder)
            self.selection_placeholder.hide()
            self.selection_tab_layout.addWidget(self._selection_widget, 1)
        else:
            self._selection_widget.set_selected_paths(selected)

        if self._stack_prepare_widget is None:
            if hasattr(self, "workflow_terminal") and self.workflow_terminal is not None:
                self.workflow_terminal.clear()
            self._stack_prepare_widget = StackPrepareDialog(
                selected_paths=selected,
                theme_name=getattr(self.parent(), "_theme_name", "Dark"),
                export_callback=self._export_callback,
                close_callback=lambda: self._go_to_step(1),
                selected_paths_provider=self.effective_selected_paths,
                validation_error_provider=self.multimodal_validation_error,
                covars_df_provider=self._filtered_covars_df,
                selection_suffix_provider=self._selection_suffix,
                multimodal_config_provider=self._multimodal_process_config,
                include_covars_widget=False,
                show_embedded_terminal=False,
                show_import_button=False,
                show_process_button=False,
                default_results_dir=getattr(self.parent(), "_results_dir_default", ""),
                default_bids_dir=getattr(self.parent(), "_bids_dir_default", ""),
                default_atlas_dir=getattr(self.parent(), "_atlas_dir_default", ""),
                parent=self.setup_page,
            )
            _write_diagnostic_line("Created embedded stack_prepare_widget with parent=setup_page")
            try:
                self._stack_prepare_widget.destroyed.connect(
                    lambda *_args: _write_diagnostic_line("embedded stack_prepare_widget destroyed")
                )
            except Exception:
                pass
            if hasattr(self._stack_prepare_widget, "configuration_changed"):
                self._stack_prepare_widget.configuration_changed.connect(self._schedule_refresh_optional_steps)
            if hasattr(self._stack_prepare_widget, "process_state_changed"):
                self._stack_prepare_widget.process_state_changed.connect(self._sync_workflow_process_button)
            if hasattr(self._stack_prepare_widget, "log_message_emitted"):
                self._stack_prepare_widget.log_message_emitted.connect(self._append_workflow_terminal)
            self.setup_tab_layout.removeWidget(self.setup_placeholder)
            self.setup_placeholder.hide()
            self.setup_tab_layout.addWidget(self._stack_prepare_widget, 1)
        else:
            self._stack_prepare_widget.set_selected_paths(selected)
            self._stack_prepare_widget.show()
        if hasattr(self._stack_prepare_widget, "refresh_process_state"):
            self._stack_prepare_widget.refresh_process_state()
        self._refresh_optional_steps()
        self._go_to_step(1)

class ConnectomeViewer(QMainWindow):
    _global_font_adjusted = False

    def __init__(self) -> None:
        super().__init__()
        self._increase_global_font_size()
        self._entries = {}
        self._derived_counter = 0
        self.titles = {}
        self._covars_cache = {}
        self._valid_keys_cache = {}
        self._parcel_metadata_cache = {}
        self._group_values_cache = {}
        self._current_matrix = None
        self._current_parcel_labels = None
        self._current_parcel_names = None
        self._current_axes = None
        self._zoom_selector = None
        self._matrix_full_xlim = None
        self._matrix_full_ylim = None
        self._colorbar = ColorBar()
        self._custom_cmaps = set()
        self._preferences = self._load_preferences()
        self._last_gradients = None
        self._active_parcellation_path = None
        self._active_parcellation_img = None
        self._active_parcellation_data = None
        self._gradients_busy = False
        self._gradients_progress_state = {"minimum": 0, "maximum": 1, "value": 0, "text": "Idle"}
        self._surface_dialog = None
        self._gradient_scatter_dialog = None
        self._gradient_classification_dialog = None
        self._gradients_dialog = None
        self._nbs_dialog = None
        self._selector_dialog = None
        self._harmonize_dialog = None
        self._batch_import_dialog = None
        self._stack_prepare_dialog = None
        self._label_info_dialog = None
        self._participants_info_dialog = None
        self._left_panel_saved_width = 320
        self._right_panel_saved_width = 240
        self._plot_title_full = ""
        self._plot_title_tooltip = ""
        self._export_grid_columns = 4
        self._export_grid_rotate = False
        self._export_grid_output_path = ""
        self._export_grid_selected_filter = "PDF (*.pdf)"
        self._theme_name = self._preferences["theme"]
        self._default_matrix_colormap = self._preferences["matrix_colormap"]
        self._default_gradient_colormap = self._preferences["gradient_colormap"]
        self._gradient_colormap_name = self._default_gradient_colormap
        self._gradient_selected_entry_id = None
        self._gradient_precomputed_bundle = None
        self._gradient_precomputed_selected_row = None
        self._gradient_use_precomputed_bundle = False
        self._gradient_component_count = 4
        self._gradient_hemisphere_mode = "both"
        self._gradient_surface_mesh = "fsaverage4"
        self._gradient_surface_render_count = 1
        self._gradient_surface_procrustes = False
        self._gradient_classification_surface_mesh = self._gradient_surface_mesh
        self._gradient_classification_hemisphere_mode = "both"
        self._gradient_scatter_rotation = "Default"
        self._gradient_scatter_triangular_rgb = False
        self._gradient_classification_fit_mode = "triangle"
        self._gradient_triangular_color_order = "RBG"
        self._gradient_classification_colormap_name = self._default_gradient_colormap
        self._gradient_classification_component = "1"
        self._gradient_classification_x_axis = "gradient2"
        self._gradient_classification_y_axis = "gradient1"
        self._gradient_classification_ignore_lh_parcel = ""
        self._gradient_classification_ignore_rh_parcel = ""
        self._gradient_classification_adjacency_path = ""
        self._gradient_classification_adjacency_cache = None
        self._gradient_network_component = "all"
        self._gradient_component_rotations = ["Default"] * 10
        self._matlab_cmd_default = self._preferences["matlab_cmd"]
        self._matlab_nbs_path_default = self._preferences["matlab_nbs_path"]
        self._results_dir_default = self._preferences["results_dir"]
        self._bids_dir_default = self._preferences["bids_dir"]
        self._atlas_dir_default = self._preferences["atlas_dir"]
        self._zoom_level = int(self._preferences.get("zoom_level", 0))
        self._base_app_font_point_size = None
        self._hover_vline = None
        self._hover_hline = None
        app = QApplication.instance()
        if app is not None:
            app_font = app.font()
            if app_font.pointSize() > 0:
                self._base_app_font_point_size = app_font.pointSize()
        self._compact_ui = False
        self._styled_groups = []
        self.setWindowTitle("Donald")
        self._set_window_icon()
        self.setAcceptDrops(True)
        self._hist_dialog = None
        self._build_ui()
        self._build_menu_bar()
        self._apply_zoom_level(self._zoom_level)

    @classmethod
    def _increase_global_font_size(cls) -> None:
        if cls._global_font_adjusted:
            return
        app = QApplication.instance()
        if app is None:
            return
        screen = app.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            dpi = screen.logicalDotsPerInch()
            if geom.width() <= 1600 or geom.height() <= 900 or dpi >= 120.0:
                cls._global_font_adjusted = True
                return
        font = app.font()
        if font.pointSize() > 0:
            font.setPointSize(font.pointSize() + 1)
        elif font.pixelSize() > 0:
            font.setPixelSize(font.pixelSize() + 1)
        else:
            return
        app.setFont(font)
        cls._global_font_adjusted = True

    @staticmethod
    def _normalize_theme_name(theme_name: str) -> str:
        theme = str(theme_name or "Dark").strip().title()
        if theme not in {"Light", "Dark", "Teya", "Donald"}:
            theme = "Dark"
        return theme

    def _default_preferences(self):
        return {
            "theme": "Dark",
            "matrix_colormap": DEFAULT_COLORMAP,
            "gradient_colormap": DEFAULT_GRADIENT_COLORMAP,
            "matlab_cmd": DEFAULT_MATLAB_CMD,
            "matlab_nbs_path": DEFAULT_MATLAB_NBS_PATH,
            "results_dir": "",
            "bids_dir": _first_env_value("BIDSDATAPATH"),
            "atlas_dir": "",
            "zoom_level": 0,
        }

    def _available_colormap_names(self):
        custom = _discover_custom_cmaps()
        self._custom_cmaps = set(custom)
        names = []
        seen = set()
        for name in COLORMAPS + COLORBAR_BARS + custom:
            if name and name not in seen:
                names.append(name)
                seen.add(name)
        return names

    @staticmethod
    def _normalize_executable_path(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        path = Path(text).expanduser()
        if path.is_file():
            try:
                return str(path.resolve())
            except Exception:
                return str(path)
        resolved = shutil.which(text)
        if resolved:
            try:
                return str(Path(resolved).resolve())
            except Exception:
                return resolved
        return text

    @staticmethod
    def _normalize_directory_path(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        path = Path(text).expanduser()
        if path.is_dir():
            try:
                return str(path.resolve())
            except Exception:
                return str(path)
        return str(path)

    def _sanitize_preferences(self, raw_values):
        prefs = dict(self._default_preferences())
        if isinstance(raw_values, dict):
            prefs.update(raw_values)
        names = self._available_colormap_names()
        fallback_matrix = (
            DEFAULT_COLORMAP if DEFAULT_COLORMAP in names else (names[0] if names else DEFAULT_COLORMAP)
        )
        fallback_gradient = (
            DEFAULT_GRADIENT_COLORMAP
            if DEFAULT_GRADIENT_COLORMAP in names
            else (names[0] if names else DEFAULT_GRADIENT_COLORMAP)
        )
        prefs["theme"] = self._normalize_theme_name(prefs.get("theme"))
        matrix_name = str(prefs.get("matrix_colormap", fallback_matrix) or "").strip()
        gradient_name = str(prefs.get("gradient_colormap", fallback_gradient) or "").strip()
        prefs["matrix_colormap"] = matrix_name if matrix_name in names else fallback_matrix
        prefs["gradient_colormap"] = gradient_name if gradient_name in names else fallback_gradient
        matlab_cmd = str(prefs.get("matlab_cmd", DEFAULT_MATLAB_CMD) or "").strip()
        matlab_cmd = self._normalize_executable_path(matlab_cmd)
        default_matlab_cmd = self._normalize_executable_path(DEFAULT_MATLAB_CMD)
        prefs["matlab_cmd"] = matlab_cmd or default_matlab_cmd or ""
        matlab_nbs_path = str(prefs.get("matlab_nbs_path", DEFAULT_MATLAB_NBS_PATH) or "").strip()
        matlab_nbs_path = self._normalize_directory_path(matlab_nbs_path)
        default_nbs_path = self._normalize_directory_path(DEFAULT_MATLAB_NBS_PATH)
        prefs["matlab_nbs_path"] = matlab_nbs_path or default_nbs_path or ""
        prefs["results_dir"] = self._normalize_directory_path(str(prefs.get("results_dir", "") or "").strip())
        prefs["bids_dir"] = self._normalize_directory_path(str(prefs.get("bids_dir", "") or "").strip())
        prefs["atlas_dir"] = self._normalize_directory_path(str(prefs.get("atlas_dir", "") or "").strip())
        try:
            zoom_level = int(prefs.get("zoom_level", 0))
        except Exception:
            zoom_level = 0
        prefs["zoom_level"] = max(-6, min(12, zoom_level))
        return prefs

    def _validate_nbs_preferences(self) -> bool:
        matlab_cmd = self._normalize_executable_path(self._matlab_cmd_default)
        nbs_path = self._normalize_directory_path(self._matlab_nbs_path_default)

        matlab_ok = False
        if matlab_cmd:
            matlab_path = Path(matlab_cmd)
            if matlab_path.is_file():
                matlab_ok = True
            elif shutil.which(matlab_cmd):
                matlab_ok = True

        nbs_ok = bool(nbs_path) and Path(nbs_path).is_dir()

        if matlab_ok and nbs_ok:
            self._matlab_cmd_default = matlab_cmd
            self._matlab_nbs_path_default = nbs_path
            return True

        message = (
            "NBS is not configured.\n"
            "Go to Settings > Preferences and select valid MATLAB executable and NBS paths."
        )
        try:
            QMessageBox.warning(self, "NBS Configuration Required", message)
        except Exception:
            pass
        self.statusBar().showMessage(
            "NBS blocked: configure MATLAB and NBS paths in Settings > Preferences."
        )
        return False

    def _load_preferences(self):
        if not CONFIG_PATH.exists():
            return self._sanitize_preferences({})
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            raw = {}
        return self._sanitize_preferences(raw)

    def _save_preferences(self) -> bool:
        payload = {
            "theme": self._theme_name,
            "matrix_colormap": self._default_matrix_colormap,
            "gradient_colormap": self._default_gradient_colormap,
            "matlab_cmd": self._matlab_cmd_default,
            "matlab_nbs_path": self._matlab_nbs_path_default,
            "results_dir": self._results_dir_default,
            "bids_dir": self._bids_dir_default,
            "atlas_dir": self._atlas_dir_default,
            "zoom_level": int(self._zoom_level),
        }
        try:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            return True
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to save preferences: {exc}")
            return False

    def _build_menu_bar(self) -> None:
        bar = self.menuBar()
        bar.clear()

        file_menu = bar.addMenu("File")
        self.add_npz_action = QAction("Add NPZ", self)
        self.add_npz_action.setShortcut("Ctrl+O")
        self.add_npz_action.triggered.connect(self._open_files)
        file_menu.addAction(self.add_npz_action)

        file_menu.addSeparator()
        close_action = QAction("Exit", self)
        close_action.setShortcut("Ctrl+Q")
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)

        view_menu = bar.addMenu("View")
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut("Ctrl+=")
        zoom_in_action.triggered.connect(self._zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self._zoom_out)
        view_menu.addAction(zoom_out_action)

        zoom_reset_action = QAction("Reset Zoom", self)
        zoom_reset_action.setShortcut("Ctrl+0")
        zoom_reset_action.triggered.connect(self._zoom_reset)
        view_menu.addAction(zoom_reset_action)

        analysis_menu = bar.addMenu("Analysis")
        self.compute_gradients_action = QAction("Gradients", self)
        self.compute_gradients_action.triggered.connect(self._open_gradients_dialog)
        analysis_menu.addAction(self.compute_gradients_action)

        self.nbs_prepare_action = QAction("NBS Prepare", self)
        self.nbs_prepare_action.triggered.connect(self._open_nbs_prepare_dialog)
        analysis_menu.addAction(self.nbs_prepare_action)

        self.harmonize_prepare_action = QAction("Harmonize Prepare", self)
        self.harmonize_prepare_action.triggered.connect(self._open_harmonize_prepare_dialog)
        analysis_menu.addAction(self.harmonize_prepare_action)

        settings_menu = bar.addMenu("Settings")
        preferences_action = QAction("Preferences", self)
        preferences_action.triggered.connect(self._open_preferences_dialog)
        settings_menu.addAction(preferences_action)

        self._update_nbs_prepare_button()
        self._update_gradients_button()

    def _apply_zoom_level(self, zoom_level: int, show_status: bool = False) -> None:
        try:
            level = int(zoom_level)
        except Exception:
            level = 0
        level = max(-6, min(12, level))
        self._zoom_level = level
        if self._base_app_font_point_size is None:
            return
        app = QApplication.instance()
        if app is None:
            return
        target_pt = max(7, self._base_app_font_point_size + level)
        font = app.font()
        if font.pointSize() <= 0:
            return
        font.setPointSize(target_pt)
        app.setFont(font)
        if show_status:
            self.statusBar().showMessage(f"UI zoom: {target_pt} pt")

    def _zoom_in(self) -> None:
        self._apply_zoom_level(self._zoom_level + 1, show_status=True)

    def _zoom_out(self) -> None:
        self._apply_zoom_level(self._zoom_level - 1, show_status=True)

    def _zoom_reset(self) -> None:
        self._apply_zoom_level(0, show_status=True)

    def _open_preferences_dialog(self, _checked=False, *, initial_setup: bool = False) -> bool:
        names = self._available_colormap_names()
        dialog = PreferencesDialog(
            theme_name=self._theme_name,
            matrix_cmap=self._default_matrix_colormap,
            gradient_cmap=self._default_gradient_colormap,
            matlab_cmd=self._matlab_cmd_default,
            matlab_nbs_path=self._matlab_nbs_path_default,
            results_dir=self._results_dir_default,
            bids_dir=self._bids_dir_default,
            atlas_dir=self._atlas_dir_default,
            colormap_names=names,
            dialog_title="Initial Configuration" if initial_setup else "Preferences",
            require_results_dir=True,
            parent=self,
        )
        if dialog.exec() != _dialog_accepted_code():
            return False
        new_values = dialog.values()
        new_values["zoom_level"] = self._zoom_level
        prefs = self._sanitize_preferences(new_values)
        self._theme_name = prefs["theme"]
        self._default_matrix_colormap = prefs["matrix_colormap"]
        self._default_gradient_colormap = prefs["gradient_colormap"]
        self._gradient_colormap_name = self._default_gradient_colormap
        self._matlab_cmd_default = prefs["matlab_cmd"]
        self._matlab_nbs_path_default = prefs["matlab_nbs_path"]
        self._results_dir_default = prefs["results_dir"]
        self._bids_dir_default = prefs["bids_dir"]
        self._atlas_dir_default = prefs["atlas_dir"]
        self._preferences = prefs

        if getattr(self, "_nbs_dialog", None) is not None:
            try:
                self._nbs_dialog._matlab_cmd_default = self._matlab_cmd_default
                self._nbs_dialog._matlab_nbs_path_default = self._matlab_nbs_path_default
            except Exception:
                pass

        self._apply_theme(self._theme_name)
        self._reload_colormaps()
        if hasattr(self, "cmap_combo") and self.cmap_combo.findText(self._default_matrix_colormap) >= 0:
            self.cmap_combo.setCurrentText(self._default_matrix_colormap)
        self._sync_gradients_dialog_state()

        if self._save_preferences():
            self.statusBar().showMessage(f"Preferences saved to {CONFIG_PATH}.")
        return True

    def _build_ui(self) -> None:
        screen = QApplication.primaryScreen()
        screen_geom = screen.availableGeometry() if screen is not None else None
        screen_dpi = screen.logicalDotsPerInch() if screen is not None else 96.0
        compact_ui = False
        if screen_geom is not None:
            compact_ui = screen_geom.width() <= 1600 or screen_geom.height() <= 900
        compact_ui = compact_ui or screen_dpi >= 120.0
        self._compact_ui = compact_ui

        central = QWidget(self)
        main_layout = QHBoxLayout(central)
        if compact_ui:
            main_layout.setContentsMargins(6, 6, 6, 6)
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(8 if compact_ui else 10)
        self.main_splitter.setChildrenCollapsible(True)
        main_layout.addWidget(self.main_splitter)

        left_panel = QWidget()
        controls_layout = QVBoxLayout(left_panel)
        controls_layout.setSpacing(8 if compact_ui else 10)

        header = QLabel("Connectome matrices (.npz)")
        controls_layout.addWidget(header)

        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        self.export_button = QPushButton("Export Grid")
        self.export_button.clicked.connect(self._export_grid)
        export_layout.addWidget(self.export_button)

        self.write_matrix_button = QPushButton("Write to File")
        self.write_matrix_button.clicked.connect(self._write_selected_matrix_to_file)
        self.write_matrix_button.setEnabled(False)
        export_layout.addWidget(self.write_matrix_button)

        selector_group = QGroupBox("Selector")
        selector_layout = QVBoxLayout(selector_group)
        key_label = QLabel("Matrix key:")
        selector_layout.addWidget(key_label)

        self.key_combo = QComboBox()
        self.key_combo.currentIndexChanged.connect(self._on_key_changed)
        selector_layout.addWidget(self.key_combo)

        sample_label = QLabel("Sample index:")
        selector_layout.addWidget(sample_label)

        sample_layout = QHBoxLayout()
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(-1, 0)
        self.sample_spin.setSpecialValueText("Average")
        self.sample_spin.setEnabled(False)
        self.sample_spin.valueChanged.connect(self._on_sample_changed)
        sample_layout.addWidget(self.sample_spin)
        self.sample_add_button = QPushButton("Add")
        self.sample_add_button.clicked.connect(self._add_sample_entry)
        self.sample_add_button.setEnabled(False)
        sample_layout.addWidget(self.sample_add_button)
        selector_layout.addLayout(sample_layout)

        covar_label = QLabel("Covariate:")
        selector_layout.addWidget(covar_label)

        self.covar_combo = QComboBox()
        selector_layout.addWidget(self.covar_combo)

        covar_value_label = QLabel("Covar value:")
        self.covar_value_edit = QLineEdit("")
        self.covar_value_edit.setPlaceholderText("Numeric value")
        # Hidden legacy selector widgets kept for compatibility with existing
        # internals; aggregation now runs via popup dialog.
        covar_label.hide()
        self.covar_combo.hide()
        covar_value_label.hide()
        self.covar_value_edit.hide()

        self.selector_prepare_button = QPushButton("Prepare")
        self.selector_prepare_button.clicked.connect(self._open_selector_prepare_dialog)
        self.selector_prepare_button.setEnabled(False)
        selector_layout.addWidget(self.selector_prepare_button)

        self.cmap_combo = QComboBox()
        self.cmap_combo.currentIndexChanged.connect(self._on_colormap_changed)

        list_group = QGroupBox("Matrices")
        list_layout = QVBoxLayout(list_group)
        self.add_button = QPushButton("Add Files")
        self.add_button.clicked.connect(self._open_files)
        list_layout.addWidget(self.add_button)

        self.add_batch_button = QPushButton("Add Batch")
        self.add_batch_button.clicked.connect(self._open_batch_folder)
        list_layout.addWidget(self.add_batch_button)

        self.clear_button = QPushButton("Clear List")
        self.clear_button.clicked.connect(self._clear_files)
        list_layout.addWidget(self.clear_button)

        self.hist_button = QPushButton("View Histogram")
        self.hist_button.clicked.connect(self._open_histogram)
        list_layout.addWidget(self.hist_button)

        title_label = QLabel("Plot title:")
        list_layout.addWidget(title_label)

        self.title_edit = QLineEdit("")
        self.title_edit.setPlaceholderText("Defaults to file name")
        self.title_edit.editingFinished.connect(self._on_title_edited)
        self.title_edit.returnPressed.connect(self._on_title_edited)
        list_layout.addWidget(self.title_edit)

        self.file_list = QListWidget()
        if QT_LIB == 6:
            self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        else:
            self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.currentItemChanged.connect(self._on_selection_changed)
        list_layout.addWidget(self.file_list)

        move_layout = QHBoxLayout()
        self.move_up_button = QPushButton("Move Up")
        self.move_up_button.clicked.connect(lambda: self._move_selected(-1))
        move_layout.addWidget(self.move_up_button)
        self.move_down_button = QPushButton("Move Down")
        self.move_down_button.clicked.connect(lambda: self._move_selected(1))
        move_layout.addWidget(self.move_down_button)
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self._remove_selected)
        move_layout.addWidget(self.remove_button)
        list_layout.addLayout(move_layout)

        hint = QLabel("Drag & drop .npz files here.")
        hint.setWordWrap(True)
        list_layout.addWidget(hint)

        info_group = QGroupBox("Info")
        info_layout = QVBoxLayout(info_group)
        self.view_labels_button = QPushButton("View Labels")
        self.view_labels_button.clicked.connect(self._open_label_info_dialog)
        self.view_labels_button.setEnabled(False)
        info_layout.addWidget(self.view_labels_button)
        self.view_participants_button = QPushButton("View Participants")
        self.view_participants_button.clicked.connect(self._open_participants_info_dialog)
        self.view_participants_button.setEnabled(False)
        info_layout.addWidget(self.view_participants_button)

        gradients_group = QGroupBox("Gradients")
        gradients_layout = QVBoxLayout(gradients_group)
        self.gradients_open_button = QPushButton("Gradients")
        self.gradients_open_button.clicked.connect(self._open_gradients_dialog)
        self.gradients_open_button.setEnabled(False)
        gradients_layout.addWidget(self.gradients_open_button)

        nbs_group = QGroupBox("NBS")
        nbs_layout = QVBoxLayout(nbs_group)
        self.nbs_prepare_button = QPushButton("Prepare")
        self.nbs_prepare_button.clicked.connect(self._open_nbs_prepare_dialog)
        self.nbs_prepare_button.setEnabled(False)
        nbs_layout.addWidget(self.nbs_prepare_button)

        harmonize_group = QGroupBox("Harmonize")
        harmonize_layout = QVBoxLayout(harmonize_group)
        self.harmonize_prepare_button = QPushButton("Prepare")
        self.harmonize_prepare_button.clicked.connect(self._open_harmonize_prepare_dialog)
        self.harmonize_prepare_button.setEnabled(False)
        harmonize_layout.addWidget(self.harmonize_prepare_button)

        self._reload_colormaps()

        self._styled_groups = [
            list_group,
            selector_group,
            info_group,
            export_group,
            gradients_group,
            nbs_group,
            harmonize_group,
        ]

        controls_layout.addWidget(list_group)
        controls_layout.addWidget(selector_group)
        controls_layout.addWidget(info_group)
        controls_layout.addWidget(export_group)

        controls_layout.addStretch(1)

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setSpacing(6 if compact_ui else 8)

        plot_toolbar = QHBoxLayout()
        plot_toolbar.setSpacing(6 if compact_ui else 8)
        self.left_sidebar_button = QPushButton("◀")
        self.left_sidebar_button.setFixedWidth(30)
        self.left_sidebar_button.clicked.connect(self._toggle_left_sidebar)
        plot_toolbar.addWidget(self.left_sidebar_button)

        self.plot_title_label = QLabel("")
        title_font = self.plot_title_label.font()
        if title_font.pointSize() > 0:
            title_font.setPointSize(max(title_font.pointSize() - 1, 9))
        self.plot_title_label.setFont(title_font)
        self.plot_title_label.setToolTip("")
        plot_toolbar.addWidget(self.plot_title_label, 1)

        plot_toolbar.addWidget(QLabel("Color map:"))
        self.cmap_combo.setMinimumWidth(150 if compact_ui else 190)
        plot_toolbar.addWidget(self.cmap_combo)

        plot_toolbar.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "Teya", "Donald"])
        self.theme_combo.setCurrentText(self._theme_name)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        self.theme_combo.setMinimumWidth(90 if compact_ui else 110)
        plot_toolbar.addWidget(self.theme_combo)

        plot_toolbar.addWidget(QLabel("Display scaling:"))
        self.display_auto_check = QCheckBox("Auto")
        self.display_auto_check.setChecked(True)
        self.display_auto_check.stateChanged.connect(self._on_display_scaling_changed)
        plot_toolbar.addWidget(self.display_auto_check)

        self.display_min_edit = QLineEdit("")
        self.display_min_edit.setPlaceholderText("Min")
        self.display_min_edit.setFixedWidth(78 if compact_ui else 90)
        self.display_min_edit.setEnabled(False)
        self.display_min_edit.editingFinished.connect(self._on_display_scaling_changed)
        plot_toolbar.addWidget(self.display_min_edit)

        self.display_max_edit = QLineEdit("")
        self.display_max_edit.setPlaceholderText("Max")
        self.display_max_edit.setFixedWidth(78 if compact_ui else 90)
        self.display_max_edit.setEnabled(False)
        self.display_max_edit.editingFinished.connect(self._on_display_scaling_changed)
        plot_toolbar.addWidget(self.display_max_edit)

        plot_toolbar.addWidget(QLabel("Scale:"))
        self.display_scale_combo = QComboBox()
        self.display_scale_combo.addItems(["Linear", "Log"])
        self.display_scale_combo.setCurrentText("Linear")
        self.display_scale_combo.currentTextChanged.connect(self._on_display_scaling_changed)
        self.display_scale_combo.setFixedWidth(78 if compact_ui else 92)
        plot_toolbar.addWidget(self.display_scale_combo)

        self.zoom_region_check = QCheckBox("Zoom region")
        self.zoom_region_check.setToolTip("Drag on the matrix to zoom into a region.")
        self.zoom_region_check.stateChanged.connect(self._on_zoom_region_toggled)
        plot_toolbar.addWidget(self.zoom_region_check)

        self.zoom_reset_button = QPushButton("Reset view")
        self.zoom_reset_button.setEnabled(False)
        self.zoom_reset_button.clicked.connect(self._reset_matrix_zoom)
        plot_toolbar.addWidget(self.zoom_reset_button)

        self.right_sidebar_button = QPushButton("▶")
        self.right_sidebar_button.setFixedWidth(30)
        self.right_sidebar_button.clicked.connect(self._toggle_right_sidebar)
        plot_toolbar.addWidget(self.right_sidebar_button)
        center_layout.addLayout(plot_toolbar)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        center_layout.addWidget(self.canvas, 1)

        self.hover_label = QLabel("")
        self.hover_label.setWordWrap(True)
        center_layout.addWidget(self.hover_label, 0)

        right_panel = QWidget()
        right_panel_layout = QVBoxLayout(right_panel)
        right_panel_layout.addWidget(gradients_group)
        right_panel_layout.addWidget(nbs_group)
        right_panel_layout.addWidget(harmonize_group)
        right_panel_layout.addStretch(1)
        left_panel.setMinimumWidth(0)
        right_panel.setMinimumWidth(0)

        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(center_panel)
        self.main_splitter.addWidget(right_panel)
        self.main_splitter.splitterMoved.connect(self._on_splitter_moved)
        self.main_splitter.setCollapsible(0, True)
        self.main_splitter.setCollapsible(1, False)
        self.main_splitter.setCollapsible(2, True)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        if screen_geom is not None:
            total_w = max(screen_geom.width(), 1000)
            left_w = int(total_w * (0.31 if compact_ui else 0.26))
            right_w = int(total_w * (0.15 if compact_ui else 0.18))
            center_w = max(total_w - left_w - right_w, 480)
            self.main_splitter.setSizes([left_w, center_w, right_w])
        else:
            self.main_splitter.setSizes([380, 900, 240])
        current_sizes = self.main_splitter.sizes()
        if len(current_sizes) == 3:
            if current_sizes[0] > 0:
                self._left_panel_saved_width = current_sizes[0]
            if current_sizes[2] > 0:
                self._right_panel_saved_width = current_sizes[2]

        self.setCentralWidget(central)
        self._apply_theme(self._theme_name)
        self._update_parcellation_label()
        self._refresh_sidebar_toggle_buttons()
        self._set_plot_title("")
        self._update_nbs_prepare_button()
        self.statusBar().showMessage("Ready.")

    def _set_window_icon(self) -> None:
        icon = _load_app_icon()
        if icon.isNull():
            return
        self.setWindowIcon(icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(icon)

    def _on_theme_changed(self, theme_name: str) -> None:
        self._apply_theme(theme_name)

    def _theme_icon_color(self) -> str:
        if self._theme_name == "Dark":
            return "#e8edf7"
        if self._theme_name == "Teya":
            return "#06b6b0"
        if self._theme_name == "Donald":
            return "#ffffff"
        return "#1f2937"

    def _base_theme_stylesheet(self) -> str:
        if self._theme_name == "Dark":
            return (
                "QMainWindow, QWidget { background-color: #1f2430; color: #e5e7eb; }"
                "QLabel { color: #e5e7eb; }"
                "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QTableWidget, QProgressBar {"
                "background: #2a3140; color: #e5e7eb; border: 1px solid #556070; border-radius: 5px; padding: 4px;"
                "}"
                "QPushButton {"
                "background: #2d3646; color: #e5e7eb; border: 1px solid #5f6d82; border-radius: 6px; padding: 5px 10px;"
                "}"
                "QPushButton:hover { background: #374256; }"
                "QPushButton:pressed { background: #2b3444; }"
                "QPushButton:disabled { color: #8e98a8; background: #252c38; border-color: #464f5e; }"
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; }"
                "QPushButton#workflowStepButton:checked { background: #2563eb; border: 2px solid #60a5fa; color: #ffffff; font-weight: 600; }"
                "QListWidget::item:selected, QTableWidget::item:selected { background: #3b82f6; color: #ffffff; }"
                "QComboBox QAbstractItemView { background: #2a3140; color: #e5e7eb; selection-background-color: #3b82f6; }"
                "QStatusBar { background: #242a35; color: #e5e7eb; border-top: 1px solid #3d4556; }"
                "QToolTip { background-color: #101722; color: #e5e7eb; border: 1px solid #4b5563; }"
            )
        if self._theme_name == "Teya":
            return (
                "QMainWindow, QWidget { background-color: #ffd0e5; color: #0b7f7a; }"
                "QLabel { color: #0b7f7a; }"
                "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QTableWidget, QProgressBar {"
                "background: #ffe6f1; color: #0b7f7a; border: 1px solid #1db8b2; border-radius: 5px; padding: 4px;"
                "}"
                "QPushButton {"
                "background: #ffc0dc; color: #0b7f7a; border: 1px solid #1db8b2; border-radius: 6px; padding: 5px 10px;"
                "}"
                "QPushButton:hover { background: #ffb1d5; }"
                "QPushButton:pressed { background: #ffa3cd; }"
                "QPushButton:disabled { color: #68a9a6; background: #ffd9ea; border-color: #87cfcb; }"
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; }"
                "QPushButton#workflowStepButton:checked { background: #2ecfc9; border: 2px solid #0b7f7a; color: #073f3c; font-weight: 700; }"
                "QListWidget::item:selected, QTableWidget::item:selected { background: #2ecfc9; color: #073f3c; }"
                "QComboBox QAbstractItemView { background: #ffe6f1; color: #0b7f7a; selection-background-color: #2ecfc9; }"
                "QStatusBar { background: #ffbddb; color: #0b7f7a; border-top: 1px solid #1db8b2; }"
                "QToolTip { background-color: #ffeef6; color: #0b7f7a; border: 1px solid #1db8b2; }"
            )
        if self._theme_name == "Donald":
            return (
                "QMainWindow, QWidget { background-color: #d97706; color: #ffffff; }"
                "QLabel { color: #ffffff; }"
                "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QTableWidget, QProgressBar {"
                "background: #c96a04; color: #ffffff; border: 1px solid #f3a451; border-radius: 5px; padding: 4px;"
                "}"
                "QPushButton {"
                "background: #b85f00; color: #ffffff; border: 1px solid #f3a451; border-radius: 6px; padding: 5px 10px;"
                "}"
                "QPushButton:hover { background: #c76b06; }"
                "QPushButton:pressed { background: #a85400; }"
                "QPushButton:disabled { color: #ffe3be; background: #d58933; border-color: #f0b97c; }"
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; }"
                "QPushButton#workflowStepButton:checked { background: #b85f00; border: 2px solid #ffd19e; color: #ffffff; font-weight: 700; }"
                "QListWidget::item:selected, QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
                "QComboBox QAbstractItemView { background: #c96a04; color: #ffffff; selection-background-color: #2563eb; }"
                "QStatusBar { background: #b85f00; color: #ffffff; border-top: 1px solid #f3a451; }"
                "QToolTip { background-color: #f08c19; color: #ffffff; border: 1px solid #ffd19e; }"
            )
        return (
            "QMainWindow, QWidget { background-color: #f3f5f8; color: #1f2937; }"
            "QLabel { color: #1f2937; }"
            "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QTableWidget, QProgressBar {"
            "background: #ffffff; color: #111827; border: 1px solid #c9d0da; border-radius: 5px; padding: 4px;"
            "}"
            "QPushButton {"
            "background: #ffffff; color: #1f2937; border: 1px solid #b7c0cc; border-radius: 6px; padding: 5px 10px;"
            "}"
            "QPushButton:hover { background: #edf2f7; }"
            "QPushButton:pressed { background: #e6ebf2; }"
            "QPushButton:disabled { color: #9099a5; background: #edf1f5; border-color: #d2d9e2; }"
            "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; }"
            "QPushButton#workflowStepButton:checked { background: #2563eb; border: 2px solid #1d4ed8; color: #ffffff; font-weight: 600; }"
            "QListWidget::item:selected, QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
            "QComboBox QAbstractItemView { background: #ffffff; color: #111827; selection-background-color: #2563eb; }"
            "QStatusBar { background: #e8edf3; color: #1f2937; border-top: 1px solid #cfd7e2; }"
            "QToolTip { background-color: #ffffff; color: #1f2937; border: 1px solid #c9d0da; }"
        )

    def _groupbox_stylesheet(self) -> str:
        font_pt = "10pt" if self._compact_ui else "11pt"
        margin_top = "8px" if self._compact_ui else "10px"
        title_left = "8px" if self._compact_ui else "10px"
        padding_top = "4px" if self._compact_ui else "6px"
        if self._theme_name == "Dark":
            border_color = "#4f5a6b"
            bg_color = "#242b36"
            text_color = "#e5e7eb"
        elif self._theme_name == "Teya":
            border_color = "#1db8b2"
            bg_color = "#ffe0ef"
            text_color = "#0b7f7a"
        elif self._theme_name == "Donald":
            border_color = "#f3a451"
            bg_color = "#c96a04"
            text_color = "#ffffff"
        else:
            border_color = "#c9ced6"
            bg_color = "#fcfcfc"
            text_color = "#1f2937"
        return (
            "QGroupBox {"
            "font-weight: 600;"
            f"font-size: {font_pt};"
            f"color: {text_color};"
            f"border: 1px solid {border_color};"
            "border-radius: 6px;"
            f"margin-top: {margin_top};"
            f"padding-top: {padding_top};"
            f"background: {bg_color};"
            "}"
            "QGroupBox::title {"
            "subcontrol-origin: margin;"
            f"left: {title_left};"
            "padding: 0 4px;"
            "}"
        )

    def _apply_groupbox_style(self) -> None:
        style = self._groupbox_stylesheet()
        for group in self._styled_groups:
            if group is not None:
                group.setStyleSheet(style)

    def _apply_theme(self, theme_name: str) -> None:
        theme = (theme_name or "Dark").strip().title()
        if theme not in {"Light", "Dark", "Teya", "Donald"}:
            theme = "Dark"
        self._theme_name = theme
        if hasattr(self, "theme_combo") and self.theme_combo.currentText() != theme:
            self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentText(theme)
            self.theme_combo.blockSignals(False)

        self.setStyleSheet(self._base_theme_stylesheet())
        self._apply_groupbox_style()
        if hasattr(self, "plot_title_label"):
            if theme == "Dark":
                plot_color = "#e5e7eb"
            elif theme == "Teya":
                plot_color = "#0b7f7a"
            elif theme == "Donald":
                plot_color = "#ffffff"
            else:
                plot_color = "#2f3640"
            self.plot_title_label.setStyleSheet(f"color: {plot_color};")
        self._apply_button_icons(
            compact_ui=self._compact_ui,
            color=self._theme_icon_color(),
        )
        if getattr(self, "_nbs_dialog", None) is not None and hasattr(self._nbs_dialog, "set_theme"):
            try:
                self._nbs_dialog.set_theme(theme)
            except Exception:
                pass
        if getattr(self, "_selector_dialog", None) is not None and hasattr(self._selector_dialog, "set_theme"):
            try:
                self._selector_dialog.set_theme(theme)
            except Exception:
                pass
        if getattr(self, "_gradients_dialog", None) is not None and hasattr(self._gradients_dialog, "set_theme"):
            try:
                self._gradients_dialog.set_theme(theme)
            except Exception:
                pass
        if getattr(self, "_harmonize_dialog", None) is not None and hasattr(self._harmonize_dialog, "set_theme"):
            try:
                self._harmonize_dialog.set_theme(theme)
            except Exception:
                pass
        if getattr(self, "_stack_prepare_dialog", None) is not None and hasattr(
            self._stack_prepare_dialog, "set_theme"
        ):
            try:
                self._stack_prepare_dialog.set_theme(theme)
            except Exception:
                pass
        if getattr(self, "_batch_import_dialog", None) is not None:
            try:
                self._batch_import_dialog.setStyleSheet(self.styleSheet())
            except Exception:
                pass
        if getattr(self, "_surface_dialog", None) is not None and hasattr(
            self._surface_dialog, "set_theme"
        ):
            try:
                self._surface_dialog.set_theme(theme)
            except Exception:
                pass
        if getattr(self, "_gradient_scatter_dialog", None) is not None and hasattr(
            self._gradient_scatter_dialog, "set_theme"
        ):
            try:
                self._gradient_scatter_dialog.set_theme(theme)
            except Exception:
                pass
        if getattr(self, "_gradient_classification_dialog", None) is not None and hasattr(
            self._gradient_classification_dialog, "set_theme"
        ):
            try:
                self._gradient_classification_dialog.set_theme(theme)
            except Exception:
                pass
        if getattr(self, "_label_info_dialog", None) is not None:
            try:
                self._label_info_dialog.setStyleSheet(self.styleSheet())
            except Exception:
                pass
        if getattr(self, "_participants_info_dialog", None) is not None:
            try:
                self._participants_info_dialog.setStyleSheet(self.styleSheet())
            except Exception:
                pass

    def _svg_icon(self, filename: str, color: str = None) -> QIcon:
        icon_path = Path(__file__).with_name("icons") / "svg" / filename
        if not icon_path.exists():
            return QIcon()

        svg_text = None
        if color:
            try:
                svg_text = icon_path.read_text(encoding="utf-8")
                svg_text = svg_text.replace("currentColor", color).replace("currentcolor", color)
            except Exception:
                svg_text = None

        if svg_text:
            try:
                pixmap = QPixmap()
                if pixmap.loadFromData(svg_text.encode("utf-8"), "SVG"):
                    return QIcon(pixmap)
            except Exception:
                pass
            if QSvgRenderer is not None and QPainter is not None:
                try:
                    renderer = QSvgRenderer(svg_text.encode("utf-8"))
                    if renderer.isValid():
                        size = renderer.defaultSize()
                        if not size.isValid():
                            size = QSize(24, 24)
                        pixmap = QPixmap(size)
                        transparent = Qt.GlobalColor.transparent if QT_LIB == 6 else Qt.transparent
                        pixmap.fill(transparent)
                        painter = QPainter(pixmap)
                        renderer.render(painter)
                        painter.end()
                        return QIcon(pixmap)
                except Exception:
                    pass

        icon = QIcon(str(icon_path))
        if not icon.isNull():
            return icon

        if QSvgRenderer is not None and QPainter is not None:
            try:
                renderer = QSvgRenderer(str(icon_path))
                if renderer.isValid():
                    size = renderer.defaultSize()
                    if not size.isValid():
                        size = QSize(24, 24)
                    pixmap = QPixmap(size)
                    transparent = Qt.GlobalColor.transparent if QT_LIB == 6 else Qt.transparent
                    pixmap.fill(transparent)
                    painter = QPainter(pixmap)
                    renderer.render(painter)
                    painter.end()
                    return QIcon(pixmap)
            except Exception:
                pass
        return QIcon()

    def _apply_button_icons(self, compact_ui: bool = False, color: str = None) -> None:
        icon_size = QSize(16, 16) if compact_ui else QSize(18, 18)
        mapping = [
            (getattr(self, "add_button", None), "folder_plus.svg"),
            (getattr(self, "add_batch_button", None), "folder_plus.svg"),
            (getattr(self, "remove_button", None), "trash.svg"),
            (getattr(self, "clear_button", None), "broom_clear.svg"),
            (getattr(self, "hist_button", None), "histogram.svg"),
            (getattr(self, "view_labels_button", None), "info.svg"),
            (getattr(self, "view_participants_button", None), "info.svg"),
            (getattr(self, "export_button", None), "export_grid.svg"),
            (getattr(self, "write_matrix_button", None), "save_disk.svg"),
            (getattr(self, "move_up_button", None), "arrow_up.svg"),
            (getattr(self, "move_down_button", None), "arrow_down.svg"),
            (getattr(self, "gradients_open_button", None), "cube_3d.svg"),
            (getattr(self, "nbs_prepare_button", None), "wrench_prepare.svg"),
            (getattr(self, "harmonize_prepare_button", None), "wrench_prepare.svg"),
            (getattr(self, "selector_prepare_button", None), "filter_threshold.svg"),
            (getattr(self, "sample_add_button", None), "folder_plus.svg"),
        ]
        for button, icon_name in mapping:
            if button is None:
                continue
            icon = self._svg_icon(icon_name, color=color)
            if icon.isNull():
                continue
            button.setIcon(icon)
            button.setIconSize(icon_size)

    def _file_entry_id(self, path: Path) -> str:
        return f"file::{path}"

    def _new_derived_id(self) -> str:
        self._derived_counter += 1
        return f"derived::{self._derived_counter}"

    def _entry_ids(self):
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            entry_id = item.data(USER_ROLE)
            if entry_id:
                yield entry_id

    def _current_entry_id(self):
        item = self.file_list.currentItem()
        if item is None:
            return None
        return item.data(USER_ROLE)

    def _current_entry(self):
        entry_id = self._current_entry_id()
        if entry_id is None:
            return None
        return self._entries.get(entry_id)

    def _current_source_path(self):
        entry = self._current_entry()
        if entry is None:
            return None
        source_path = entry.get("source_path", entry.get("path"))
        if not source_path:
            return None
        return Path(source_path)

    def ensure_initial_configuration(self) -> bool:
        if str(self._results_dir_default or "").strip():
            return True
        return self._open_preferences_dialog(initial_setup=True)

    def _default_open_dir(self) -> Path:
        bids_dir = str(self._bids_dir_default or "").strip()
        if bids_dir:
            return Path(bids_dir).expanduser()
        source_path = self._current_source_path()
        if source_path is not None:
            return source_path.parent
        return Path.cwd()

    def _default_results_dir(self) -> Path:
        results_dir = str(self._results_dir_default or "").strip()
        if results_dir:
            return Path(results_dir).expanduser()
        source_path = self._current_source_path()
        if source_path is not None:
            return source_path.parent
        return Path.cwd()

    def _default_dialog_dir(self) -> Path:
        return self._default_open_dir()

    def _default_parcellation_dir(self) -> Path:
        atlas_dir = str(self._atlas_dir_default or "").strip()
        if atlas_dir:
            return Path(atlas_dir).expanduser()
        if DEFAULT_PARCELLATION_DIR.exists():
            return DEFAULT_PARCELLATION_DIR
        return ROOTDIR

    def _update_view_labels_button(self) -> None:
        if not hasattr(self, "view_labels_button"):
            return
        source_path = self._current_source_path()
        self.view_labels_button.setEnabled(source_path is not None and source_path.exists())
        self._update_view_participants_button()

    def _update_view_participants_button(self) -> None:
        if not hasattr(self, "view_participants_button"):
            return
        source_path = self._current_source_path()
        enabled = False
        if source_path is not None and source_path.exists():
            info = self._covars_cache.get(source_path)
            if info is None:
                info = _load_covars_info(source_path)
                self._covars_cache[source_path] = info
            enabled = bool(_covars_columns(info))
        self.view_participants_button.setEnabled(enabled)

    def _open_label_info_dialog(self) -> None:
        source_path = self._current_source_path()
        if source_path is None or not source_path.exists():
            self.statusBar().showMessage("Select a matrix backed by an NPZ file to inspect labels.")
            return
        try:
            dialog = LabelInfoDialog(source_path, parent=self)
            dialog.setStyleSheet(self.styleSheet())
            dialog.finished.connect(lambda *_args: setattr(self, "_label_info_dialog", None))
            self._label_info_dialog = dialog
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            self.statusBar().showMessage(f"Opened label info for {source_path.name}.")
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to open label info: {exc}")

    def _open_participants_info_dialog(self) -> None:
        source_path = self._current_source_path()
        if source_path is None or not source_path.exists():
            self.statusBar().showMessage("Select a matrix backed by an NPZ file to inspect participants.")
            return
        covars_info = self._covars_cache.get(source_path)
        if covars_info is None:
            covars_info = _load_covars_info(source_path)
            self._covars_cache[source_path] = covars_info
        if covars_info is None:
            self.statusBar().showMessage("Covars not found in selected file.")
            return
        columns, rows = _covars_to_rows(covars_info)
        if not columns:
            self.statusBar().showMessage("Selected file has no tabular covars to display.")
            return
        try:
            dialog = ParticipantsInfoDialog(source_path, covars_info, parent=self)
            dialog.setStyleSheet(self.styleSheet())
            dialog.finished.connect(lambda *_args: setattr(self, "_participants_info_dialog", None))
            self._participants_info_dialog = dialog
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            self.statusBar().showMessage(
                f"Opened participants view for {source_path.name} ({len(rows)} rows)."
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to open participants view: {exc}")

    def _update_parcellation_label(self) -> None:
        if getattr(self, "_gradients_dialog", None) is not None:
            try:
                self._gradients_dialog.set_parcellation_path(self._active_parcellation_path)
            except Exception:
                pass

    def _select_parcellation_template(self) -> bool:
        template_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select 3D parcellation template",
            str(self._default_parcellation_dir()),
            "NIfTI files (*.nii *.nii.gz);;All files (*)",
        )
        if not template_path:
            return False
        return self._set_active_parcellation(Path(template_path))

    def _set_active_parcellation(self, template_path: Path) -> bool:
        try:
            import nibabel as nib
        except Exception as exc:
            self.statusBar().showMessage(f"nibabel not available: {exc}")
            return False
        try:
            template_img = nib.load(str(template_path))
            template_data = np.asarray(template_img.get_fdata(), dtype=int)
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to load template: {exc}")
            return False
        if template_data.ndim != 3:
            self.statusBar().showMessage("Template must be a 3D NIfTI image.")
            return False
        self._active_parcellation_path = Path(template_path)
        self._active_parcellation_img = template_img
        self._active_parcellation_data = template_data
        self._update_parcellation_label()
        self._reset_gradients_output()
        return True

    def _reset_gradients_output(self) -> None:
        self._last_gradients = None
        self._set_gradients_progress(0, 1, 0, "Idle")
        self._sync_gradients_dialog_state()

    def _set_gradients_progress(self, minimum: int, maximum: int, value: int, text: str) -> None:
        self._gradients_progress_state = {
            "minimum": int(minimum),
            "maximum": int(maximum),
            "value": int(value),
            "text": str(text or ""),
        }
        if getattr(self, "_gradients_dialog", None) is not None:
            try:
                self._gradients_dialog.set_progress(int(minimum), int(maximum), int(value), str(text or ""))
            except Exception:
                pass

    def _current_gradient_component_count(self) -> int:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_component_count = int(dialog.component_count())
            except Exception:
                pass
        try:
            value = int(self._gradient_component_count)
        except Exception:
            value = 4
        return max(1, min(10, value))

    @staticmethod
    def _normalize_gradient_hemisphere_mode(value: str) -> str:
        text = str(value or "both").strip().lower()
        if text not in {"both", "lh", "rh", "separate"}:
            text = "both"
        return text

    @staticmethod
    def _normalize_gradient_surface_mesh(value: str) -> str:
        text = str(value or "fsaverage4").strip().lower()
        valid = {"fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6", "fsaverage7"}
        if text not in valid:
            text = "fsaverage4"
        return text

    @staticmethod
    def _normalize_gradient_surface_render_count(value, max_components=None) -> int:
        try:
            count = int(value)
        except Exception:
            count = 1
        if count < 1:
            count = 1
        if max_components is not None:
            count = min(count, max(1, int(max_components)))
        return count

    @staticmethod
    def _normalize_gradient_surface_procrustes(enabled) -> bool:
        return bool(enabled)

    @staticmethod
    def _normalize_gradient_classification_component(value, max_components=None) -> str:
        text = str(value or "1").strip().lower()
        if text.startswith("c"):
            text = text[1:]
        try:
            index = int(text)
        except Exception:
            index = 1
        if index < 1:
            index = 1
        if max_components is not None:
            index = min(index, max(1, int(max_components)))
        return str(index)

    @staticmethod
    def _normalize_gradient_classification_axis(value: str, default: str = "gradient1") -> str:
        fallback = str(default or "gradient1").strip().lower()
        if fallback not in {"gradient1", "gradient2", "spatial"}:
            fallback = "gradient1"
        text = str(value or "").strip().lower()
        mapping = {
            "gradient1": "gradient1",
            "gradient 1": "gradient1",
            "g1": "gradient1",
            "c1": "gradient1",
            "1": "gradient1",
            "gradient2": "gradient2",
            "gradient 2": "gradient2",
            "g2": "gradient2",
            "c2": "gradient2",
            "2": "gradient2",
            "spatial": "spatial",
            "space": "spatial",
        }
        normalized = mapping.get(text, mapping.get(text.replace(" ", ""), fallback))
        if normalized not in {"gradient1", "gradient2", "spatial"}:
            normalized = fallback
        return normalized

    @staticmethod
    def _normalize_gradient_scatter_rotation(value: str) -> str:
        text = str(value or "Default").strip()
        valid = {"Default", "+90", "-90", "180"}
        if text not in valid:
            text = "Default"
        return text

    @staticmethod
    def _normalize_gradient_triangular_color_order(value: str) -> str:
        text = str(value or "RBG").strip().upper()
        valid = {"RGB", "RBG", "GRB", "GBR", "BRG", "BGR"}
        if text not in valid:
            text = "RBG"
        return text

    @staticmethod
    def _normalize_gradient_classification_fit_mode(value: str) -> str:
        text = str(value or "triangle").strip().lower()
        if text not in {"triangle", "square"}:
            text = "triangle"
        return text

    @staticmethod
    def _normalize_gradient_rotation_preset(value: str) -> str:
        text = str(value or "Default").strip()
        valid = {"Default", "X +90", "X -90", "Y +90", "Y -90", "Y 180", "Z +90", "Z -90"}
        if text not in valid:
            text = "Default"
        return text

    @staticmethod
    def _normalize_gradient_network_component(value, max_components=None) -> str:
        text = str(value or "all").strip().lower()
        if text in {"", "all"}:
            return "all"
        if text.startswith("c"):
            text = text[1:]
        try:
            index = int(text)
        except Exception:
            return "all"
        if index < 1:
            return "all"
        if max_components is not None and index > int(max_components):
            return "all"
        return str(index)

    def _ensure_gradient_rotation_count(self, count: int) -> None:
        target = max(1, min(10, int(count)))
        current = list(self._gradient_component_rotations or [])
        while len(current) < target:
            current.append("Default")
        self._gradient_component_rotations = current[:10]

    def _available_gradient_network_component_count(self) -> int:
        results = self._last_gradients or {}
        try:
            if results:
                value = int(results.get("n_grad", 0))
                if value > 0:
                    return max(1, min(10, value))
        except Exception:
            pass
        bundle = self._gradient_precomputed_bundle
        if isinstance(bundle, dict):
            try:
                value = int(bundle.get("component_count_total", 0))
            except Exception:
                value = 0
            if value > 0:
                return max(1, min(10, value))
        return self._current_gradient_component_count()

    def _gradient_surface_procrustes_available(self) -> bool:
        bundle = self._gradient_precomputed_bundle
        if not isinstance(bundle, dict):
            return False
        avg = bundle.get("gradients_avg")
        if avg is None:
            return False
        try:
            avg_array = np.asarray(avg, dtype=float)
        except Exception:
            return False
        return avg_array.ndim == 2 and avg_array.shape[0] > 0 and avg_array.shape[1] > 0

    def _selected_gradient_hemisphere_mode(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_hemisphere_mode = self._normalize_gradient_hemisphere_mode(
                    dialog.selected_hemisphere()
                )
            except Exception:
                pass
        return self._normalize_gradient_hemisphere_mode(self._gradient_hemisphere_mode)

    def _selected_gradient_surface_mesh(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_surface_mesh = self._normalize_gradient_surface_mesh(
                    dialog.selected_surface_mesh()
                )
            except Exception:
                pass
        self._gradient_surface_mesh = self._normalize_gradient_surface_mesh(self._gradient_surface_mesh)
        return self._gradient_surface_mesh

    def _selected_gradient_surface_render_count(self) -> int:
        dialog = getattr(self, "_gradients_dialog", None)
        available = self._available_gradient_network_component_count()
        if dialog is not None and hasattr(dialog, "selected_surface_render_component_count"):
            try:
                self._gradient_surface_render_count = self._normalize_gradient_surface_render_count(
                    dialog.selected_surface_render_component_count(),
                    max_components=available,
                )
            except Exception:
                pass
        self._gradient_surface_render_count = self._normalize_gradient_surface_render_count(
            self._gradient_surface_render_count,
            max_components=available,
        )
        return self._gradient_surface_render_count

    def _selected_gradient_surface_procrustes(self) -> bool:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None and hasattr(dialog, "use_surface_procrustes"):
            try:
                self._gradient_surface_procrustes = self._normalize_gradient_surface_procrustes(
                    dialog.use_surface_procrustes()
                )
            except Exception:
                pass
        if not self._gradient_surface_procrustes_available():
            self._gradient_surface_procrustes = False
        return bool(self._gradient_surface_procrustes)

    def _selected_gradient_classification_surface_mesh(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_classification_surface_mesh = self._normalize_gradient_surface_mesh(
                    dialog.selected_classification_surface_mesh()
                )
            except Exception:
                pass
        self._gradient_classification_surface_mesh = self._normalize_gradient_surface_mesh(
            self._gradient_classification_surface_mesh
        )
        return self._gradient_classification_surface_mesh

    def _selected_gradient_classification_hemisphere_mode(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_classification_hemisphere_mode = self._normalize_gradient_hemisphere_mode(
                    dialog.selected_classification_hemisphere()
                )
            except Exception:
                pass
        self._gradient_classification_hemisphere_mode = self._normalize_gradient_hemisphere_mode(
            self._gradient_classification_hemisphere_mode
        )
        return self._gradient_classification_hemisphere_mode

    def _selected_gradient_scatter_rotation(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_scatter_rotation = self._normalize_gradient_scatter_rotation(
                    dialog.selected_scatter_rotation()
                )
            except Exception:
                pass
        self._gradient_scatter_rotation = self._normalize_gradient_scatter_rotation(
            self._gradient_scatter_rotation
        )
        return self._gradient_scatter_rotation

    def _selected_gradient_triangular_rgb(self) -> bool:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_scatter_triangular_rgb = bool(dialog.use_triangular_rgb())
            except Exception:
                pass
        self._gradient_scatter_triangular_rgb = bool(self._gradient_scatter_triangular_rgb)
        return self._gradient_scatter_triangular_rgb

    def _selected_gradient_classification_fit_mode(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_classification_fit_mode = self._normalize_gradient_classification_fit_mode(
                    dialog.selected_classification_fit_mode()
                )
            except Exception:
                pass
        self._gradient_classification_fit_mode = self._normalize_gradient_classification_fit_mode(
            self._gradient_classification_fit_mode
        )
        return self._gradient_classification_fit_mode

    def _selected_gradient_triangular_color_order(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_triangular_color_order = self._normalize_gradient_triangular_color_order(
                    dialog.selected_triangular_color_order()
                )
            except Exception:
                pass
        self._gradient_triangular_color_order = self._normalize_gradient_triangular_color_order(
            self._gradient_triangular_color_order
        )
        return self._gradient_triangular_color_order

    def _selected_gradient_classification_colormap(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        names = self._available_colormap_names()
        if dialog is not None:
            try:
                value = str(dialog.selected_classification_colormap() or "").strip()
                if value:
                    self._gradient_classification_colormap_name = value
            except Exception:
                pass
        current = str(self._gradient_classification_colormap_name or "").strip()
        if current not in names:
            current = self._gradient_colormap_name if self._gradient_colormap_name in names else (names[0] if names else "spectrum_fsl")
            self._gradient_classification_colormap_name = current
        return current

    def _selected_gradient_classification_component(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        max_components = self._available_gradient_network_component_count()
        if dialog is not None:
            try:
                self._gradient_classification_component = self._normalize_gradient_classification_component(
                    dialog.selected_classification_component(),
                    max_components=max_components,
                )
            except Exception:
                pass
        self._gradient_classification_component = self._normalize_gradient_classification_component(
            self._gradient_classification_component,
            max_components=max_components,
        )
        return self._gradient_classification_component

    def _selected_gradient_classification_x_axis(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_classification_x_axis = self._normalize_gradient_classification_axis(
                    dialog.selected_classification_x_axis(),
                    default="gradient2",
                )
            except Exception:
                pass
        self._gradient_classification_x_axis = self._normalize_gradient_classification_axis(
            self._gradient_classification_x_axis,
            default="gradient2",
        )
        return self._gradient_classification_x_axis

    def _selected_gradient_classification_y_axis(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_classification_y_axis = self._normalize_gradient_classification_axis(
                    dialog.selected_classification_y_axis(),
                    default="gradient1",
                )
            except Exception:
                pass
        self._gradient_classification_y_axis = self._normalize_gradient_classification_axis(
            self._gradient_classification_y_axis,
            default="gradient1",
        )
        return self._gradient_classification_y_axis

    def _selected_gradient_classification_ignore_lh_parcel(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None and hasattr(dialog, "selected_classification_ignore_lh_parcel"):
            try:
                self._gradient_classification_ignore_lh_parcel = str(
                    dialog.selected_classification_ignore_lh_parcel() or ""
                ).strip()
            except Exception:
                pass
        self._gradient_classification_ignore_lh_parcel = str(
            self._gradient_classification_ignore_lh_parcel or ""
        ).strip()
        return self._gradient_classification_ignore_lh_parcel

    def _selected_gradient_classification_ignore_rh_parcel(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None and hasattr(dialog, "selected_classification_ignore_rh_parcel"):
            try:
                self._gradient_classification_ignore_rh_parcel = str(
                    dialog.selected_classification_ignore_rh_parcel() or ""
                ).strip()
            except Exception:
                pass
        self._gradient_classification_ignore_rh_parcel = str(
            self._gradient_classification_ignore_rh_parcel or ""
        ).strip()
        return self._gradient_classification_ignore_rh_parcel

    def _gradient_classification_ignore_parcel_options(self):
        results = self._last_gradients or {}
        projection_labels_raw = results.get("projection_labels", None)
        if projection_labels_raw is None:
            return {"lh": [], "rh": []}
        try:
            projection_labels = np.asarray(projection_labels_raw, dtype=int).reshape(-1)
        except Exception:
            return {"lh": [], "rh": []}
        parcel_names = _to_string_list(results.get("parcel_names"))
        if projection_labels.size == 0:
            return {"lh": [], "rh": []}
        if not parcel_names or len(parcel_names) != projection_labels.size:
            parcel_names = [f"Parcel {int(label)}" for label in projection_labels.tolist()]
        try:
            hemisphere_codes = np.asarray(self._gradient_projection_hemisphere_codes(), dtype=int).reshape(-1)
        except Exception:
            hemisphere_codes = np.full(projection_labels.shape, -1, dtype=int)
        if hemisphere_codes.shape != projection_labels.shape:
            hemisphere_codes = np.full(projection_labels.shape, -1, dtype=int)

        lh_names = []
        rh_names = []
        seen_lh = set()
        seen_rh = set()
        for name, code in zip(parcel_names, hemisphere_codes.tolist()):
            text = str(name or "").strip()
            if not text:
                continue
            if int(code) in {0, 2} and text not in seen_lh:
                seen_lh.add(text)
                lh_names.append(text)
            if int(code) in {1, 2} and text not in seen_rh:
                seen_rh.add(text)
                rh_names.append(text)
        return {"lh": lh_names, "rh": rh_names}

    def _is_gradient_classification_axis_available(self, axis_key: str, results=None) -> bool:
        current = self._last_gradients if results is None else results
        if not current:
            return False
        axis = self._normalize_gradient_classification_axis(axis_key)
        try:
            n_grad = int(current.get("n_grad", 0))
        except Exception:
            n_grad = 0
        if axis == "gradient1":
            return n_grad >= 1
        if axis == "gradient2":
            return n_grad >= 2
        if axis == "spatial":
            try:
                projection_labels = np.asarray(current.get("projection_labels"), dtype=int).reshape(-1)
            except Exception:
                projection_labels = np.zeros(0, dtype=int)
            return projection_labels.size > 0 and (
                bool(str(current.get("template_path") or "").strip()) or self._active_parcellation_img is not None
            )
        return False

    @staticmethod
    def _gradient_classification_axis_label(axis_key: str, axis_role: str = "x") -> str:
        axis = ConnectomeViewer._normalize_gradient_classification_axis(axis_key)
        if axis == "gradient1":
            return "Gradient 1"
        if axis == "gradient2":
            return "Gradient 2"
        return "Spatial 1" if str(axis_role).strip().lower() == "x" else "Spatial 2"

    @staticmethod
    def _infer_projection_hemisphere_from_name(name):
        text = str(name or "").strip().lower()
        if not text:
            return None
        if any(token in text for token in ("brainstem", "brain-stem", "midbrain")):
            return "midline"
        tokens = [token for token in re.split(r"[^a-z0-9]+", text) if token]
        if tokens:
            first = tokens[0]
            if first in {"lh", "left", "l"}:
                return "lh"
            if first in {"rh", "right", "r"}:
                return "rh"
            if "lh" in tokens or "left" in tokens:
                return "lh"
            if "rh" in tokens or "right" in tokens:
                return "rh"
        if text.startswith(("lh_", "lh-", "left_", "left-", "ctx-lh", "hemi-l")):
            return "lh"
        if text.startswith(("rh_", "rh-", "right_", "right-", "ctx-rh", "hemi-r")):
            return "rh"
        return None

    def _gradient_projection_hemisphere_codes(self):
        results = self._last_gradients or {}
        if not results:
            raise RuntimeError("No gradients are available.")
        projection_labels = np.asarray(results.get("projection_labels"), dtype=int).reshape(-1)
        if projection_labels.size == 0:
            raise RuntimeError("No projected parcel labels are available.")

        cached = results.get("hemisphere_codes")
        if isinstance(cached, dict):
            cached_labels = np.asarray(cached.get("projection_labels"), dtype=int).reshape(-1)
            cached_codes = np.asarray(cached.get("codes"), dtype=int).reshape(-1)
            if (
                cached_labels.shape == projection_labels.shape
                and np.array_equal(cached_labels, projection_labels)
                and cached_codes.shape == projection_labels.shape
            ):
                return cached_codes

        codes = np.full(projection_labels.shape, -1, dtype=int)
        parcel_names = _to_string_list(results.get("parcel_names"))
        if parcel_names and len(parcel_names) == projection_labels.size:
            for idx, parcel_name in enumerate(parcel_names):
                hemisphere = self._infer_projection_hemisphere_from_name(parcel_name)
                if hemisphere == "lh":
                    codes[idx] = 0
                elif hemisphere == "rh":
                    codes[idx] = 1
                elif hemisphere == "midline":
                    codes[idx] = 2

        unresolved = codes < 0
        if np.any(unresolved):
            template_img, template_data = self._gradient_template_img_and_data()
            centroid_codes = None
            try:
                centroids_world = np.asarray(
                    nettools.compute_centroids(
                        template_img,
                        labels=np.asarray(projection_labels, dtype=int),
                        world=True,
                    ),
                    dtype=float,
                )
                if centroids_world.shape[0] == projection_labels.size:
                    centroid_x = np.asarray(centroids_world[:, 0], dtype=float)
                    finite_x = centroid_x[np.isfinite(centroid_x)]
                    if finite_x.size and np.nanmin(finite_x) < 0.0 < np.nanmax(finite_x):
                        centroid_codes = np.where(centroid_x < 0.0, 0, 1).astype(int, copy=False)
            except Exception:
                centroid_codes = None

            if centroid_codes is None:
                centroids_vox = np.asarray(
                    nettools.compute_centroids(
                        template_img,
                        labels=np.asarray(projection_labels, dtype=int),
                        world=False,
                    ),
                    dtype=float,
                )
                if centroids_vox.shape[0] != projection_labels.size:
                    raise RuntimeError("Parcel centroid count does not match the projected labels.")
                midline_x = float(np.asarray(template_data, dtype=int).shape[0] * 0.5)
                centroid_codes = np.where(centroids_vox[:, 0] < midline_x, 0, 1).astype(int, copy=False)

            codes[unresolved] = centroid_codes[unresolved]

        results["hemisphere_codes"] = {
            "projection_labels": np.asarray(projection_labels, dtype=int),
            "codes": np.asarray(codes, dtype=int),
        }
        return np.asarray(codes, dtype=int)

    def _gradient_projection_hemisphere_mask(self, hemisphere_mode: str, projection_labels=None):
        mode = self._normalize_gradient_hemisphere_mode(hemisphere_mode)
        labels = np.asarray(
            self._last_gradients.get("projection_labels")
            if projection_labels is None
            else projection_labels,
            dtype=int,
        ).reshape(-1)
        if labels.size == 0 or mode in {"both", "separate"}:
            return np.ones(labels.shape, dtype=bool)
        codes = self._gradient_projection_hemisphere_codes()
        if codes.shape != labels.shape:
            raise RuntimeError("Hemisphere membership is out of sync with the projected labels.")
        if mode == "lh":
            return np.asarray((codes == 0) | (codes == 2), dtype=bool)
        return np.asarray((codes == 1) | (codes == 2), dtype=bool)

    def _can_classify_gradients(self) -> bool:
        results = self._last_gradients or {}
        if not results:
            return False
        try:
            n_grad = int(results.get("n_grad", 0))
        except Exception:
            n_grad = 0
        if n_grad < 1:
            return False
        x_axis = self._selected_gradient_classification_x_axis()
        y_axis = self._selected_gradient_classification_y_axis()
        return self._is_gradient_classification_axis_available(x_axis, results) and self._is_gradient_classification_axis_available(
            y_axis,
            results,
        )

    def _current_gradient_rotation_presets(self):
        dialog = getattr(self, "_gradients_dialog", None)
        count = self._current_gradient_component_count()
        if dialog is not None:
            try:
                presets = [
                    self._normalize_gradient_rotation_preset(value)
                    for value in dialog.rotation_presets()[:count]
                ]
                while len(presets) < count:
                    presets.append("Default")
                for idx, value in enumerate(presets):
                    if idx < len(self._gradient_component_rotations):
                        self._gradient_component_rotations[idx] = value
                    else:
                        self._gradient_component_rotations.append(value)
            except Exception:
                pass
        self._ensure_gradient_rotation_count(count)
        return [
            self._normalize_gradient_rotation_preset(value)
            for value in self._gradient_component_rotations[:count]
        ]

    def _selected_gradient_network_component(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        max_components = self._available_gradient_network_component_count()
        if dialog is not None:
            try:
                self._gradient_network_component = self._normalize_gradient_network_component(
                    dialog.selected_network_component(),
                    max_components=max_components,
                )
            except Exception:
                pass
        self._gradient_network_component = self._normalize_gradient_network_component(
            self._gradient_network_component,
            max_components=max_components,
        )
        return self._gradient_network_component

    def _has_square_current_matrix(self) -> bool:
        if self._current_matrix is None:
            return False
        matrix = np.asarray(self._current_matrix)
        return matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]

    def _update_gradients_button(self) -> None:
        enabled = bool(self._available_gradient_matrix_entries()) or isinstance(
            self._gradient_precomputed_bundle,
            dict,
        )
        if hasattr(self, "gradients_open_button"):
            self.gradients_open_button.setEnabled(enabled)
        if hasattr(self, "compute_gradients_action"):
            self.compute_gradients_action.setEnabled(enabled)
        self._sync_gradients_dialog_state()

    def _set_gradient_source_mode(self, use_precomputed: bool, *, reset_results: bool = False) -> None:
        next_mode = bool(use_precomputed) and isinstance(self._gradient_precomputed_bundle, dict)
        if next_mode == bool(self._gradient_use_precomputed_bundle):
            if reset_results:
                self._reset_gradients_output()
            return
        self._gradient_use_precomputed_bundle = next_mode
        if reset_results:
            self._reset_gradients_output()
        else:
            self._sync_gradients_dialog_state()

    def _on_gradient_component_changed(self, value: int) -> None:
        try:
            self._gradient_component_count = max(1, min(10, int(value)))
        except Exception:
            self._gradient_component_count = 4
        self._ensure_gradient_rotation_count(self._gradient_component_count)
        self._gradient_network_component = self._normalize_gradient_network_component(
            self._gradient_network_component,
            max_components=self._available_gradient_network_component_count(),
        )
        self._gradient_classification_component = self._normalize_gradient_classification_component(
            self._gradient_classification_component,
            max_components=self._available_gradient_network_component_count(),
        )
        self._sync_gradients_dialog_state()

    def _on_gradient_colormap_changed(self, value: str) -> None:
        text = str(value or "").strip()
        if text:
            self._gradient_colormap_name = text

    def _on_gradient_hemisphere_changed(self, value: str) -> None:
        self._gradient_hemisphere_mode = self._normalize_gradient_hemisphere_mode(value)

    def _on_gradient_surface_mesh_changed(self, value: str) -> None:
        self._gradient_surface_mesh = self._normalize_gradient_surface_mesh(value)

    def _on_gradient_surface_render_count_changed(self, value: int) -> None:
        self._gradient_surface_render_count = self._normalize_gradient_surface_render_count(
            value,
            max_components=self._available_gradient_network_component_count(),
        )

    def _on_gradient_surface_procrustes_changed(self, enabled: bool) -> None:
        self._gradient_surface_procrustes = self._normalize_gradient_surface_procrustes(enabled)

    def _on_gradient_classification_surface_mesh_changed(self, value: str) -> None:
        self._gradient_classification_surface_mesh = self._normalize_gradient_surface_mesh(value)

    def _on_gradient_classification_hemisphere_changed(self, value: str) -> None:
        self._gradient_classification_hemisphere_mode = self._normalize_gradient_hemisphere_mode(value)

    def _on_gradient_scatter_rotation_changed(self, value: str) -> None:
        self._gradient_scatter_rotation = self._normalize_gradient_scatter_rotation(value)

    def _on_gradient_triangular_rgb_changed(self, enabled: bool) -> None:
        self._gradient_scatter_triangular_rgb = bool(enabled)

    def _on_gradient_classification_fit_mode_changed(self, value: str) -> None:
        self._gradient_classification_fit_mode = self._normalize_gradient_classification_fit_mode(value)

    def _on_gradient_triangular_color_order_changed(self, value: str) -> None:
        self._gradient_triangular_color_order = self._normalize_gradient_triangular_color_order(value)

    def _on_gradient_classification_colormap_changed(self, value: str) -> None:
        text = str(value or "").strip()
        if text:
            self._gradient_classification_colormap_name = text

    def _on_gradient_classification_component_changed(self, value: str) -> None:
        self._gradient_classification_component = self._normalize_gradient_classification_component(
            value,
            max_components=self._available_gradient_network_component_count(),
        )

    def _on_gradient_classification_x_axis_changed(self, value: str) -> None:
        self._gradient_classification_x_axis = self._normalize_gradient_classification_axis(
            value,
            default="gradient2",
        )
        self._sync_gradients_dialog_state()

    def _on_gradient_classification_y_axis_changed(self, value: str) -> None:
        self._gradient_classification_y_axis = self._normalize_gradient_classification_axis(
            value,
            default="gradient1",
        )
        self._sync_gradients_dialog_state()

    def _on_gradient_classification_ignore_lh_changed(self, value: str) -> None:
        self._gradient_classification_ignore_lh_parcel = str(value or "").strip()

    def _on_gradient_classification_ignore_rh_changed(self, value: str) -> None:
        self._gradient_classification_ignore_rh_parcel = str(value or "").strip()

    @staticmethod
    def _load_gradient_classification_adjacency_npz(path: Path):
        with np.load(path, allow_pickle=True) as npz:
            if "adjacency_mat" not in npz:
                raise KeyError("Key 'adjacency_mat' was not found in the selected NPZ.")
            adjacency = np.asarray(npz["adjacency_mat"], dtype=float)
            parcel_labels = None
            for key in ("parcel_labels", "parcel_labels_group", "parcel_labels_group.npy"):
                if key in npz:
                    parcel_labels = npz[key]
                    break
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency_mat must be a square 2D matrix.")
        label_indices = _coerce_label_indices(parcel_labels, adjacency.shape[0])
        if label_indices is None:
            raise ValueError(
                "parcel_labels is missing, invalid, or does not match adjacency_mat size."
            )
        labels_array = np.asarray(label_indices, dtype=int)
        if np.unique(labels_array).size != labels_array.size:
            raise ValueError("parcel_labels in the adjacency NPZ must be unique.")
        return {
            "path": str(path),
            "adjacency": adjacency,
            "parcel_labels": labels_array,
        }

    def _gradient_classification_adjacency_data(self):
        path_text = str(self._gradient_classification_adjacency_path or "").strip()
        if not path_text:
            return None
        cached = self._gradient_classification_adjacency_cache
        if isinstance(cached, dict) and str(cached.get("path") or "") == path_text:
            return cached
        path = Path(path_text)
        if not path.exists():
            raise RuntimeError(f"Adjacency file not found: {path.name}")
        data = self._load_gradient_classification_adjacency_npz(path)
        self._gradient_classification_adjacency_cache = data
        return data

    def _set_gradient_classification_adjacency(self, path: Path) -> bool:
        try:
            data = self._load_gradient_classification_adjacency_npz(path)
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to load classification adjacency: {exc}")
            return False
        self._gradient_classification_adjacency_path = str(path)
        self._gradient_classification_adjacency_cache = data
        self._sync_gradients_dialog_state()
        self.statusBar().showMessage(f"Loaded classification adjacency from {path.name}.")
        return True

    def _select_gradient_classification_adjacency(self) -> None:
        start_dir = self._default_results_dir()
        existing = str(self._gradient_classification_adjacency_path or "").strip()
        if existing:
            existing_path = Path(existing)
            if existing_path.exists():
                start_dir = existing_path.parent
        else:
            source_path = self._current_source_path()
            if source_path is not None and source_path.exists():
                start_dir = source_path.parent
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select classification adjacency NPZ",
            str(start_dir),
            "NumPy archive (*.npz);;All files (*)",
        )
        if not selected:
            return
        self._set_gradient_classification_adjacency(Path(selected))

    def _clear_gradient_classification_adjacency(self, *, show_status: bool = True) -> None:
        self._gradient_classification_adjacency_path = ""
        self._gradient_classification_adjacency_cache = None
        self._sync_gradients_dialog_state()
        if show_status:
            self.statusBar().showMessage("Removed classification adjacency.")

    @staticmethod
    def _npz_optional_scalar_text(npz, *keys):
        for key in keys:
            if key not in npz:
                continue
            try:
                values = np.asarray(npz[key])
            except Exception:
                continue
            if values.ndim == 0:
                return _display_text(values.item()).strip()
            if values.size == 1:
                return _display_text(values.reshape(-1)[0]).strip()
        return ""

    @staticmethod
    def _npz_optional_display_vector(npz, *keys):
        for key in keys:
            if key not in npz:
                continue
            try:
                values = _flatten_display_vector(npz[key])
            except Exception:
                values = None
            if values is not None:
                return [str(value) for value in values]
        return None

    @staticmethod
    def _canonicalize_precomputed_gradients(raw_gradients, *, row_count=None, label_count=None):
        gradients = np.asarray(raw_gradients, dtype=float)
        if gradients.ndim != 3:
            raise ValueError("Precomputed gradients must be a 3D array.")

        sample_axis = 0
        if row_count is not None:
            try:
                target_rows = int(row_count)
            except Exception:
                target_rows = 0
            if target_rows > 0:
                candidates = [idx for idx, size in enumerate(gradients.shape) if int(size) == target_rows]
                if candidates:
                    sample_axis = 0 if 0 in candidates else candidates[0]
        gradients = np.moveaxis(gradients, sample_axis, 0)

        if label_count is not None:
            try:
                target_labels = int(label_count)
            except Exception:
                target_labels = 0
            if target_labels > 0:
                if gradients.shape[2] == target_labels:
                    pass
                elif gradients.shape[1] == target_labels:
                    gradients = np.swapaxes(gradients, 1, 2)
                else:
                    raise ValueError(
                        f"Could not align gradients shape {tuple(gradients.shape)} to {target_labels} parcel labels."
                    )
        elif gradients.shape[1] > gradients.shape[2]:
            gradients = np.swapaxes(gradients, 1, 2)

        return np.asarray(gradients, dtype=float)

    @staticmethod
    def _canonicalize_average_gradients(raw_gradients_avg, *, label_count=None):
        gradients_avg = np.asarray(raw_gradients_avg, dtype=float)
        if gradients_avg.ndim != 2:
            raise ValueError("gradients_avg must be a 2D array.")
        if label_count is not None:
            try:
                target_labels = int(label_count)
            except Exception:
                target_labels = 0
            if target_labels > 0:
                if gradients_avg.shape[1] == target_labels:
                    return np.asarray(gradients_avg, dtype=float)
                if gradients_avg.shape[0] == target_labels:
                    return np.asarray(gradients_avg.T, dtype=float)
                raise ValueError(
                    f"Could not align gradients_avg shape {tuple(gradients_avg.shape)} to {target_labels} parcel labels."
                )
        if gradients_avg.shape[1] >= gradients_avg.shape[0]:
            return np.asarray(gradients_avg, dtype=float)
        return np.asarray(gradients_avg.T, dtype=float)

    def _load_precomputed_gradient_bundle(self, path: Path):
        path = Path(path)
        try:
            with np.load(path, allow_pickle=True) as npz:
                if "gradients" not in npz:
                    return None

                raw_gradients = np.asarray(npz["gradients"], dtype=float)

                parcel_labels_raw = None
                parcel_names_raw = None
                for key in PARCEL_LABEL_KEYS:
                    if key in npz:
                        parcel_labels_raw = np.asarray(npz[key]).reshape(-1)
                        break
                for key in PARCEL_NAME_KEYS:
                    if key in npz:
                        parcel_names_raw = np.asarray(npz[key]).reshape(-1)
                        break

                label_count = None
                parcel_labels = None
                if parcel_labels_raw is not None:
                    coerced = _coerce_label_indices(parcel_labels_raw, parcel_labels_raw.size)
                    if coerced is None:
                        raise ValueError("parcel_labels_group is present but invalid.")
                    parcel_labels = np.asarray(coerced, dtype=int)
                    label_count = parcel_labels.size

                gradients_avg = None
                if "gradients_avg" in npz:
                    try:
                        gradients_avg = self._canonicalize_average_gradients(
                            npz["gradients_avg"],
                            label_count=label_count,
                        )
                    except Exception:
                        gradients_avg = None

                subject_values = self._npz_optional_display_vector(
                    npz,
                    "subject_id_list",
                    "participant_id_list",
                    "subject_ids",
                    "participant_ids",
                )
                session_values = self._npz_optional_display_vector(
                    npz,
                    "session_id_list",
                    "session_ids",
                )

                candidate_row_count = 0
                if subject_values:
                    candidate_row_count = len(subject_values)
                elif session_values:
                    candidate_row_count = len(session_values)

                covars_info = _load_covars_info(path)
                covars_columns, covars_rows = _covars_to_rows(covars_info)
                if covars_rows:
                    candidate_row_count = len(covars_rows)

                canonical_gradients = self._canonicalize_precomputed_gradients(
                    raw_gradients,
                    row_count=candidate_row_count or None,
                    label_count=label_count,
                )

                n_rows = int(canonical_gradients.shape[0])
                n_components = int(canonical_gradients.shape[1])
                n_labels = int(canonical_gradients.shape[2])

                if parcel_labels is None:
                    parcel_labels = np.arange(1, n_labels + 1, dtype=int)
                elif parcel_labels.size != n_labels:
                    raise ValueError(
                        f"parcel_labels_group has {parcel_labels.size} labels but gradients expect {n_labels} parcels."
                    )
                if np.unique(parcel_labels).size != parcel_labels.size:
                    raise ValueError("parcel_labels_group must be unique for precomputed gradients.")

                parcel_names = _to_string_list(parcel_names_raw) if parcel_names_raw is not None else None
                if parcel_names is None or len(parcel_names) != n_labels:
                    parcel_names = [f"Parcel {int(label)}" for label in parcel_labels.tolist()]

                row_dicts = []
                for row in list(covars_rows or []):
                    if isinstance(row, dict):
                        row_dicts.append(dict(row))
                    else:
                        row_dicts.append(
                            {
                                str(column): _display_text(value)
                                for column, value in zip(list(covars_columns or []), list(row))
                            }
                        )
                if row_dicts and len(row_dicts) != n_rows:
                    row_dicts = []
                    covars_columns = []
                if not row_dicts:
                    row_dicts = [{} for _ in range(n_rows)]
                    covars_columns = []

                def _merge_vector_column(column_name: str, values, *, overwrite: bool = False) -> None:
                    if values is None or len(values) != n_rows:
                        return
                    if column_name not in covars_columns:
                        covars_columns.append(column_name)
                    for row_idx, raw_value in enumerate(values):
                        text = _display_text(raw_value).strip()
                        if overwrite or not str(row_dicts[row_idx].get(column_name, "")).strip():
                            row_dicts[row_idx][column_name] = text

                _merge_vector_column("participant_id", subject_values)
                _merge_vector_column("session_id", session_values)
                _merge_vector_column(
                    "group",
                    self._npz_optional_display_vector(npz, "group"),
                )
                _merge_vector_column(
                    "modality",
                    self._npz_optional_display_vector(npz, "modality"),
                )
                _merge_vector_column(
                    "metabolites",
                    self._npz_optional_display_vector(npz, "metabolites"),
                )

                parcellation_path = None
                parcellation_text = self._npz_optional_scalar_text(npz, "parc_path", "parc_path.npy")
                if parcellation_text:
                    candidate = Path(parcellation_text).expanduser()
                    if not candidate.is_absolute():
                        candidate = (path.parent / candidate).resolve()
                    parcellation_path = candidate
        except Exception:
            raise

        summary = f"{path.name} | rows: {n_rows} | components: {n_components} | parcels: {n_labels}"
        return {
            "path": path,
            "label": path.name,
            "summary": summary,
            "gradients": canonical_gradients,
            "component_count_total": n_components,
            "n_rows": n_rows,
            "n_labels": n_labels,
            "gradients_avg": None if gradients_avg is None else np.asarray(gradients_avg, dtype=float),
            "parcel_labels": np.asarray(parcel_labels, dtype=int),
            "parcel_names": list(parcel_names),
            "covars_columns": [str(column) for column in covars_columns],
            "covars_rows": row_dicts,
            "parcellation_path": parcellation_path,
        }

    @staticmethod
    def _gradient_precomputed_row_pair(bundle, row_index: int):
        if not isinstance(bundle, dict):
            return "", ""
        rows = list(bundle.get("covars_rows") or [])
        if row_index < 0 or row_index >= len(rows):
            return "", ""
        row = rows[row_index]
        lower_map = {str(key).lower(): key for key in row.keys()}
        participant_key = (
            lower_map.get("participant_id")
            or lower_map.get("subject_id")
            or lower_map.get("participant")
            or lower_map.get("subject")
        )
        session_key = lower_map.get("session_id") or lower_map.get("session") or lower_map.get("ses")
        participant = _display_text(row.get(participant_key, "")).strip() if participant_key else ""
        session = _display_text(row.get(session_key, "")).strip() if session_key else ""
        return participant, session

    def _gradient_precomputed_selection_text(self) -> str:
        bundle = self._gradient_precomputed_bundle
        if not isinstance(bundle, dict):
            return "Selected pair: none"
        row_index = self._gradient_precomputed_selected_row
        if row_index is None:
            return "Selected pair: none"
        participant, session = self._gradient_precomputed_row_pair(bundle, int(row_index))
        parts = [f"row {int(row_index)}"]
        if participant:
            parts.append(participant)
        if session:
            parts.append(session)
        return "Selected pair: " + " | ".join(parts)

    def _activate_precomputed_gradient_bundle(self, bundle) -> None:
        if not isinstance(bundle, dict):
            return
        self._gradient_precomputed_bundle = bundle
        self._gradient_precomputed_selected_row = None
        self._gradient_use_precomputed_bundle = True
        try:
            self._gradient_component_count = max(
                1,
                min(10, int(bundle.get("component_count_total", 2))),
            )
        except Exception:
            self._gradient_component_count = 2
        self._gradient_surface_render_count = 1
        self._gradient_surface_procrustes = False

        parcellation_path = bundle.get("parcellation_path")
        did_reset = False
        if isinstance(parcellation_path, Path) and parcellation_path.exists():
            active_path = Path(self._active_parcellation_path) if self._active_parcellation_path is not None else None
            if active_path != parcellation_path:
                did_reset = bool(self._set_active_parcellation(parcellation_path))
        if not did_reset:
            self._reset_gradients_output()

        self._update_gradients_button()
        self._open_gradients_dialog(prefer_precomputed=True)
        if getattr(self, "_gradients_dialog", None) is not None and hasattr(self._gradients_dialog, "focus_precomputed_tab"):
            try:
                self._gradients_dialog.focus_precomputed_tab()
            except Exception:
                pass

        status = f"Loaded precomputed gradients from {bundle.get('label', 'bundle')}. Select a participant/session row."
        if isinstance(parcellation_path, Path) and not parcellation_path.exists():
            status += " Bundle parcellation path was not found; set it manually before confirming."
        self.statusBar().showMessage(status)

    def _confirm_precomputed_gradient_row(self, row_index: int) -> None:
        bundle = self._gradient_precomputed_bundle
        if not isinstance(bundle, dict):
            self.statusBar().showMessage("No precomputed gradients bundle is loaded.")
            return

        try:
            row_index = int(row_index)
        except Exception:
            self.statusBar().showMessage("Invalid precomputed gradient row.")
            return

        gradients_stack = np.asarray(bundle.get("gradients"), dtype=float)
        if gradients_stack.ndim != 3:
            self.statusBar().showMessage("Precomputed gradients array is invalid.")
            return
        if row_index < 0 or row_index >= gradients_stack.shape[0]:
            self.statusBar().showMessage("Selected participant/session row is out of range.")
            return

        parcel_labels = np.asarray(bundle.get("parcel_labels"), dtype=int).reshape(-1)
        parcel_names = list(bundle.get("parcel_names") or [])
        if len(parcel_names) != parcel_labels.size:
            parcel_names = [f"Parcel {int(label)}" for label in parcel_labels.tolist()]
        selected_components = np.asarray(gradients_stack[row_index], dtype=float)
        if selected_components.ndim != 2:
            self.statusBar().showMessage("Selected precomputed gradients row is not 2D.")
            return
        if selected_components.shape[1] == parcel_labels.size:
            pass
        elif selected_components.shape[0] == parcel_labels.size:
            selected_components = np.swapaxes(selected_components, 0, 1)
        else:
            self.statusBar().showMessage(
                "Selected precomputed gradients row does not align with the parcel label count."
            )
            return

        total_components = int(selected_components.shape[0])
        n_grad = max(1, min(10, total_components))
        components = np.asarray(selected_components[:n_grad, :], dtype=float).T
        if components.shape[0] != parcel_labels.size:
            self.statusBar().showMessage("Projected node count does not match the parcel label count.")
            return
        participant, session = self._gradient_precomputed_row_pair(bundle, row_index)
        display_bits = [bundle.get("label", "precomputed gradients")]
        if participant:
            display_bits.append(participant)
        if session:
            display_bits.append(session)
        source_name = " | ".join(display_bits)

        stem_bits = [bundle.get("path", Path("gradients")).stem]
        if participant:
            stem_bits.append(self._safe_name_fragment(_normalize_subject_token(participant)))
        if session:
            stem_bits.append(self._safe_name_fragment(_normalize_session_token(session)))
        default_name = "_".join(bit for bit in stem_bits if bit) + f"_diffusion_components-{n_grad}.nii.gz"

        template_path_text = ""
        bundle_parcellation = bundle.get("parcellation_path")
        if isinstance(bundle_parcellation, Path):
            template_path_text = str(bundle_parcellation)
        elif self._active_parcellation_path is not None:
            template_path_text = str(self._active_parcellation_path)

        self._gradient_component_count = n_grad
        self._gradient_surface_render_count = 1
        self._gradient_precomputed_selected_row = row_index
        self._last_gradients = {
            "gradients": np.asarray(components, dtype=float),
            "n_grad": n_grad,
            "n_nodes": parcel_labels.size,
            "projected_data": None,
            "affine": None,
            "header": None,
            "source_name": source_name,
            "source_dir": str(bundle["path"].parent),
            "output_name": default_name,
            "keep_indices": np.arange(parcel_labels.size, dtype=int),
            "projection_labels": np.asarray(parcel_labels, dtype=int),
            "support_mask": None,
            "template_path": template_path_text,
            "parcel_names": parcel_names,
            "matrix_entry_id": None,
            "matrix_label": bundle.get("label", ""),
            "precomputed_source_path": str(bundle["path"]),
            "precomputed_row_index": int(row_index),
        }
        self._set_gradients_progress(0, n_grad, n_grad, self._gradient_precomputed_selection_text())
        self._sync_gradients_dialog_state()
        self.statusBar().showMessage(
            f"Loaded precomputed gradients for row {row_index}. No diffusion embedding or fsaverage projection was run; projection will happen only when needed."
        )

    def _classification_scatter_edge_pairs(self, projection_labels, finite_mask):
        adjacency_data = self._gradient_classification_adjacency_data()
        if not adjacency_data:
            return np.zeros((0, 2), dtype=int), None

        labels = np.asarray(projection_labels, dtype=int).reshape(-1)
        mask = np.asarray(finite_mask, dtype=bool).reshape(-1)
        if labels.shape != mask.shape:
            raise RuntimeError("Classification labels do not align with the scatter mask.")
        labels = labels[mask]
        if labels.size < 2:
            return np.zeros((0, 2), dtype=int), None

        adjacency = np.asarray(adjacency_data["adjacency"], dtype=float)
        adjacency_labels = np.asarray(adjacency_data["parcel_labels"], dtype=int).reshape(-1)
        label_to_index = {int(label): idx for idx, label in enumerate(adjacency_labels.tolist())}

        mapped_positions = []
        mapped_indices = []
        missing_count = 0
        for scatter_index, label in enumerate(labels.tolist()):
            adjacency_index = label_to_index.get(int(label))
            if adjacency_index is None:
                missing_count += 1
                continue
            mapped_positions.append(int(scatter_index))
            mapped_indices.append(int(adjacency_index))

        if len(mapped_indices) < 2:
            note = (
                "Adjacency file does not overlap with the current classification labels."
                if missing_count
                else "Adjacency file does not contain enough nodes for edge rendering."
            )
            return np.zeros((0, 2), dtype=int), note

        mapped_positions = np.asarray(mapped_positions, dtype=int)
        mapped_indices = np.asarray(mapped_indices, dtype=int)
        edge_matrix = np.asarray(adjacency[np.ix_(mapped_indices, mapped_indices)], dtype=float)
        edge_matrix = np.nan_to_num(edge_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        edge_matrix = np.maximum(np.abs(edge_matrix), np.abs(edge_matrix.T))
        upper_i, upper_j = np.triu_indices(edge_matrix.shape[0], k=1)
        keep = edge_matrix[upper_i, upper_j] > 0.0
        if not np.any(keep):
            note = "Adjacency file loaded, but no non-zero edges matched the displayed nodes."
            return np.zeros((0, 2), dtype=int), note

        edge_pairs = np.column_stack(
            (
                mapped_positions[upper_i[keep]],
                mapped_positions[upper_j[keep]],
            )
        ).astype(int, copy=False)
        note = f"Adjacency edges: {edge_pairs.shape[0]}"
        if missing_count:
            note += f" ({missing_count} labels unmatched)"
        return edge_pairs, note

    @staticmethod
    def _gradient_entry_source_path(entry):
        if entry is None:
            return None
        source_path = entry.get("source_path", entry.get("path"))
        if not source_path:
            return None
        return Path(source_path)

    def _gradient_matrix_label_for_entry(self, entry) -> str:
        if entry is None:
            return ""
        label = str(entry.get("label") or "").strip()
        key = str(entry.get("selected_key") or "").strip()
        source_path = self._gradient_entry_source_path(entry)
        if not label and source_path is not None:
            label = source_path.name
        if key and key not in label:
            label = f"{label} [{key}]" if label else key
        return label or "matrix"

    def _available_gradient_matrix_entries(self):
        options = []
        for entry_id in self._entry_ids():
            entry = self._entries.get(entry_id)
            if entry is None:
                continue
            options.append(
                {
                    "id": entry_id,
                    "label": self._gradient_matrix_label_for_entry(entry),
                }
            )
        return options

    def _selected_gradient_entry_id(self):
        stored = self._gradient_selected_entry_id
        if stored in self._entries:
            return stored
        current = self._current_entry_id()
        if current in self._entries:
            return current
        for option in self._available_gradient_matrix_entries():
            entry_id = option.get("id")
            if entry_id in self._entries:
                return entry_id
        return None

    def _selected_gradient_entry(self):
        entry_id = self._selected_gradient_entry_id()
        if entry_id is None:
            return None
        return self._entries.get(entry_id)

    def _gradient_matrix_for_entry(self, entry):
        if entry is None:
            return None
        if entry is self._current_entry() and self._current_matrix is not None:
            return np.asarray(self._current_matrix)
        matrix, _selected_key = self._matrix_for_entry(entry)
        return np.asarray(matrix)

    def _has_square_matrix_entry(self, entry) -> bool:
        if entry is None:
            return False
        try:
            matrix = self._gradient_matrix_for_entry(entry)
        except Exception:
            return False
        return matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]

    def _selected_gradient_matrix_label(self) -> str:
        return self._gradient_matrix_label_for_entry(self._selected_gradient_entry())

    def _on_gradient_matrix_entry_changed(self, entry_id) -> None:
        normalized_id = entry_id if entry_id in self._entries else None
        if normalized_id == self._gradient_selected_entry_id:
            return
        self._gradient_selected_entry_id = normalized_id
        self._reset_gradients_output()

    def _on_gradient_network_component_changed(self, value: str) -> None:
        self._gradient_network_component = self._normalize_gradient_network_component(
            value,
            max_components=self._available_gradient_network_component_count(),
        )

    def _on_gradient_rotation_changed(self, index: int, value: str) -> None:
        try:
            idx = int(index)
        except Exception:
            return
        if idx < 0 or idx >= 10:
            return
        self._ensure_gradient_rotation_count(idx + 1)
        self._gradient_component_rotations[idx] = self._normalize_gradient_rotation_preset(value)

    def _sync_gradients_dialog_state(self) -> None:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is None:
            return
        matrix_options = self._available_gradient_matrix_entries()
        selected_entry_id = self._selected_gradient_entry_id()
        if self._gradient_selected_entry_id not in self._entries:
            self._gradient_selected_entry_id = selected_entry_id
        selected_entry = self._selected_gradient_entry()
        names = self._available_colormap_names()
        current_cmap = str(self._gradient_colormap_name or "").strip()
        if current_cmap not in names:
            current_cmap = (
                self._default_gradient_colormap
                if self._default_gradient_colormap in names
                else (names[0] if names else "spectrum_fsl")
            )
            self._gradient_colormap_name = current_cmap
        precomputed_bundle = (
            self._gradient_precomputed_bundle
            if bool(self._gradient_use_precomputed_bundle) and isinstance(self._gradient_precomputed_bundle, dict)
            else None
        )
        if precomputed_bundle is not None:
            component_count = max(
                1,
                min(
                    10,
                    int(precomputed_bundle.get("component_count_total", self._current_gradient_component_count())),
                ),
            )
            self._gradient_component_count = component_count
            matrix_source = f"Precomputed gradients: {precomputed_bundle.get('label', 'bundle')}"
        else:
            component_count = self._current_gradient_component_count()
            matrix_source = self._gradient_matrix_label_for_entry(selected_entry)

        dialog.set_matrix_options(matrix_options, selected_entry_id=selected_entry_id)
        dialog.set_matrix_source(matrix_source)
        dialog.set_component_count(component_count)
        dialog.set_surface_render_component_limit(self._available_gradient_network_component_count())
        dialog.set_surface_render_component_count(self._selected_gradient_surface_render_count())
        dialog.set_surface_procrustes_available(self._gradient_surface_procrustes_available())
        dialog.set_surface_procrustes_enabled(self._selected_gradient_surface_procrustes())
        dialog.set_colormap_names(names, current_colormap=current_cmap)
        dialog.set_parcellation_path(self._active_parcellation_path)
        dialog.set_hemisphere_mode(self._selected_gradient_hemisphere_mode())
        dialog.set_surface_mesh(self._selected_gradient_surface_mesh())
        dialog.set_classification_surface_mesh(self._selected_gradient_classification_surface_mesh())
        dialog.set_classification_hemisphere_mode(self._selected_gradient_classification_hemisphere_mode())
        dialog.set_scatter_rotation(self._selected_gradient_scatter_rotation())
        dialog.set_triangular_rgb(self._selected_gradient_triangular_rgb())
        dialog.set_classification_fit_mode(self._selected_gradient_classification_fit_mode())
        dialog.set_triangular_color_order(self._selected_gradient_triangular_color_order())
        dialog.set_classification_colormap(self._selected_gradient_classification_colormap())
        dialog.set_classification_component_options(
            self._available_gradient_network_component_count(),
            selected_component=self._selected_gradient_classification_component(),
        )
        dialog.set_classification_axes(
            self._selected_gradient_classification_x_axis(),
            self._selected_gradient_classification_y_axis(),
        )
        ignore_options = self._gradient_classification_ignore_parcel_options()
        dialog.set_classification_ignore_parcel_options(
            ignore_options.get("lh", []),
            ignore_options.get("rh", []),
            selected_lh=self._selected_gradient_classification_ignore_lh_parcel(),
            selected_rh=self._selected_gradient_classification_ignore_rh_parcel(),
        )
        dialog.set_classification_adjacency_path(self._gradient_classification_adjacency_path)
        dialog.set_network_component_options(
            self._available_gradient_network_component_count(),
            selected_component=self._selected_gradient_network_component(),
        )
        dialog.set_rotation_presets(self._current_gradient_rotation_presets())
        progress = self._gradients_progress_state
        dialog.set_progress(
            progress["minimum"],
            progress["maximum"],
            progress["value"],
            progress["text"],
        )
        dialog.set_precomputed_mode(precomputed_bundle is not None)
        if precomputed_bundle is not None:
            row_payload = []
            for row_index, row_data in enumerate(list(precomputed_bundle.get("covars_rows") or [])):
                payload_row = {"__row_index__": row_index}
                payload_row.update({str(key): _display_text(value) for key, value in dict(row_data).items()})
                row_payload.append(payload_row)
            dialog.set_precomputed_rows(
                precomputed_bundle.get("covars_columns") or [],
                row_payload,
                selected_row=self._gradient_precomputed_selected_row,
                summary_text=precomputed_bundle.get("summary", ""),
                selection_text=self._gradient_precomputed_selection_text(),
            )
        else:
            dialog.set_precomputed_rows([], [], selected_row=None, summary_text="", selection_text="")
        dialog.set_can_compute(precomputed_bundle is None and self._has_square_matrix_entry(selected_entry))
        dialog.set_busy(self._gradients_busy)
        dialog.set_has_results(bool(self._last_gradients))
        dialog.set_can_classify(self._can_classify_gradients())

    def _open_gradients_dialog(self, *_args, prefer_precomputed=None) -> None:
        has_matrix_entries = bool(self._available_gradient_matrix_entries())
        has_precomputed_bundle = isinstance(self._gradient_precomputed_bundle, dict)
        if bool(prefer_precomputed) and has_precomputed_bundle:
            self._set_gradient_source_mode(True, reset_results=False)
        elif has_matrix_entries:
            self._set_gradient_source_mode(False, reset_results=False)
        elif has_precomputed_bundle:
            self._set_gradient_source_mode(True, reset_results=False)
        else:
            self._set_gradient_source_mode(False, reset_results=False)
        if getattr(self, "_gradients_dialog", None) is None:
            try:
                from window.gradients_prepare import GradientsPrepareDialog
            except Exception:
                try:
                    from mrsi_viewer.window.gradients_prepare import GradientsPrepareDialog
                except Exception as exc:
                    self.statusBar().showMessage(f"Failed to open Gradients window: {exc}")
                    return

            self._gradients_dialog = GradientsPrepareDialog(
                theme_name=self._theme_name,
                component_count=self._gradient_component_count,
                colormap_names=self._available_colormap_names(),
                current_colormap=self._gradient_colormap_name,
                parcellation_path=self._active_parcellation_path,
                open_parcellation_callback=self._select_parcellation_template,
                compute_callback=self._compute_gradients,
                save_callback=self._save_gradients_projection,
                render_3d_callback=self._render_gradients_3d,
                classify_callback=self._classify_gradients_fsaverage,
                render_network_callback=self._render_gradients_network,
                matrix_changed_callback=self._on_gradient_matrix_entry_changed,
                component_changed_callback=self._on_gradient_component_changed,
                colormap_changed_callback=self._on_gradient_colormap_changed,
                hemisphere_changed_callback=self._on_gradient_hemisphere_changed,
                surface_mesh_changed_callback=self._on_gradient_surface_mesh_changed,
                surface_render_count_changed_callback=self._on_gradient_surface_render_count_changed,
                surface_procrustes_changed_callback=self._on_gradient_surface_procrustes_changed,
                classification_surface_mesh_changed_callback=self._on_gradient_classification_surface_mesh_changed,
                classification_hemisphere_changed_callback=self._on_gradient_classification_hemisphere_changed,
                scatter_rotation_changed_callback=self._on_gradient_scatter_rotation_changed,
                triangular_rgb_changed_callback=self._on_gradient_triangular_rgb_changed,
                classification_fit_mode_changed_callback=self._on_gradient_classification_fit_mode_changed,
                triangular_color_order_changed_callback=self._on_gradient_triangular_color_order_changed,
                classification_colormap_changed_callback=self._on_gradient_classification_colormap_changed,
                classification_component_changed_callback=self._on_gradient_classification_component_changed,
                classification_x_axis_changed_callback=self._on_gradient_classification_x_axis_changed,
                classification_y_axis_changed_callback=self._on_gradient_classification_y_axis_changed,
                classification_ignore_lh_changed_callback=self._on_gradient_classification_ignore_lh_changed,
                classification_ignore_rh_changed_callback=self._on_gradient_classification_ignore_rh_changed,
                open_classification_adjacency_callback=self._select_gradient_classification_adjacency,
                remove_classification_adjacency_callback=self._clear_gradient_classification_adjacency,
                precomputed_row_confirm_callback=self._confirm_precomputed_gradient_row,
                network_component_changed_callback=self._on_gradient_network_component_changed,
                rotation_changed_callback=self._on_gradient_rotation_changed,
                parent=self,
            )

        self._sync_gradients_dialog_state()
        self._gradients_dialog.show()
        try:
            self._gradients_dialog.raise_()
            self._gradients_dialog.activateWindow()
        except Exception:
            pass
        self.statusBar().showMessage("Opened Gradients window.")

    def _current_nbs_source(self):
        entry = self._current_entry()
        if entry is None or entry.get("kind") != "file":
            return None
        source_path = entry.get("path")
        if source_path is None:
            return None
        source_path = Path(source_path)
        if not source_path.exists():
            return None
        key = self._ensure_entry_key(entry)
        if not key:
            return None

        stack_axis = entry.get("stack_axis")
        stack_len = entry.get("stack_len")
        if stack_axis is not None and stack_len is not None and int(stack_len) > 1:
            return {"path": source_path, "key": key, "stack_len": int(stack_len)}

        try:
            raw = _load_matrix_from_npz(source_path, key, average=False)
        except Exception:
            return None
        if raw.ndim != 3:
            return None
        axis = _stack_axis(raw.shape)
        if axis is None or int(raw.shape[axis]) <= 1:
            return None
        return {"path": source_path, "key": key, "stack_len": int(raw.shape[axis])}

    def _update_nbs_prepare_button(self) -> None:
        enabled = self._current_nbs_source() is not None
        if not hasattr(self, "nbs_prepare_button"):
            if hasattr(self, "nbs_prepare_action"):
                self.nbs_prepare_action.setEnabled(enabled)
            self._update_gradients_button()
            self._update_selector_prepare_button()
            self._update_harmonize_prepare_button()
            return
        self.nbs_prepare_button.setEnabled(enabled)
        if hasattr(self, "nbs_prepare_action"):
            self.nbs_prepare_action.setEnabled(enabled)
        self._update_gradients_button()
        self._update_selector_prepare_button()
        self._update_harmonize_prepare_button()

    def _current_selector_source(self):
        return self._current_nbs_source()

    def _current_harmonize_source(self):
        return self._current_nbs_source()

    def _update_selector_prepare_button(self) -> None:
        if not hasattr(self, "selector_prepare_button"):
            return
        source = self._current_selector_source()
        enabled = False
        if source is not None:
            source_path = source["path"]
            info = self._covars_cache.get(source_path)
            if info is None:
                info = _load_covars_info(source_path)
                self._covars_cache[source_path] = info
            enabled = bool(_covars_columns(info))
        self.selector_prepare_button.setEnabled(enabled)
        self._update_harmonize_prepare_button()
        self._update_write_to_file_button()

    def _update_harmonize_prepare_button(self) -> None:
        if not hasattr(self, "harmonize_prepare_button"):
            if hasattr(self, "harmonize_prepare_action"):
                self.harmonize_prepare_action.setEnabled(False)
            return
        source = self._current_harmonize_source()
        enabled = False
        if source is not None:
            source_path = source["path"]
            info = self._covars_cache.get(source_path)
            if info is None:
                info = _load_covars_info(source_path)
                self._covars_cache[source_path] = info
            enabled = bool(_covars_columns(info))
        self.harmonize_prepare_button.setEnabled(enabled)
        if hasattr(self, "harmonize_prepare_action"):
            self.harmonize_prepare_action.setEnabled(enabled)

    def _update_write_to_file_button(self) -> None:
        if not hasattr(self, "write_matrix_button"):
            return
        entry = self._current_entry()
        self.write_matrix_button.setEnabled(entry is not None)

    def _open_nbs_prepare_dialog(self) -> None:
        source = self._current_nbs_source()
        if source is None:
            self.statusBar().showMessage(
                "NBS Prepare requires a file-based matrix stack (multiple matrices)."
            )
            return

        source_path = source["path"]
        covars_info = self._covars_cache.get(source_path)
        if covars_info is None:
            covars_info = _load_covars_info(source_path)
            self._covars_cache[source_path] = covars_info
        if covars_info is None:
            self.statusBar().showMessage("Covars not found in selected file.")
            return

        covars_len = 0
        df = covars_info.get("df")
        if df is not None:
            covars_len = len(df)
        else:
            data = covars_info.get("data")
            if data is not None:
                covars_len = len(data)
        if covars_len and covars_len != source["stack_len"]:
            self.statusBar().showMessage(
                "Covars length does not match matrix stack size."
            )
            return

        try:
            from window.nbs_prepare import NBSPrepareDialog
        except Exception:
            try:
                from mrsi_viewer.window.nbs_prepare import NBSPrepareDialog
            except Exception as exc:
                self.statusBar().showMessage(f"Failed to open NBS window: {exc}")
                return

        self._nbs_dialog = NBSPrepareDialog(
            covars_info=covars_info,
            source_path=source_path,
            matrix_key=source["key"],
            matlab_cmd_default=self._matlab_cmd_default,
            matlab_nbs_path_default=self._matlab_nbs_path_default,
            output_dir_default=self._results_dir_default,
            atlas_dir_default=self._atlas_dir_default,
            bids_dir_default=self._bids_dir_default,
            theme_name=self._theme_name,
            parent=self,
        )
        self._nbs_dialog.show()
        self.statusBar().showMessage(
            f"Opened NBS Prepare ({source_path.name}, key={source['key']})."
        )

    def _open_selector_prepare_dialog(self) -> None:
        source = self._current_selector_source()
        if source is None:
            self.statusBar().showMessage(
                "Selector Prepare requires a file-based matrix stack (multiple matrices)."
            )
            return

        source_path = source["path"]
        covars_info = self._covars_cache.get(source_path)
        if covars_info is None:
            covars_info = _load_covars_info(source_path)
            self._covars_cache[source_path] = covars_info
        if covars_info is None:
            self.statusBar().showMessage("Covars not found in selected file.")
            return

        covars_len = 0
        df = covars_info.get("df")
        if df is not None:
            covars_len = len(df)
        else:
            data = covars_info.get("data")
            if data is not None:
                covars_len = len(data)
        if covars_len and covars_len != source["stack_len"]:
            self.statusBar().showMessage("Covars length does not match matrix stack size.")
            return

        try:
            from window.selector_prepare import SelectorPrepareDialog
        except Exception:
            try:
                from mrsi_viewer.window.selector_prepare import SelectorPrepareDialog
            except Exception as exc:
                self.statusBar().showMessage(f"Failed to open Selector window: {exc}")
                return

        self._selector_dialog = SelectorPrepareDialog(
            covars_info=covars_info,
            source_path=source_path,
            matrix_key=source["key"],
            theme_name=self._theme_name,
            export_callback=self._import_selector_aggregate,
            parent=self,
        )
        self._selector_dialog.show()
        self.statusBar().showMessage(
            f"Opened Selector Prepare ({source_path.name}, key={source['key']})."
        )

    def _open_harmonize_prepare_dialog(self) -> None:
        source = self._current_harmonize_source()
        if source is None:
            self.statusBar().showMessage(
                "Harmonize Prepare requires a file-based matrix stack (multiple matrices)."
            )
            return

        source_path = source["path"]
        covars_info = self._covars_cache.get(source_path)
        if covars_info is None:
            covars_info = _load_covars_info(source_path)
            self._covars_cache[source_path] = covars_info
        if covars_info is None:
            self.statusBar().showMessage("Covars not found in selected file.")
            return

        covars_len = 0
        df = covars_info.get("df")
        if df is not None:
            covars_len = len(df)
        else:
            data = covars_info.get("data")
            if data is not None:
                covars_len = len(data)
        if covars_len and covars_len != source["stack_len"]:
            self.statusBar().showMessage("Covars length does not match matrix stack size.")
            return

        try:
            from window.harmonize_prepare import HarmonizePrepareDialog
        except Exception:
            try:
                from mrsi_viewer.window.harmonize_prepare import HarmonizePrepareDialog
            except Exception as exc:
                self.statusBar().showMessage(f"Failed to open Harmonize window: {exc}")
                return

        self._harmonize_dialog = HarmonizePrepareDialog(
            covars_info=covars_info,
            source_path=source_path,
            matrix_key=source["key"],
            output_dir_default=self._results_dir_default,
            theme_name=self._theme_name,
            export_callback=self._import_harmonized_result,
            parent=self,
        )
        self._harmonize_dialog.show()
        self.statusBar().showMessage(
            f"Opened Harmonize Prepare ({source_path.name}, key={source['key']})."
        )

    def _import_harmonized_result(self, payload) -> bool:
        output_raw = str(payload.get("output_path") or "").strip()
        if not output_raw:
            self.statusBar().showMessage("Harmonize export payload missing output path.")
            return False
        output_path = Path(output_raw)
        if not output_path.is_file():
            self.statusBar().showMessage(f"Harmonized file not found: {output_path}")
            return False

        self._invalidate_path_caches(output_path)
        self._add_files([str(output_path)])
        target_id = self._file_entry_id(output_path)
        selected = False
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item is None:
                continue
            if item.data(USER_ROLE) == target_id:
                self.file_list.setCurrentItem(item)
                selected = True
                break
        if not selected:
            self.statusBar().showMessage(
                f"Harmonized file saved to {output_path.name} (already in workspace)."
            )
        else:
            self.statusBar().showMessage(
                f"Imported harmonized matrix stack: {output_path.name}."
            )
        return True

    def _import_stacked_result(self, payload) -> bool:
        output_raw = str(payload.get("output_path") or "").strip()
        if not output_raw:
            self.statusBar().showMessage("Stack export payload missing output path.")
            return False
        output_path = Path(output_raw)
        if not output_path.is_file():
            self.statusBar().showMessage(f"Stacked file not found: {output_path}")
            return False

        self._invalidate_path_caches(output_path)
        self._add_files([str(output_path)])
        target_id = self._file_entry_id(output_path)
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item is None:
                continue
            if item.data(USER_ROLE) == target_id:
                self.file_list.setCurrentItem(item)
                self.statusBar().showMessage(f"Imported stacked matrix file: {output_path.name}.")
                return True
        self.statusBar().showMessage(f"Stacked matrix saved to {output_path.name}.")
        return True

    def _open_stack_prepare_dialog(self, selected_paths) -> None:
        paths = [str(path) for path in (selected_paths or []) if str(path).strip()]
        if not paths:
            self.statusBar().showMessage("Stack requires at least one selected matrix file.")
            return

        try:
            from window.stack_prepare import StackPrepareDialog
        except Exception:
            try:
                from mrsi_viewer.window.stack_prepare import StackPrepareDialog
            except Exception as exc:
                self.statusBar().showMessage(f"Failed to open Stack window: {exc}")
                return

        self._stack_prepare_dialog = StackPrepareDialog(
            selected_paths=paths,
            theme_name=self._theme_name,
            export_callback=self._import_stacked_result,
            default_results_dir=self._results_dir_default,
            default_bids_dir=self._bids_dir_default,
            default_atlas_dir=self._atlas_dir_default,
            parent=self,
        )
        self._stack_prepare_dialog.show()
        self.statusBar().showMessage(f"Opened Stack Prepare ({len(paths)} selected files).")

    def _import_selector_aggregate(self, payload) -> bool:
        try:
            matrix = np.asarray(payload.get("matrix"), dtype=float)
        except Exception as exc:
            self.statusBar().showMessage(f"Invalid aggregated matrix payload: {exc}")
            return False
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            self.statusBar().showMessage("Aggregated matrix must be square.")
            return False

        source_path_raw = str(payload.get("source_path", "")).strip()
        source_path = Path(source_path_raw) if source_path_raw else None
        source_name = source_path.name if source_path is not None else "matrix"
        matrix_key = str(payload.get("matrix_key") or "matrix")
        method = str(payload.get("method") or "mean").strip().lower()
        method_label = "zfisher" if method == "zfisher" else "avg"
        selected_rows = list(payload.get("selected_rows") or [])
        n_selected = len(selected_rows)
        n_total = int(payload.get("n_total_rows") or n_selected)
        filter_covar = str(payload.get("filter_covar") or "").strip()
        filter_values = [str(v) for v in (payload.get("filter_values") or []) if str(v) != ""]

        if filter_covar and filter_values:
            filter_text = f"{filter_covar}={','.join(filter_values)}"
        else:
            filter_text = f"n={n_selected}/{n_total}"
        label = f"{method_label} {matrix_key} [{filter_text}] ({source_name})"

        derived_id = self._new_derived_id()
        self._entries[derived_id] = {
            "id": derived_id,
            "kind": "derived",
            "matrix": matrix,
            "source_path": source_path,
            "selected_key": matrix_key,
            "sample_index": None,
            "auto_title": True,
            "label": label,
            "aggregation_method": method,
            "selected_rows": selected_rows,
            "covar_name": filter_covar or None,
            "covar_value": ",".join(filter_values) if filter_values else None,
        }
        self.titles[derived_id] = label
        item = QListWidgetItem(label)
        item.setData(USER_ROLE, derived_id)
        self.file_list.addItem(item)
        self.file_list.setCurrentItem(item)
        self.statusBar().showMessage(f"Imported aggregated matrix ({method_label}).")
        return True

    def _open_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select .npz files",
            str(self._default_open_dir()),
            "NumPy archives (*.npz);;All files (*)",
        )
        self._add_files(paths)

    def _batch_connectivity_paths(self, folder_path: Path):
        folder_path = Path(folder_path)
        candidates = []
        try:
            for root, dirs, files in os.walk(folder_path, followlinks=False):
                dirs.sort(key=str.lower)
                files.sort(key=str.lower)
                root_path = Path(root)
                for filename in files:
                    name = filename.lower()
                    if not name.endswith(".npz"):
                        continue
                    if "connectivity" not in name:
                        continue
                    candidates.append(root_path / filename)
        except Exception:
            return candidates
        return sorted(candidates, key=lambda path: str(path.relative_to(folder_path)).lower())

    def _open_batch_import_dialog(self, folder_path: Path):
        folder_path = Path(folder_path)
        candidate_paths = self._batch_connectivity_paths(folder_path)
        if not candidate_paths:
            self.statusBar().showMessage(
                f"No .npz files containing 'connectivity' found in {folder_path.name} or its subfolders."
            )
            QMessageBox.information(
                self,
                "No Connectivity Matrices",
                (
                    f"No files ending in .npz and containing 'connectivity' were found in:\n"
                    f"{folder_path}\n\n"
                    f"Subfolders were scanned recursively."
                ),
            )
            return []

        dialog = BatchMatrixImportDialog(
            folder_path,
            candidate_paths,
            export_callback=self._import_stacked_result,
            parent=self,
        )
        self._batch_import_dialog = dialog
        if hasattr(dialog, "set_theme"):
            dialog.set_theme(self._theme_name)
        try:
            if dialog.exec() != _dialog_accepted_code():
                return []

            selected_paths = dialog.selected_paths()
            added_paths = self._add_files(selected_paths)
            if not added_paths:
                self.statusBar().showMessage("No new batch files were added.")
                return []
            self.statusBar().showMessage(
                f"Added {len(added_paths)} matrix files from {folder_path.name}."
            )
            return added_paths
        finally:
            if self._batch_import_dialog is dialog:
                self._batch_import_dialog = None

    def _open_batch_folder(self) -> None:
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select folder with connectivity matrices",
            str(self._default_open_dir()),
        )
        if not selected_dir:
            return

        self._open_batch_import_dialog(Path(selected_dir))

    def _add_files(self, paths):
        added_paths = []
        added_any = False
        loaded_precomputed_bundle = None
        for raw_path in paths:
            if not raw_path:
                continue
            path = Path(raw_path)
            if path.suffix.lower() != ".npz" or not path.exists():
                continue
            self._invalidate_path_caches(path)
            try:
                precomputed_bundle = self._load_precomputed_gradient_bundle(path)
            except Exception as exc:
                self.statusBar().showMessage(
                    f"Failed to load precomputed gradients from {path.name}: {exc}"
                )
                continue
            if precomputed_bundle is not None:
                loaded_precomputed_bundle = precomputed_bundle
                continue
            entry_id = self._file_entry_id(path)
            if entry_id in self._entries:
                continue
            entry = {
                "id": entry_id,
                "kind": "file",
                "path": path,
                "selected_key": None,
                "sample_index": None,
                "auto_title": True,
                "label": path.name,
            }
            self._entries[entry_id] = entry
            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
            item.setData(USER_ROLE, entry_id)
            self.file_list.addItem(item)
            added_paths.append(path)
            added_any = True

        if added_any and self.file_list.currentItem() is None:
            self.file_list.setCurrentRow(self.file_list.count() - 1)

        if loaded_precomputed_bundle is not None:
            self._activate_precomputed_gradient_bundle(loaded_precomputed_bundle)

        if not added_any:
            if loaded_precomputed_bundle is None:
                self.statusBar().showMessage("No valid .npz files added.")
        return added_paths

    def _remove_selected(self) -> None:
        item = self.file_list.currentItem()
        if item is None:
            return
        entry_id = item.data(USER_ROLE)
        if entry_id in self._entries:
            self._entries.pop(entry_id, None)
        self.titles.pop(entry_id, None)
        row = self.file_list.row(item)
        self.file_list.takeItem(row)
        if self.file_list.count() == 0:
            self._clear_plot()
        else:
            self._update_nbs_prepare_button()

    def _clear_files(self) -> None:
        self._entries.clear()
        self.titles.clear()
        self._covars_cache.clear()
        self._valid_keys_cache.clear()
        self._parcel_metadata_cache.clear()
        self._group_values_cache.clear()
        self._gradient_precomputed_bundle = None
        self._gradient_precomputed_selected_row = None
        self._gradient_use_precomputed_bundle = False
        self.file_list.clear()
        self._clear_plot()
        self._update_nbs_prepare_button()
        self.statusBar().showMessage("File list cleared.")

    def _get_valid_keys_cached(self, path: Path):
        path = Path(path)
        cached = self._valid_keys_cache.get(path)
        if cached is None:
            cached = list(_get_valid_keys(path))
            self._valid_keys_cache[path] = cached
        return list(cached)

    def _invalidate_path_caches(self, path: Path) -> None:
        path = Path(path)
        self._covars_cache.pop(path, None)
        self._valid_keys_cache.pop(path, None)
        self._parcel_metadata_cache.pop(path, None)
        self._group_values_cache.pop(path, None)

    def _load_parcel_metadata_cached(self, path: Path):
        path = Path(path)
        if path not in self._parcel_metadata_cache:
            self._parcel_metadata_cache[path] = _load_parcel_metadata(path)
        return self._parcel_metadata_cache[path]

    def _load_group_value_cached(self, path: Path, index: int):
        path = Path(path)
        if path not in self._group_values_cache:
            try:
                with np.load(path, allow_pickle=True) as npz:
                    group_data = npz["group"] if "group" in npz else None
            except Exception:
                group_data = None
            self._group_values_cache[path] = group_data

        group_data = self._group_values_cache.get(path)
        if group_data is None:
            return None
        if np.isscalar(group_data):
            return str(group_data)
        group_arr = np.asarray(group_data)
        if group_arr.ndim == 0:
            return str(group_arr.item())
        if index < 0 or index >= len(group_arr):
            return None
        return str(group_arr[index])

    def _refresh_key_options(self, entry) -> None:
        self.key_combo.blockSignals(True)
        valid_keys = self._get_valid_keys_cached(entry["path"])
        self.key_combo.clear()
        self.key_combo.addItems(valid_keys)
        selected_key = entry.get("selected_key")
        if selected_key and selected_key in valid_keys:
            self.key_combo.setCurrentText(selected_key)
        elif valid_keys:
            entry["selected_key"] = valid_keys[0]
            self.key_combo.setCurrentIndex(0)
        else:
            entry["selected_key"] = None
        self.key_combo.setEnabled(bool(valid_keys))
        self.key_combo.blockSignals(False)

    def _refresh_covars_options(self, source_path: Path, entry) -> None:
        info = self._covars_cache.get(source_path)
        if info is None:
            info = _load_covars_info(source_path)
            self._covars_cache[source_path] = info
        columns = _covars_columns(info)
        # Keep legacy fields populated for compatibility with existing metadata.
        self.covar_combo.blockSignals(True)
        self.covar_combo.clear()
        for col in columns:
            self.covar_combo.addItem(col)
        covar_name = entry.get("covar_name")
        if covar_name and self.covar_combo.findText(covar_name) >= 0:
            self.covar_combo.setCurrentText(covar_name)
        self.covar_combo.blockSignals(False)
        enabled = self.covar_combo.count() > 0
        self.covar_combo.setEnabled(enabled)
        self._update_selector_prepare_button()

    def _default_title_for_entry(self, entry) -> str:
        base_label = entry.get("label", "Matrix")
        if entry.get("kind") != "file":
            return base_label
        sample_index = entry.get("sample_index")
        if sample_index is None or sample_index < 0:
            return base_label
        source_path = entry.get("path")
        if source_path is None:
            return base_label
        info = self._covars_cache.get(source_path)
        if info is None:
            info = _load_covars_info(source_path)
            self._covars_cache[source_path] = info
        if info is None:
            return base_label
        participant = _covars_series(info, "participant_id")
        session = _covars_series(info, "session_id")
        if participant is None or session is None:
            return base_label
        if sample_index >= len(participant) or sample_index >= len(session):
            return base_label
        participant_value = str(participant[sample_index])
        session_value = str(session[sample_index])
        group_value = self._load_group_value_cached(source_path, sample_index)
        group, sub = _parse_participant_id(participant_value)
        if group_value:
            group = group_value
        sub = _strip_prefix(sub, "sub-")
        ses = _strip_prefix(session_value, "ses-")
        if group:
            return f"{group}-sub-{sub}_ses-{ses}"
        return f"sub-{sub}_ses-{ses}"

    def _apply_title_for_entry(self, entry, force: bool = False) -> str:
        entry_id = entry["id"]
        default_title = self._default_title_for_entry(entry)
        if force or entry.get("auto_title", True):
            self.titles[entry_id] = default_title
            self.title_edit.setText(default_title)
            return default_title
        current = self.title_edit.text().strip() or self.titles.get(entry_id, default_title)
        self.titles[entry_id] = current
        return current

    def _on_title_edited(self) -> None:
        entry = self._current_entry()
        if entry is None:
            return
        entry_id = entry["id"]
        default_title = self._default_title_for_entry(entry)
        text = self.title_edit.text().strip()
        if text and text != default_title:
            entry["auto_title"] = False
            self.titles[entry_id] = text
        else:
            entry["auto_title"] = True
            self.titles[entry_id] = default_title
            self.title_edit.setText(default_title)
        self._plot_selected()

    def _on_sample_changed(self, value: int) -> None:
        entry = self._current_entry()
        if entry is None or entry.get("kind") != "file":
            return
        entry["sample_index"] = value
        if entry.get("auto_title", True):
            self._apply_title_for_entry(entry, force=True)
        self._plot_selected()

    def _update_sample_controls(self, entry, axis, stack_len) -> None:
        self.sample_spin.blockSignals(True)
        if axis is None or stack_len is None:
            self.sample_spin.setEnabled(False)
            self.sample_spin.setRange(-1, 0)
            self.sample_spin.setSpecialValueText("Average")
            self.sample_spin.setValue(-1)
            self.sample_add_button.setEnabled(False)
            if entry is not None:
                entry["sample_index"] = None
                entry["stack_axis"] = None
                entry["stack_len"] = None
        else:
            self.sample_spin.setEnabled(True)
            self.sample_spin.setRange(-1, max(stack_len - 1, 0))
            self.sample_spin.setSpecialValueText("Average")
            entry.setdefault("sample_index", -1)
            if entry["sample_index"] is None or entry["sample_index"] >= stack_len:
                entry["sample_index"] = -1
            entry["stack_axis"] = axis
            entry["stack_len"] = stack_len
            self.sample_spin.setValue(entry["sample_index"])
            self.sample_add_button.setEnabled(True)
        self.sample_spin.blockSignals(False)

    def _reload_colormaps(self) -> None:
        names = self._available_colormap_names()
        current = self.cmap_combo.currentText() if hasattr(self, "cmap_combo") else ""
        self.cmap_combo.blockSignals(True)
        self.cmap_combo.clear()
        self.cmap_combo.addItems(names)
        if current and current in names:
            self.cmap_combo.setCurrentText(current)
        elif self._default_matrix_colormap in names:
            self.cmap_combo.setCurrentText(self._default_matrix_colormap)
        elif DEFAULT_COLORMAP in names:
            self.cmap_combo.setCurrentText(DEFAULT_COLORMAP)
        elif names:
            self.cmap_combo.setCurrentIndex(0)
        self.cmap_combo.blockSignals(False)
        current_3d = str(self._gradient_colormap_name or "").strip()
        if current_3d not in names:
            if self._default_gradient_colormap in names:
                current_3d = self._default_gradient_colormap
            elif "spectrum_fsl" in names:
                current_3d = "spectrum_fsl"
            elif names:
                current_3d = names[0]
            else:
                current_3d = "spectrum_fsl"
            self._gradient_colormap_name = current_3d
        self._sync_gradients_dialog_state()

    def _default_entry_display_settings(self):
        return {
            "matrix_colormap": self._default_matrix_colormap,
            "display_auto": True,
            "display_min_text": "",
            "display_max_text": "",
            "display_scale": "linear",
        }

    def _ensure_entry_display_settings(self, entry):
        if entry is None:
            return None
        defaults = self._default_entry_display_settings()
        available_names = self._available_colormap_names()
        cmap_name = str(entry.get("matrix_colormap", defaults["matrix_colormap"]) or "").strip()
        if not cmap_name or (available_names and cmap_name not in available_names):
            if defaults["matrix_colormap"] in available_names:
                cmap_name = defaults["matrix_colormap"]
            elif DEFAULT_COLORMAP in available_names:
                cmap_name = DEFAULT_COLORMAP
            elif available_names:
                cmap_name = available_names[0]
            else:
                cmap_name = defaults["matrix_colormap"]
        entry["matrix_colormap"] = cmap_name
        entry["display_auto"] = bool(entry.get("display_auto", defaults["display_auto"]))
        entry["display_min_text"] = str(entry.get("display_min_text", defaults["display_min_text"]) or "").strip()
        entry["display_max_text"] = str(entry.get("display_max_text", defaults["display_max_text"]) or "").strip()
        scale_name = str(entry.get("display_scale", defaults["display_scale"]) or "").strip().lower()
        entry["display_scale"] = "log" if scale_name.startswith("log") else "linear"
        return entry

    def _load_display_controls_for_entry(self, entry) -> None:
        if entry is None:
            return
        self._ensure_entry_display_settings(entry)

        self.cmap_combo.blockSignals(True)
        cmap_name = str(entry.get("matrix_colormap") or "").strip()
        if cmap_name and self.cmap_combo.findText(cmap_name) >= 0:
            self.cmap_combo.setCurrentText(cmap_name)
        elif self.cmap_combo.count() > 0:
            self.cmap_combo.setCurrentIndex(0)
            entry["matrix_colormap"] = self.cmap_combo.currentText().strip() or DEFAULT_COLORMAP
        self.cmap_combo.blockSignals(False)

        auto_scale = bool(entry.get("display_auto", True))
        self.display_auto_check.blockSignals(True)
        self.display_auto_check.setChecked(auto_scale)
        self.display_auto_check.blockSignals(False)

        self.display_min_edit.blockSignals(True)
        self.display_min_edit.setText(str(entry.get("display_min_text", "") or ""))
        self.display_min_edit.blockSignals(False)

        self.display_max_edit.blockSignals(True)
        self.display_max_edit.setText(str(entry.get("display_max_text", "") or ""))
        self.display_max_edit.blockSignals(False)

        self.display_scale_combo.blockSignals(True)
        self.display_scale_combo.setCurrentText("Log" if entry.get("display_scale") == "log" else "Linear")
        self.display_scale_combo.blockSignals(False)

        self.display_min_edit.setEnabled(not auto_scale)
        self.display_max_edit.setEnabled(not auto_scale)

    def _store_display_controls_for_entry(self, entry) -> None:
        if entry is None:
            return
        self._ensure_entry_display_settings(entry)
        entry["matrix_colormap"] = self.cmap_combo.currentText().strip() or DEFAULT_COLORMAP
        entry["display_auto"] = bool(self.display_auto_check.isChecked())
        entry["display_min_text"] = self.display_min_edit.text().strip()
        entry["display_max_text"] = self.display_max_edit.text().strip()
        entry["display_scale"] = (
            "log" if self.display_scale_combo.currentText().strip().lower().startswith("log") else "linear"
        )

    def _on_colormap_changed(self, *_args) -> None:
        entry = self._current_entry()
        if entry is None:
            return
        self._store_display_controls_for_entry(entry)
        self._plot_selected()

    def _selected_colormap_name(self, entry=None) -> str:
        if entry is not None:
            self._ensure_entry_display_settings(entry)
            name = str(entry.get("matrix_colormap") or "").strip()
            return name or DEFAULT_COLORMAP
        name = self.cmap_combo.currentText().strip()
        return name or DEFAULT_COLORMAP

    def _selected_colormap(self, entry=None):
        name = self._selected_colormap_name(entry)
        if name in self._custom_cmaps:
            try:
                return self._colorbar.load_fsl_cmap(name)
            except Exception:
                return name
        return name

    def _set_plot_title(self, full_title: str, tooltip_text: str = "") -> None:
        self._plot_title_full = str(full_title or "")
        self._plot_title_tooltip = str(tooltip_text or full_title or "")
        self._update_plot_title_label()

    def _update_plot_title_label(self) -> None:
        if not hasattr(self, "plot_title_label"):
            return
        full_title = self._plot_title_full
        self.plot_title_label.setToolTip(self._plot_title_tooltip)
        if not full_title:
            self.plot_title_label.setText("")
            return
        metrics = QFontMetrics(self.plot_title_label.font())
        avail = max(self.plot_title_label.width() - 4, 60)
        elided = metrics.elidedText(full_title, Qt.ElideRight, avail)
        self.plot_title_label.setText(elided)

    def _current_display_limits(self, entry=None):
        if entry is not None:
            self._ensure_entry_display_settings(entry)
            auto_scale = bool(entry.get("display_auto", True))
            min_text = str(entry.get("display_min_text", "") or "").strip()
            max_text = str(entry.get("display_max_text", "") or "").strip()
        else:
            if not hasattr(self, "display_auto_check"):
                return None, None, None
            auto_scale = bool(self.display_auto_check.isChecked())
            min_text = self.display_min_edit.text().strip() if hasattr(self, "display_min_edit") else ""
            max_text = self.display_max_edit.text().strip() if hasattr(self, "display_max_edit") else ""
        if auto_scale:
            return None, None, None
        vmin = None
        vmax = None
        if min_text:
            try:
                vmin = float(min_text)
            except ValueError:
                return None, None, "Display min must be numeric."
        if max_text:
            try:
                vmax = float(max_text)
            except ValueError:
                return None, None, "Display max must be numeric."
        if vmin is not None and vmax is not None and vmin >= vmax:
            return None, None, "Display min must be smaller than max."
        return vmin, vmax, None

    def _current_display_scale(self, entry=None) -> str:
        if entry is not None:
            self._ensure_entry_display_settings(entry)
            choice = str(entry.get("display_scale", "linear") or "").strip().lower()
        else:
            if not hasattr(self, "display_scale_combo"):
                return "linear"
            choice = self.display_scale_combo.currentText().strip().lower()
        if choice.startswith("log"):
            return "log"
        return "linear"

    def _log_scale_error(self, matrix, vmin, vmax):
        if matrix is None:
            return "Log scale requires matrix values."
        values = np.asarray(matrix, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return "Log scale requires finite values."
        max_val = float(np.max(finite))
        if max_val <= 0:
            return "Log scale requires positive values (max > 0)."
        min_val = float(np.min(finite))
        if min_val < 0:
            return "Log scale does not support negative values."
        if vmin is not None and vmin <= 0:
            return "Display min must be > 0 for log scale."
        if vmax is not None and vmax <= 0:
            return "Display max must be > 0 for log scale."
        return None

    def _on_display_scaling_changed(self, *_args) -> None:
        auto_scale = self.display_auto_check.isChecked()
        self.display_min_edit.setEnabled(not auto_scale)
        self.display_max_edit.setEnabled(not auto_scale)
        entry = self._current_entry()
        if entry is not None:
            self._store_display_controls_for_entry(entry)
            self._plot_selected()

    def _on_zoom_region_toggled(self, *_args) -> None:
        if not hasattr(self, "zoom_region_check"):
            return
        if self.zoom_region_check.isChecked():
            self._enable_zoom_selector()
            self.statusBar().showMessage("Drag on the matrix to zoom into a region.")
        else:
            self._reset_zoom_selector()

    def _update_zoom_selector(self) -> None:
        self._reset_zoom_selector()
        if hasattr(self, "zoom_region_check") and self.zoom_region_check.isChecked():
            self._enable_zoom_selector()

    def _enable_zoom_selector(self) -> None:
        if RectangleSelector is None:
            self.statusBar().showMessage("Region zoom is unavailable (matplotlib widgets missing).")
            if hasattr(self, "zoom_region_check"):
                self.zoom_region_check.blockSignals(True)
                self.zoom_region_check.setChecked(False)
                self.zoom_region_check.blockSignals(False)
            return
        if self._current_axes is None:
            return
        if self._zoom_selector is not None:
            try:
                self._zoom_selector.set_active(True)
            except Exception:
                pass
            return
        self._zoom_selector = RectangleSelector(
            self._current_axes,
            self._on_zoom_region_selected,
            useblit=False,
            button=[1],
            minspanx=1,
            minspany=1,
        )
        try:
            self._zoom_selector.set_active(True)
        except Exception:
            pass

    def _reset_zoom_selector(self) -> None:
        if self._zoom_selector is None:
            return
        try:
            self._zoom_selector.set_active(False)
        except Exception:
            pass
        self._zoom_selector = None

    def _on_zoom_region_selected(self, eclick, erelease) -> None:
        if self._current_axes is None or self._current_matrix is None:
            return
        if (
            eclick.xdata is None
            or eclick.ydata is None
            or erelease.xdata is None
            or erelease.ydata is None
        ):
            return
        x0, x1 = eclick.xdata, erelease.xdata
        y0, y1 = eclick.ydata, erelease.ydata
        if abs(x1 - x0) < 1 or abs(y1 - y0) < 1:
            return
        nrows, ncols = self._current_matrix.shape[:2]
        x0 = max(-0.5, min(ncols - 0.5, x0))
        x1 = max(-0.5, min(ncols - 0.5, x1))
        y0 = max(-0.5, min(nrows - 0.5, y0))
        y1 = max(-0.5, min(nrows - 0.5, y1))
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        ax = self._current_axes
        if ax.xaxis_inverted():
            ax.set_xlim(xmax, xmin)
        else:
            ax.set_xlim(xmin, xmax)
        if ax.yaxis_inverted():
            ax.set_ylim(ymax, ymin)
        else:
            ax.set_ylim(ymin, ymax)
        if hasattr(self, "zoom_reset_button"):
            self.zoom_reset_button.setEnabled(True)
        self.canvas.draw_idle()

    def _reset_matrix_zoom(self) -> None:
        if self._current_axes is None:
            return
        if self._matrix_full_xlim is not None and self._matrix_full_ylim is not None:
            self._current_axes.set_xlim(self._matrix_full_xlim)
            self._current_axes.set_ylim(self._matrix_full_ylim)
        else:
            self._current_axes.autoscale()
        if hasattr(self, "zoom_reset_button"):
            self.zoom_reset_button.setEnabled(False)
        self.canvas.draw_idle()

    def _on_splitter_moved(self, *_args) -> None:
        sizes = self.main_splitter.sizes()
        if len(sizes) == 3:
            if sizes[0] > 0:
                self._left_panel_saved_width = sizes[0]
            if sizes[2] > 0:
                self._right_panel_saved_width = sizes[2]
        self._refresh_sidebar_toggle_buttons()

    def _refresh_sidebar_toggle_buttons(self) -> None:
        if not hasattr(self, "left_sidebar_button") or not hasattr(self, "right_sidebar_button"):
            return
        sizes = self.main_splitter.sizes()
        if len(sizes) != 3:
            return
        left_visible = sizes[0] > 0
        right_visible = sizes[2] > 0
        self.left_sidebar_button.setText("◀" if left_visible else "▶")
        self.left_sidebar_button.setToolTip(
            "Collapse data sidebar" if left_visible else "Expand data sidebar"
        )
        self.right_sidebar_button.setText("▶" if right_visible else "◀")
        self.right_sidebar_button.setToolTip(
            "Collapse analyses sidebar" if right_visible else "Expand analyses sidebar"
        )

    def _toggle_left_sidebar(self) -> None:
        sizes = self.main_splitter.sizes()
        if len(sizes) != 3:
            return
        left, center, right = sizes
        total = max(left + center + right, 1)
        if left > 0:
            self._left_panel_saved_width = left
            left = 0
        else:
            desired = max(self._left_panel_saved_width, 220)
            max_left = max(total - right - 320, 120)
            left = min(desired, max_left)
        center = max(total - left - right, 260)
        self.main_splitter.setSizes([left, center, right])
        self._refresh_sidebar_toggle_buttons()

    def _toggle_right_sidebar(self) -> None:
        sizes = self.main_splitter.sizes()
        if len(sizes) != 3:
            return
        left, center, right = sizes
        total = max(left + center + right, 1)
        if right > 0:
            self._right_panel_saved_width = right
            right = 0
        else:
            desired = max(self._right_panel_saved_width, 200)
            max_right = max(total - left - 320, 120)
            right = min(desired, max_right)
        center = max(total - left - right, 260)
        self.main_splitter.setSizes([left, center, right])
        self._refresh_sidebar_toggle_buttons()

    def _selected_surface_colormap_name(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                name = dialog.selected_colormap()
            except Exception:
                name = ""
            if name:
                self._gradient_colormap_name = name
                return name
        name = str(self._gradient_colormap_name or "").strip()
        if name:
            return name
        return "spectrum_fsl"

    def _selected_surface_colormap(self, name: str = None):
        if name is None:
            name = self._selected_surface_colormap_name()
        else:
            name = str(name or "").strip() or self._selected_surface_colormap_name()
        if name == "spectrum_fsl":
            try:
                return self._colorbar.load_fsl_cmap(name)
            except Exception:
                try:
                    return self._colorbar.bars(name)
                except Exception:
                    return "viridis"
        if name in COLORBAR_BARS:
            try:
                return self._colorbar.bars(name)
            except Exception:
                return "viridis"
        if name in self._custom_cmaps:
            try:
                return self._colorbar.load_fsl_cmap(name)
            except Exception:
                try:
                    return self._colorbar.bars(name)
                except Exception:
                    return "viridis"
        return name

    def _move_selected(self, direction: int) -> None:
        row = self.file_list.currentRow()
        if row < 0:
            return
        new_row = row + direction
        if new_row < 0 or new_row >= self.file_list.count():
            return
        item = self.file_list.takeItem(row)
        self.file_list.insertItem(new_row, item)
        self.file_list.setCurrentRow(new_row)

    def _on_key_changed(self) -> None:
        entry = self._current_entry()
        if entry and entry.get("kind") == "file":
            key = self.key_combo.currentText().strip()
            entry["selected_key"] = key or None
            entry["sample_index"] = None
        self._plot_selected()

    def _open_histogram(self) -> None:
        if self.file_list.count() == 0:
            self.statusBar().showMessage("No matrices to plot.")
            return
        dialog = HistogramDialog(self._entries, list(self._entry_ids()), self.titles, parent=self)
        dialog.show()
        self._hist_dialog = dialog

    def _ensure_entry_key(self, entry):
        if entry.get("kind") != "file":
            return entry.get("selected_key")
        key = entry.get("selected_key")
        if key:
            return key
        valid_keys = self._get_valid_keys_cached(entry["path"])
        if not valid_keys:
            return None
        entry["selected_key"] = valid_keys[0]
        return entry["selected_key"]

    def _matrix_for_entry(self, entry):
        if entry.get("kind") == "derived":
            return entry.get("matrix"), entry.get("selected_key")
        key = self._ensure_entry_key(entry)
        if not key:
            raise KeyError("No valid matrix key selected")
        raw = _load_matrix_from_npz(entry["path"], key, average=False)
        if raw.ndim == 3:
            axis = _stack_axis(raw.shape)
            if axis is None:
                matrix = _average_to_square(raw)
            else:
                entry.setdefault("sample_index", -1)
                entry["stack_axis"] = axis
                entry["stack_len"] = raw.shape[axis]
                sample_index = entry.get("sample_index", -1)
                if sample_index is None or sample_index < 0:
                    matrix = _average_to_square(raw)
                else:
                    matrix = _select_stack_slice(raw, axis, sample_index)
        else:
            matrix = raw
        return matrix, key

    @staticmethod
    def _safe_name_fragment(text: str) -> str:
        token = str(text or "").strip()
        if not token:
            return "matrix"
        cleaned = []
        for ch in token:
            if ch.isalnum() or ch in {"-", "_"}:
                cleaned.append(ch)
            else:
                cleaned.append("_")
        out = "".join(cleaned).strip("_")
        while "__" in out:
            out = out.replace("__", "_")
        return out or "matrix"

    def _collect_export_metadata(self, entry, selected_key):
        metadata = {}
        labels = None
        names = None

        source_path_raw = entry.get("source_path", entry.get("path"))
        source_path = Path(source_path_raw) if source_path_raw else None
        if source_path is not None and source_path.exists():
            try:
                with np.load(source_path, allow_pickle=True) as npz:
                    for key in PARCEL_LABEL_KEYS:
                        if key in npz:
                            labels = np.asarray(npz[key])
                            break
                    for key in PARCEL_NAME_KEYS:
                        if key in npz:
                            names = np.asarray(npz[key])
                            break
                    for key in ("group", "modality", "metabolites"):
                        if key in npz:
                            metadata[key] = np.asarray(npz[key])
            except Exception:
                pass

            if labels is None or names is None:
                try:
                    extra_labels, extra_names = self._load_parcel_metadata_cached(source_path)
                    if labels is None and extra_labels is not None:
                        labels = np.asarray(extra_labels)
                    if names is None and extra_names is not None:
                        names = np.asarray(extra_names)
                except Exception:
                    pass

        if labels is None and self._current_parcel_labels:
            labels = np.asarray(self._current_parcel_labels)
        if names is None and self._current_parcel_names:
            names = np.asarray(self._current_parcel_names)

        if labels is not None:
            metadata["parcel_labels_group"] = labels
        if names is not None:
            metadata["parcel_names_group"] = names
        if source_path is not None:
            metadata["source_file"] = np.asarray(str(source_path))
        if selected_key:
            metadata["source_key"] = np.asarray(str(selected_key))
        sample_index = entry.get("sample_index")
        if sample_index is not None:
            try:
                metadata["sample_index"] = np.asarray(int(sample_index))
            except Exception:
                pass
        return metadata

    def _write_selected_matrix_to_file(self) -> None:
        entry = self._current_entry()
        if entry is None:
            self.statusBar().showMessage("No matrix selected.")
            return

        try:
            matrix, selected_key = self._matrix_for_entry(entry)
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to resolve selected matrix: {exc}")
            return

        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            self.statusBar().showMessage("Selected matrix is not a square 2D matrix.")
            return

        source_path_raw = entry.get("source_path", entry.get("path"))
        source_path = Path(source_path_raw) if source_path_raw else None
        source_stem = source_path.stem if source_path is not None else self._safe_name_fragment(entry.get("label", "matrix"))
        key_part = self._safe_name_fragment(selected_key or "matrix")
        default_name = f"{self._safe_name_fragment(source_stem)}_{key_part}_matrix_pop_avg.npz"
        start_dir = self._default_results_dir()

        save_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Write selected matrix to NPZ",
            str(start_dir / default_name),
            "NumPy archive (*.npz);;All files (*)",
        )
        if not save_path:
            return
        output_path = Path(save_path)
        if output_path.suffix.lower() != ".npz":
            output_path = output_path.with_suffix(".npz")

        payload = {"matrix_pop_avg": matrix}
        payload.update(self._collect_export_metadata(entry, selected_key))
        try:
            np.savez(output_path, **payload)
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to write NPZ: {exc}")
            return
        self.statusBar().showMessage(f"Wrote selected matrix to {output_path.name}.")

    def _export_grid(self) -> None:
        if self.file_list.count() == 0:
            self.statusBar().showMessage("No matrices to export.")
            return
        current_entry = self._current_entry()
        if current_entry is not None:
            self._store_display_controls_for_entry(current_entry)

        default_output_path = self._export_grid_output_path or str(
            self._default_results_dir() / "connectome_grid.pdf"
        )
        dialog = ExportGridDialog(
            default_path=default_output_path,
            default_columns=self._export_grid_columns,
            rotate=self._export_grid_rotate,
            parent=self,
        )
        dialog.setStyleSheet(self.styleSheet())
        dialog.set_theme(self._theme_name)
        if dialog.exec() != _dialog_accepted_code():
            return

        values = dialog.values()
        self._export_grid_output_path = values["output_path"]
        self._export_grid_columns = int(values["columns"])
        self._export_grid_rotate = bool(values["rotate"])
        self._export_grid_selected_filter = str(values.get("selected_filter") or "PDF (*.pdf)")

        output_path = Path(values["output_path"]).expanduser()
        selected_filter = self._export_grid_selected_filter
        if output_path.suffix.lower() not in {".pdf", ".svg", ".png"}:
            if "PDF" in selected_filter:
                output_path = output_path.with_suffix(".pdf")
            elif "SVG" in selected_filter:
                output_path = output_path.with_suffix(".svg")
            else:
                output_path = output_path.with_suffix(".png")
        self._export_grid_output_path = str(output_path)

        plot_items = []
        skipped = []
        for entry_id in self._entry_ids():
            entry = self._entries.get(entry_id)
            if entry is None:
                continue
            try:
                matrix, _ = self._matrix_for_entry(entry)
            except Exception as exc:
                label = entry.get("label", entry_id)
                skipped.append(f"{label} ({exc})")
                continue
            label = entry.get("label", entry_id)
            vmin, vmax, scaling_error = self._current_display_limits(entry)
            if scaling_error:
                self.statusBar().showMessage(f"{label}: {scaling_error}")
                return
            zscale = self._current_display_scale(entry)
            if zscale == "log":
                log_error = self._log_scale_error(matrix, vmin, vmax)
                if log_error:
                    self.statusBar().showMessage(f"{label}: {log_error}")
                    return
            plot_items.append(
                {
                    "matrix": matrix,
                    "title": self.titles.get(entry_id, entry.get("label", "Matrix")),
                    "colormap": self._selected_colormap(entry),
                    "vmin": vmin,
                    "vmax": vmax,
                    "zscale": zscale,
                }
            )

        if not plot_items:
            self.statusBar().showMessage("No matrices exported (missing keys or load errors).")
            return

        cols = min(self._export_grid_columns, len(plot_items))
        rows = int(math.ceil(len(plot_items) / cols))
        export_figure = Figure(figsize=(4 * cols, 4 * rows))
        axes = export_figure.subplots(rows, cols)
        if isinstance(axes, np.ndarray):
            flat_axes = axes.flatten()
        else:
            flat_axes = [axes]

        rotate = self._export_grid_rotate
        for idx, plot_item in enumerate(plot_items):
            matrix = plot_item["matrix"]
            ax = flat_axes[idx]
            SimMatrixPlot.plot_simmatrix(
                matrix,
                ax=ax,
                titles=plot_item["title"],
                colormap=plot_item["colormap"],
                vmin=plot_item["vmin"],
                vmax=plot_item["vmax"],
                zscale=plot_item["zscale"],
            )
            _remove_axes_border(ax)
            if rotate:
                _apply_rotation(ax, matrix, 45.0)

        for ax in flat_axes[len(plot_items):]:
            ax.axis("off")

        export_figure.tight_layout()
        export_figure.savefig(str(output_path))

        if skipped:
            self.statusBar().showMessage(
                f"Exported {len(plot_items)} matrices to {output_path.name}. "
                f"Skipped {len(skipped)}."
            )
        else:
            self.statusBar().showMessage(f"Exported {len(plot_items)} matrices to {output_path.name}.")

    def _on_selection_changed(self, current, _previous) -> None:
        if current is None:
            self._clear_plot()
            return
        entry_id = current.data(USER_ROLE)
        entry = self._entries.get(entry_id)
        if entry is None:
            self._clear_plot()
            return

        if entry.get("kind") == "file":
            self.key_combo.setEnabled(True)
            self._refresh_key_options(entry)
        else:
            self.key_combo.blockSignals(True)
            self.key_combo.clear()
            selected_key = entry.get("selected_key")
            if selected_key:
                self.key_combo.addItem(selected_key)
                self.key_combo.setCurrentText(selected_key)
            self.key_combo.setEnabled(False)
            self.key_combo.blockSignals(False)

        source_path = entry.get("source_path", entry.get("path"))
        if source_path:
            self._refresh_covars_options(source_path, entry)
        self._load_display_controls_for_entry(entry)

        if entry.get("auto_title", True):
            self._apply_title_for_entry(entry, force=True)
        else:
            self.title_edit.setText(self.titles.get(entry_id, entry.get("label", "Matrix")))
        self._plot_selected()
        self._update_write_to_file_button()
        self._update_view_labels_button()

    def _plot_selected(self, *_args) -> None:
        entry_id = self._current_entry_id()
        if entry_id is None:
            self._clear_plot()
            return
        entry = self._entries.get(entry_id)
        if entry is None:
            self._clear_plot()
            return
        self._store_display_controls_for_entry(entry)

        if entry.get("kind") == "derived":
            matrix = entry.get("matrix")
            key = entry.get("selected_key")
            source_path = entry.get("source_path")
            self._update_sample_controls(entry, None, None)
        else:
            key = self._ensure_entry_key(entry)
            if not key:
                self._clear_plot()
                self.statusBar().showMessage("No valid matrix key selected.")
                return
            source_path = entry.get("path")
            try:
                raw = _load_matrix_from_npz(entry["path"], key, average=False)
            except Exception as exc:
                self._clear_plot()
                self.statusBar().showMessage(f"Failed to load {entry.get('label', 'file')}: {exc}")
                return
            if raw.ndim == 3:
                axis = _stack_axis(raw.shape)
                if axis is None:
                    self._update_sample_controls(entry, None, None)
                    matrix = _average_to_square(raw)
                else:
                    self._update_sample_controls(entry, axis, raw.shape[axis])
                    sample_index = entry.get("sample_index", -1)
                    if sample_index is None or sample_index < 0:
                        matrix = _average_to_square(raw)
                    else:
                        matrix = _select_stack_slice(raw, axis, sample_index)
            else:
                self._update_sample_controls(entry, None, None)
                matrix = raw

        current_title = self._apply_title_for_entry(entry)

        labels, names = (None, None)
        if source_path:
            labels, names = self._load_parcel_metadata_cached(source_path)
        labels_list = _to_string_list(labels)
        names_list = _to_string_list(names)
        if labels_list and len(labels_list) != matrix.shape[0]:
            labels_list = None
        if names_list and len(names_list) != matrix.shape[0]:
            names_list = None

        self._current_matrix = matrix
        self._current_parcel_labels = labels_list
        self._current_parcel_names = names_list

        vmin, vmax, scaling_error = self._current_display_limits(entry)
        if scaling_error:
            self.statusBar().showMessage(scaling_error)
            return

        colormap = self._selected_colormap(entry)
        zscale = self._current_display_scale(entry)
        if zscale == "log":
            log_error = self._log_scale_error(matrix, vmin, vmax)
            if log_error:
                self.statusBar().showMessage(log_error)
                return
        self.figure.clear()
        self._reset_zoom_selector()
        ax = self.figure.add_subplot(111)
        SimMatrixPlot.plot_simmatrix(
            matrix,
            ax=ax,
            titles=current_title,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            zscale=zscale,
        )
        self._hover_vline = ax.axvline(
            0.0,
            color="#22c55e",
            linewidth=0.8,
            alpha=0.85,
            visible=False,
            zorder=20,
        )
        self._hover_hline = ax.axhline(
            0.0,
            color="#22c55e",
            linewidth=0.8,
            alpha=0.85,
            visible=False,
            zorder=20,
        )
        _remove_axes_border(ax)
        self._current_axes = ax
        self._matrix_full_xlim = ax.get_xlim()
        self._matrix_full_ylim = ax.get_ylim()
        if hasattr(self, "zoom_reset_button"):
            self.zoom_reset_button.setEnabled(False)
        self._update_zoom_selector()
        title_tooltip = str(source_path) if source_path is not None else current_title
        self._set_plot_title(current_title, tooltip_text=title_tooltip)
        self._reset_gradients_output()
        self.canvas.draw_idle()
        self._update_nbs_prepare_button()
        key_text = f", {key}" if key else ""
        self.statusBar().showMessage(f"Plotted {entry.get('label', 'matrix')}{key_text}.")

    def _average_selected(self) -> None:
        entry = self._current_entry()
        if entry is None:
            self.statusBar().showMessage("No matrix selected.")
            return
        source_path = entry.get("path") if entry.get("kind") == "file" else entry.get("source_path")
        if source_path is None:
            self.statusBar().showMessage("No source file for averaging.")
            return
        key = entry.get("selected_key") or self._ensure_entry_key(entry)
        if not key:
            self.statusBar().showMessage("No valid matrix key selected.")
            return
        info = self._covars_cache.get(source_path)
        if info is None:
            info = _load_covars_info(source_path)
            self._covars_cache[source_path] = info
        if info is None:
            self.statusBar().showMessage("Covars not found in file.")
            return
        covar_name = self.covar_combo.currentText().strip()
        if not covar_name:
            self.statusBar().showMessage("Select a covariate.")
            return
        value_text = self.covar_value_edit.text().strip()
        if not value_text:
            self.statusBar().showMessage("Enter a covariate value.")
            return
        value_float = None
        try:
            value_float = float(value_text)
        except ValueError:
            value_float = None
        series = _covars_series(info, covar_name)
        if series is None:
            self.statusBar().showMessage("Covar column not available.")
            return
        series_arr = np.asarray(series)
        if _series_is_numeric(series_arr):
            if value_float is None:
                self.statusBar().showMessage("Covar value must be numeric.")
                return
            series_float = np.asarray(series_arr, dtype=float)
            mask = np.isclose(series_float, value_float)
        else:
            series_str = _decode_strings(series_arr)
            mask = series_str == value_text
        indices = np.where(mask)[0]
        if indices.size == 0:
            self.statusBar().showMessage("No covar matches the selected value.")
            return

        try:
            with np.load(source_path, allow_pickle=True) as npz:
                if key not in npz:
                    raise KeyError(f"Key '{key}' not found")
                matrix = npz[key]
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to load matrix: {exc}")
            return

        if matrix.ndim != 3:
            self.statusBar().showMessage("Selected matrix is not multi-dimensional.")
            return

        shape = matrix.shape
        axis = None
        if shape[0] == shape[1] != shape[2]:
            axis = 2
        elif shape[0] == shape[2] != shape[1]:
            axis = 1
        elif shape[1] == shape[2] != shape[0]:
            axis = 0
        else:
            self.statusBar().showMessage("Matrix is not a stack of square matrices.")
            return

        covars_len = len(series_arr)
        if shape[axis] != covars_len:
            self.statusBar().showMessage("Covars length does not match matrix stack size.")
            return

        if axis == 0:
            subset = matrix[indices, :, :]
            averaged = subset.mean(axis=0)
        elif axis == 1:
            subset = matrix[:, indices, :]
            averaged = subset.mean(axis=1)
        else:
            subset = matrix[:, :, indices]
            averaged = subset.mean(axis=2)

        derived_id = self._new_derived_id()
        label = f"avg {covar_name}={value_text} ({Path(source_path).name})"
        covar_value = value_float if value_float is not None else value_text
        self._entries[derived_id] = {
            "id": derived_id,
            "kind": "derived",
            "matrix": averaged,
            "source_path": source_path,
            "selected_key": key,
            "covar_name": covar_name,
            "covar_value": covar_value,
            "sample_index": None,
            "auto_title": True,
            "label": label,
        }
        self.titles[derived_id] = label
        item = QListWidgetItem(label)
        item.setData(USER_ROLE, derived_id)
        self.file_list.addItem(item)
        self.file_list.setCurrentItem(item)
        self.statusBar().showMessage(f"Added averaged matrix ({covar_name}={value_text}).")

    def _add_sample_entry(self) -> None:
        entry = self._current_entry()
        if entry is None or entry.get("kind") != "file":
            self.statusBar().showMessage("Select a file-based matrix first.")
            return
        key = self._ensure_entry_key(entry)
        if not key:
            self.statusBar().showMessage("No valid matrix key selected.")
            return
        source_path = entry.get("path")
        if source_path is None:
            self.statusBar().showMessage("No source file available.")
            return
        try:
            raw = _load_matrix_from_npz(source_path, key, average=False)
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to load matrix: {exc}")
            return
        if raw.ndim != 3:
            self.statusBar().showMessage("Selected matrix is not multi-dimensional.")
            return
        axis = _stack_axis(raw.shape)
        if axis is None:
            self.statusBar().showMessage("Matrix is not a stack of square matrices.")
            return
        sample_index = entry.get("sample_index", -1)
        if sample_index is None or sample_index < 0:
            matrix = _average_to_square(raw)
            label = f"avg {key} ({Path(source_path).name})"
        else:
            if sample_index >= raw.shape[axis]:
                self.statusBar().showMessage("Sample index out of range.")
                return
            matrix = _select_stack_slice(raw, axis, sample_index)
            label = self._default_title_for_entry(entry)

        derived_id = self._new_derived_id()
        self._entries[derived_id] = {
            "id": derived_id,
            "kind": "derived",
            "matrix": matrix,
            "source_path": source_path,
            "selected_key": key,
            "sample_index": sample_index,
            "auto_title": True,
            "label": label,
        }
        self.titles[derived_id] = label
        item = QListWidgetItem(label)
        item.setData(USER_ROLE, derived_id)
        self.file_list.addItem(item)
        self.file_list.setCurrentItem(item)
        self.statusBar().showMessage("Added sample matrix to list.")

    def _compute_gradients(self) -> None:
        self._reset_gradients_output()
        entry = self._selected_gradient_entry()
        if entry is None:
            self.statusBar().showMessage("No workspace matrix selected for gradients.")
            return
        matrix_label = self._gradient_matrix_label_for_entry(entry)
        try:
            conn_matrix = np.asarray(self._gradient_matrix_for_entry(entry), dtype=float)
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to load {matrix_label} for gradients: {exc}")
            return
        if conn_matrix.ndim != 2 or conn_matrix.shape[0] != conn_matrix.shape[1]:
            self.statusBar().showMessage(f"Gradients require a square matrix. {matrix_label} is not square.")
            return

        source_dir = self._default_results_dir()
        if self._active_parcellation_data is None:
            if not self._select_parcellation_template():
                self.statusBar().showMessage("Gradient compute canceled (no template selected).")
                return

        try:
            import nibabel as nib
        except Exception as exc:
            self.statusBar().showMessage(f"nibabel not available: {exc}")
            return

        template_img = self._active_parcellation_img
        template_data = self._active_parcellation_data
        if template_img is None or template_data is None:
            self.statusBar().showMessage("No active parcellation template.")
            return

        source_path = self._gradient_entry_source_path(entry)
        if source_path is None or not source_path.exists():
            self.statusBar().showMessage("Projection requires a source .npz with parcel labels.")
            return
        parcel_labels, _ = self._load_parcel_metadata_cached(source_path)
        label_indices = _coerce_label_indices(parcel_labels, conn_matrix.shape[0])
        if label_indices is None:
            self.statusBar().showMessage(
                "parcel_labels_group missing/invalid or does not match matrix nodes."
            )
            return
        template_labels = set(np.asarray(template_data, dtype=int).reshape(-1).tolist())
        template_labels.discard(0)
        if not template_labels:
            self.statusBar().showMessage("Template has no non-zero labels.")
            return
        keep_indices = [idx for idx, label in enumerate(label_indices) if label in template_labels]
        if not keep_indices:
            self.statusBar().showMessage(
                "No overlap between matrix parcel_labels_group and active parcellation labels."
            )
            return
        projection_labels = [label_indices[idx] for idx in keep_indices]

        _, parcel_names = self._load_parcel_metadata_cached(source_path)
        parcel_names = _to_string_list(parcel_names)
        kept_names = None
        if parcel_names and len(parcel_names) == conn_matrix.shape[0]:
            kept_names = [parcel_names[idx] for idx in keep_indices]

        n_grad = self._current_gradient_component_count()
        self._gradients_busy = True
        self._set_gradients_progress(0, n_grad, 0, f"0/{n_grad} components")
        self._sync_gradients_dialog_state()
        QApplication.processEvents()

        gradients = np.zeros((conn_matrix.shape[0], n_grad), dtype=float)
        projected_maps = []
        try:
            for comp_idx in range(1, n_grad + 1):
                try:
                    component = nettools.dimreduce_matrix(
                        conn_matrix,
                        method="diffusion",
                        scale_factor=1.0,
                        output_dim=comp_idx,
                    )
                except Exception as exc:
                    self.statusBar().showMessage(f"Failed to compute component {comp_idx}: {exc}")
                    return
                component = np.asarray(component, dtype=float).reshape(-1)
                if component.size != conn_matrix.shape[0]:
                    self.statusBar().showMessage(
                        f"Component {comp_idx} size mismatch ({component.size} vs {conn_matrix.shape[0]})."
                    )
                    return
                gradients[:, comp_idx - 1] = component
                component_restricted = component[keep_indices]
                try:
                    projected = nettools.project_to_3dspace(
                        component_restricted,
                        template_data,
                        projection_labels,
                    )
                except Exception as exc:
                    self.statusBar().showMessage(f"Failed to project component {comp_idx}: {exc}")
                    return
                projected_maps.append(np.asarray(projected, dtype=np.float32))
                self._set_gradients_progress(0, n_grad, comp_idx, f"{comp_idx}/{n_grad} components")
                QApplication.processEvents()
        finally:
            self._gradients_busy = False
            self._sync_gradients_dialog_state()

        if n_grad == 1:
            output_data = projected_maps[0]
        else:
            output_data = np.stack(projected_maps, axis=-1)
        support_mask = np.asarray(
            np.isin(template_data, np.asarray(projection_labels, dtype=int)),
            dtype=np.float32,
        )

        source_stem = source_path.stem if source_path is not None else "matrix"
        default_name = f"{source_stem}_diffusion_components-{n_grad}.nii.gz"

        self._last_gradients = {
            "gradients": gradients,
            "n_grad": n_grad,
            "n_nodes": conn_matrix.shape[0],
            "projected_data": np.asarray(output_data, dtype=np.float32),
            "affine": np.asarray(template_img.affine, dtype=float),
            "header": template_img.header.copy(),
            "source_name": matrix_label or (source_path.name if source_path is not None else "matrix"),
            "source_dir": str(source_dir),
            "output_name": default_name,
            "keep_indices": np.asarray(keep_indices, dtype=int),
            "projection_labels": np.asarray(projection_labels, dtype=int),
            "support_mask": support_mask,
            "template_path": str(self._active_parcellation_path) if self._active_parcellation_path else "",
            "parcel_names": kept_names,
            "matrix_entry_id": self._selected_gradient_entry_id(),
            "matrix_label": matrix_label,
        }
        self._set_gradients_progress(0, n_grad, n_grad, f"{n_grad}/{n_grad} components (done)")
        self._sync_gradients_dialog_state()
        self.statusBar().showMessage(
            f"Computed {n_grad} projected component(s). Click Write to File, Render fsaverage, Classify, or Render Network."
        )

    def _save_gradients_projection(self) -> None:
        if not self._last_gradients:
            self.statusBar().showMessage("No projected gradients to save. Click Compute first.")
            return
        try:
            projected_data = self._ensure_projected_gradient_data(
                int(self._last_gradients.get("n_grad", 1))
            )
        except Exception as exc:
            self.statusBar().showMessage(f"No projected data available to save: {exc}")
            return

        try:
            import nibabel as nib
        except Exception as exc:
            self.statusBar().showMessage(f"nibabel not available: {exc}")
            return

        base_dir = Path(self._last_gradients.get("source_dir", str(self._default_results_dir())))
        default_name = self._last_gradients.get("output_name", "diffusion_components.nii.gz")
        save_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save gradient projection",
            str(base_dir / default_name),
            "NIfTI GZip (*.nii.gz);;NIfTI (*.nii);;All files (*)",
        )
        if not save_path:
            return

        output_path = Path(save_path)
        lower_name = output_path.name.lower()
        if not (lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")):
            if "NIfTI (*.nii)" in selected_filter:
                output_path = output_path.with_suffix(".nii")
            else:
                output_path = output_path.with_suffix(".nii.gz")

        affine = self._last_gradients.get("affine")
        header = self._last_gradients.get("header")
        if affine is None:
            affine = np.eye(4)
        try:
            if header is not None:
                out_img = nib.Nifti1Image(projected_data, affine, header.copy())
            else:
                out_img = nib.Nifti1Image(projected_data, affine)
            nib.save(out_img, str(output_path))
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to save NIfTI: {exc}")
            return
        self.statusBar().showMessage(f"Saved projection to {output_path.name}.")

    @staticmethod
    def _compute_spectral_coords_and_order(matrix: np.ndarray):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Projection matrix must be square.")

        n_nodes = matrix.shape[0]
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        matrix = 0.5 * (matrix + matrix.T)
        np.fill_diagonal(matrix, 0.0)

        if n_nodes == 0:
            return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=int)
        if n_nodes == 1:
            return np.array([[1.0, 0.0]], dtype=float), np.array([0], dtype=int)
        if n_nodes == 2:
            return np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=float), np.array([0, 1], dtype=int)

        degree = np.sum(matrix, axis=1)
        laplacian = np.diag(degree) - matrix
        _eigvals, eigvecs = np.linalg.eigh(laplacian)
        if eigvecs.shape[1] < 3:
            raise ValueError("Could not compute the second and third Laplacian eigenvectors.")

        coords = np.asarray(eigvecs[:, 1:3], dtype=float)
        coords -= np.mean(coords, axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(coords, axis=1))
        if scale > 0.0:
            coords /= scale

        angles = np.arctan2(coords[:, 1], coords[:, 0])
        order = np.argsort(angles)
        return coords, order

    def _gradient_template_img_and_data(self):
        results = self._last_gradients or {}
        template_img = self._active_parcellation_img
        template_data = self._active_parcellation_data
        template_path_raw = str(results.get("template_path") or "").strip()
        if template_path_raw:
            template_path = Path(template_path_raw)
            active_path = Path(self._active_parcellation_path) if self._active_parcellation_path is not None else None
            if template_img is None or active_path != template_path:
                try:
                    import nibabel as nib

                    template_img = nib.load(str(template_path))
                    template_data = np.asarray(template_img.get_fdata(), dtype=int)
                except Exception as exc:
                    raise RuntimeError(f"Failed to load parcellation template: {exc}") from exc
        if template_img is None:
            raise RuntimeError("No parcellation template available.")
        if template_data is None:
            try:
                template_data = np.asarray(template_img.get_fdata(), dtype=int)
            except Exception as exc:
                raise RuntimeError(f"Failed to read parcellation template data: {exc}") from exc
        template_data = np.asarray(template_data, dtype=int)
        if template_data.ndim != 3:
            raise RuntimeError("Parcellation template must be a 3D image.")
        return template_img, template_data

    @staticmethod
    def _projected_gradient_component_count(projected_data) -> int:
        if projected_data is None:
            return 0
        array = np.asarray(projected_data)
        if array.ndim == 3:
            return 1
        if array.ndim == 4:
            return int(array.shape[3])
        return 0

    def _ensure_projected_gradient_data(self, required_components: int):
        results = self._last_gradients or {}
        if not results:
            raise RuntimeError("No gradients are available.")

        gradients = np.asarray(results.get("gradients"), dtype=float)
        if gradients.ndim != 2 or gradients.shape[1] < 1:
            raise RuntimeError("No gradient components are available.")

        projection_labels = np.asarray(results.get("projection_labels"), dtype=int).reshape(-1)
        if projection_labels.size != gradients.shape[0]:
            raise RuntimeError("Gradient projection labels are out of sync with the node data.")

        target_count = self._normalize_gradient_surface_render_count(
            required_components,
            max_components=gradients.shape[1],
        )
        projected_data = results.get("projected_data")
        current_count = self._projected_gradient_component_count(projected_data)
        if current_count >= target_count:
            return projected_data

        template_img, template_data = self._gradient_template_img_and_data()
        results["affine"] = np.asarray(template_img.affine, dtype=float)
        try:
            results["header"] = template_img.header.copy()
        except Exception:
            results["header"] = None
        if results.get("support_mask") is None:
            results["support_mask"] = np.asarray(
                np.isin(template_data, projection_labels),
                dtype=np.float32,
            )
        projected_maps = []
        if projected_data is not None and current_count > 0:
            projected_array = np.asarray(projected_data, dtype=np.float32)
            if projected_array.ndim == 3:
                projected_maps.append(projected_array)
            elif projected_array.ndim == 4:
                projected_maps.extend(
                    [np.asarray(projected_array[..., idx], dtype=np.float32) for idx in range(projected_array.shape[3])]
                )

        for comp_idx in range(current_count, target_count):
            try:
                projected = nettools.project_to_3dspace(
                    np.asarray(gradients[:, comp_idx], dtype=float),
                    template_data,
                    projection_labels,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to project Gradient {comp_idx + 1}: {exc}") from exc
            projected_maps.append(np.asarray(projected, dtype=np.float32))

        if not projected_maps:
            results["projected_data"] = None
        elif len(projected_maps) == 1:
            results["projected_data"] = projected_maps[0]
        else:
            results["projected_data"] = np.stack(projected_maps, axis=-1)
        return results.get("projected_data")

    def _surface_render_gradient_matrix(self, required_components: int, *, use_procrustes: bool = False):
        results = self._last_gradients or {}
        if not results:
            raise RuntimeError("No gradients are available.")

        gradients = np.asarray(results.get("gradients"), dtype=float)
        if gradients.ndim != 2 or gradients.shape[1] < 1:
            raise RuntimeError("No gradient components are available.")
        target_count = self._normalize_gradient_surface_render_count(
            required_components,
            max_components=gradients.shape[1],
        )
        components = np.asarray(gradients[:, :target_count], dtype=float)
        if not use_procrustes:
            return components

        bundle = self._gradient_precomputed_bundle
        if not isinstance(bundle, dict):
            return components
        gradients_avg = bundle.get("gradients_avg")
        if gradients_avg is None:
            return components
        gradients_avg = np.asarray(gradients_avg, dtype=float)
        if gradients_avg.ndim != 2:
            return components
        if gradients_avg.shape[1] != components.shape[0]:
            return components

        try:
            from brainspace.gradient.alignment import procrustes
        except Exception as exc:
            raise RuntimeError(f"brainspace procrustes is unavailable: {exc}") from exc

        aligned = np.asarray(components, dtype=float).copy()
        max_ref_components = min(aligned.shape[1], gradients_avg.shape[0])
        for comp_idx in range(max_ref_components):
            source = np.asarray(aligned[:, comp_idx], dtype=float).reshape(-1, 1)
            target = np.asarray(gradients_avg[comp_idx], dtype=float).reshape(-1, 1)
            try:
                # Align the selected subject/session component onto the average reference.
                aligned_component = procrustes(source, target, center=False, scale=False)
            except Exception as exc:
                raise RuntimeError(f"Failed Procrustes alignment for Gradient {comp_idx + 1}: {exc}") from exc
            aligned[:, comp_idx] = np.asarray(aligned_component, dtype=float).reshape(-1)
        return aligned

    def _project_gradient_matrix_to_volume(self, gradient_matrix: np.ndarray):
        results = self._last_gradients or {}
        if not results:
            raise RuntimeError("No gradients are available.")
        components = np.asarray(gradient_matrix, dtype=float)
        if components.ndim != 2 or components.shape[1] < 1:
            raise RuntimeError("Gradient matrix must be 2D with at least one component.")

        projection_labels = np.asarray(results.get("projection_labels"), dtype=int).reshape(-1)
        if projection_labels.size != components.shape[0]:
            raise RuntimeError("Gradient projection labels are out of sync with the node data.")

        template_img, template_data = self._gradient_template_img_and_data()
        results["affine"] = np.asarray(template_img.affine, dtype=float)
        try:
            results["header"] = template_img.header.copy()
        except Exception:
            results["header"] = None
        if results.get("support_mask") is None:
            results["support_mask"] = np.asarray(
                np.isin(template_data, projection_labels),
                dtype=np.float32,
            )

        projected_maps = []
        for comp_idx in range(components.shape[1]):
            try:
                projected = nettools.project_to_3dspace(
                    np.asarray(components[:, comp_idx], dtype=float),
                    template_data,
                    projection_labels,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to project Gradient {comp_idx + 1}: {exc}") from exc
            projected_maps.append(np.asarray(projected, dtype=np.float32))

        if len(projected_maps) == 1:
            return projected_maps[0]
        return np.stack(projected_maps, axis=-1)

    def _gradient_spatial_embedding(self):
        results = self._last_gradients or {}
        if not results:
            raise RuntimeError("No gradients available for spatial classification.")

        try:
            projection_labels = np.asarray(results.get("projection_labels"), dtype=int).reshape(-1)
        except Exception as exc:
            raise RuntimeError(f"Invalid projection labels for spatial classification: {exc}") from exc
        if projection_labels.size == 0:
            raise RuntimeError("No projection labels available for spatial classification.")

        cached = results.get("spatial_embedding")
        if isinstance(cached, dict):
            cached_labels = np.asarray(cached.get("projection_labels", []), dtype=int).reshape(-1)
            coords = np.asarray(cached.get("coords"), dtype=float)
            if (
                cached_labels.shape == projection_labels.shape
                and np.array_equal(cached_labels, projection_labels)
                and coords.shape == (projection_labels.size, 2)
            ):
                return cached

        try:
            from scipy.spatial.distance import cdist
        except Exception as exc:
            raise RuntimeError(f"scipy distance tools are unavailable: {exc}") from exc

        template_img, template_data = self._gradient_template_img_and_data()
        try:
            # Spatial distances should be measured in affine/world space, not voxel index space.
            centroids_world = np.asarray(
                nettools.compute_centroids(
                    template_img,
                    labels=np.asarray(projection_labels, dtype=int),
                    world=True,
                ),
                dtype=float,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to compute world-space parcel centroids: {exc}") from exc
        if centroids_world.shape[0] != projection_labels.shape[0]:
            raise RuntimeError("Spatial centroid count does not match the projected parcels.")

        distance_matrix = np.asarray(cdist(centroids_world, centroids_world, metric="euclidean"), dtype=float)
        positive_distances = distance_matrix[distance_matrix > 0]
        if positive_distances.size == 0:
            raise RuntimeError("Spatial centroid distances are degenerate.")
        sigma = float(np.median(positive_distances))
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = float(np.mean(positive_distances))
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = 1.0
        # The spectral helper expects an affinity-like matrix. Raw distances collapse badly here.
        affinity_matrix = np.exp(-np.square(distance_matrix) / (2.0 * sigma * sigma))
        np.fill_diagonal(affinity_matrix, 0.0)
        coords, order = self._compute_spectral_coords_and_order(affinity_matrix)

        try:
            projected_x = np.asarray(
                nettools.project_to_3dspace(coords[:, 0], template_data, projection_labels),
                dtype=np.float32,
            )
            projected_y = np.asarray(
                nettools.project_to_3dspace(coords[:, 1], template_data, projection_labels),
                dtype=np.float32,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to project spatial spectral coordinates: {exc}") from exc

        cached = {
            "projection_labels": np.asarray(projection_labels, dtype=int),
            "centroids_world": centroids_world,
            "distance_matrix": distance_matrix,
            "affinity_matrix": affinity_matrix,
            "coords": np.asarray(coords, dtype=float),
            "order": np.asarray(order, dtype=int),
            "projected_x": projected_x,
            "projected_y": projected_y,
        }
        results["spatial_embedding"] = cached
        return cached

    def _classification_spatial_embedding(self, axis_gradients, *, align_to_gradients: bool = True):
        spatial = dict(self._gradient_spatial_embedding() or {})
        if not align_to_gradients:
            return spatial

        gradient_coords = np.asarray(axis_gradients, dtype=float)
        spatial_coords = np.asarray(spatial.get("coords"), dtype=float)
        projection_labels = np.asarray(spatial.get("projection_labels", []), dtype=int).reshape(-1)
        if (
            gradient_coords.ndim != 2
            or spatial_coords.ndim != 2
            or gradient_coords.shape[0] != spatial_coords.shape[0]
            or spatial_coords.shape[1] != 2
            or gradient_coords.shape[1] < 2
        ):
            return spatial

        results = self._last_gradients or {}
        cached = results.get("spatial_embedding_aligned")
        if isinstance(cached, dict):
            cached_labels = np.asarray(cached.get("projection_labels", []), dtype=int).reshape(-1)
            cached_coords = np.asarray(cached.get("coords"), dtype=float)
            if (
                cached_labels.shape == projection_labels.shape
                and np.array_equal(cached_labels, projection_labels)
                and cached_coords.shape == spatial_coords.shape
            ):
                return cached

        try:
            from brainspace.gradient.alignment import procrustes
        except Exception:
            return spatial

        target_coords = np.column_stack(
            (
                np.asarray(gradient_coords[:, 1], dtype=float),
                np.asarray(gradient_coords[:, 0], dtype=float),
            )
        )
        finite_mask = np.all(np.isfinite(spatial_coords), axis=1) & np.all(np.isfinite(target_coords), axis=1)
        if int(np.sum(finite_mask)) < 3:
            return spatial

        aligned_coords = np.asarray(spatial_coords, dtype=float).copy()
        try:
            aligned_coords[finite_mask, :] = np.asarray(
                procrustes(
                    np.asarray(spatial_coords[finite_mask, :], dtype=float),
                    np.asarray(target_coords[finite_mask, :], dtype=float),
                    center=True,
                    scale=False,
                ),
                dtype=float,
            )
        except Exception:
            return spatial

        try:
            _template_img, template_data = self._gradient_template_img_and_data()
            projected_x = np.asarray(
                nettools.project_to_3dspace(aligned_coords[:, 0], template_data, projection_labels),
                dtype=np.float32,
            )
            projected_y = np.asarray(
                nettools.project_to_3dspace(aligned_coords[:, 1], template_data, projection_labels),
                dtype=np.float32,
            )
        except Exception:
            return spatial

        aligned = {
            **spatial,
            "coords": np.asarray(aligned_coords, dtype=float),
            "projected_x": projected_x,
            "projected_y": projected_y,
            "aligned_to_gradients": True,
        }
        results["spatial_embedding_aligned"] = aligned
        return aligned

    @staticmethod
    def _classification_spatial_indices(x_axis: str, y_axis: str):
        x_norm = ConnectomeViewer._normalize_gradient_classification_axis(x_axis, default="gradient1")
        y_norm = ConnectomeViewer._normalize_gradient_classification_axis(y_axis, default="gradient1")
        if x_norm == "spatial" and y_norm == "spatial":
            return 0, 1, "Spatial 1", "Spatial 2"
        if x_norm == "spatial":
            x_index = 1 if y_norm == "gradient2" else 0
            return x_index, 0, "Spatial", "Spatial"
        if y_norm == "spatial":
            y_index = 1 if x_norm == "gradient2" else 0
            return 0, y_index, "Spatial", "Spatial"
        return 0, 0, "Spatial", "Spatial"

    def _classification_axis_payload(
        self,
        axis_key: str,
        gradients,
        projected_data,
        *,
        spatial_index: int = 0,
        spatial_label: str = "Spatial",
        spatial_embedding_override=None,
    ):
        axis = self._normalize_gradient_classification_axis(axis_key, default="gradient1")
        if axis == "gradient1":
            if gradients.ndim != 2 or gradients.shape[1] < 1:
                raise RuntimeError("Gradient 1 is not available for classification.")
            if projected_data.ndim == 4:
                volume = np.asarray(projected_data[..., 0], dtype=float)
            else:
                volume = np.asarray(projected_data, dtype=float)
            return np.asarray(gradients[:, 0], dtype=float), volume, "Gradient 1"
        if axis == "gradient2":
            if gradients.ndim != 2 or gradients.shape[1] < 2:
                raise RuntimeError("Gradient 2 is not available for classification.")
            if projected_data.ndim != 4 or projected_data.shape[3] < 2:
                raise RuntimeError("Projected Gradient 2 volume is not available.")
            return np.asarray(gradients[:, 1], dtype=float), np.asarray(projected_data[..., 1], dtype=float), "Gradient 2"

        spatial = (
            dict(spatial_embedding_override)
            if isinstance(spatial_embedding_override, dict)
            else self._gradient_spatial_embedding()
        )
        coord_index = max(0, min(int(spatial_index), 1))
        volume_key = "projected_x" if coord_index == 0 else "projected_y"
        return (
            np.asarray(spatial["coords"][:, coord_index], dtype=float),
            np.asarray(spatial[volume_key], dtype=float),
            str(spatial_label or ("Spatial 1" if coord_index == 0 else "Spatial 2")),
        )

    @staticmethod
    def _rescale_classification_axis_to_range(values, volume, target_values):
        axis_values = np.asarray(values, dtype=float)
        axis_volume = np.asarray(volume, dtype=float)
        target = np.asarray(target_values, dtype=float)

        finite_axis = axis_values[np.isfinite(axis_values)]
        finite_target = target[np.isfinite(target)]
        if finite_axis.size == 0 or finite_target.size == 0:
            return axis_values, axis_volume

        src_min = float(np.min(finite_axis))
        src_max = float(np.max(finite_axis))
        tgt_min = float(np.min(finite_target))
        tgt_max = float(np.max(finite_target))
        if not all(np.isfinite(value) for value in (src_min, src_max, tgt_min, tgt_max)):
            return axis_values, axis_volume

        scaled_values = np.array(axis_values, copy=True, dtype=float)
        scaled_volume = np.array(axis_volume, copy=True, dtype=float)
        finite_values_mask = np.isfinite(scaled_values)
        finite_volume_mask = np.isfinite(scaled_volume)

        if abs(src_max - src_min) <= 1e-12:
            midpoint = 0.5 * (tgt_min + tgt_max)
            scaled_values[finite_values_mask] = midpoint
            scaled_volume[finite_volume_mask] = midpoint
            return scaled_values, scaled_volume

        scale = (tgt_max - tgt_min) / (src_max - src_min)
        scaled_values[finite_values_mask] = (scaled_values[finite_values_mask] - src_min) * scale + tgt_min
        scaled_volume[finite_volume_mask] = (scaled_volume[finite_volume_mask] - src_min) * scale + tgt_min
        return scaled_values, scaled_volume

    def _render_gradients_3d(self) -> None:
        if not self._last_gradients:
            self.statusBar().showMessage("No gradients computed yet.")
            return
        try:
            from window.plot_gradient import GradientSurfaceDialog
        except Exception:
            try:
                from mrsi_viewer.window.plot_gradient import GradientSurfaceDialog
            except Exception as exc:
                self.statusBar().showMessage(f"Gradient surface viewer unavailable: {exc}")
                return
        try:
            available_n_grad = int(self._last_gradients.get("n_grad", 1))
            render_count = self._selected_gradient_surface_render_count()
            render_count = max(1, min(render_count, available_n_grad))
            use_procrustes = self._selected_gradient_surface_procrustes()
            if use_procrustes:
                component_matrix = self._surface_render_gradient_matrix(
                    render_count,
                    use_procrustes=True,
                )
                projected_data = self._project_gradient_matrix_to_volume(component_matrix)
            else:
                projected_data = self._ensure_projected_gradient_data(render_count)
            if projected_data is None:
                raise RuntimeError("No projected fsaverage data are available.")
            render_data = np.asarray(projected_data, dtype=float)
            if render_count == 1 and render_data.ndim == 4:
                render_data = np.asarray(render_data[..., 0], dtype=float)
            elif render_data.ndim == 4 and render_data.shape[3] > render_count:
                render_data = render_data[..., :render_count]
            source_name = self._last_gradients.get("source_name", "matrix")
            cmap_name = self._selected_surface_colormap_name()
            cmap = self._selected_surface_colormap()
            hemisphere_mode = self._selected_gradient_hemisphere_mode()
            surface_mesh = self._selected_gradient_surface_mesh()
            title = (
                f"Gradient 1 - {source_name}"
                if render_count == 1
                else f"First {render_count} Gradients - {source_name}"
            )
            if use_procrustes:
                title += " | Procrustes to gradients_avg"
            self._surface_dialog = GradientSurfaceDialog.from_array(
                render_data,
                affine=self._last_gradients.get("affine"),
                title=title,
                cmap=cmap,
                cmap_name=cmap_name,
                theme_name=self._theme_name,
                hemisphere_mode=hemisphere_mode,
                fsaverage_mesh=surface_mesh,
                parent=self,
            )
            screen = QApplication.primaryScreen()
            if screen is not None:
                geom = screen.availableGeometry()
                width = max(int(geom.width() * 0.88), 900)
                height = min(max(300 + 300 * max(render_count, 1), 420), int(geom.height() * 0.9))
            else:
                width = 1500
                height = 300 + 300 * max(render_count, 1)
            self._surface_dialog.resize(width, height)
            self._surface_dialog.show()
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to render Gradient fsaverage surfaces: {exc}")
            return
        self.statusBar().showMessage("Opened Gradient fsaverage viewer.")

    def _classify_gradients_fsaverage(self) -> None:
        if not self._last_gradients:
            self.statusBar().showMessage("No gradients computed yet.")
            return

        gradients = np.asarray(self._last_gradients.get("gradients"), dtype=float)
        keep_indices = np.asarray(self._last_gradients.get("keep_indices"), dtype=int)
        if gradients.ndim != 2 or gradients.shape[1] < 1:
            self.statusBar().showMessage("No gradient components are available for classification.")
            return
        if keep_indices.size == 0:
            self.statusBar().showMessage("No node mapping available for classification.")
            return

        if np.any((keep_indices < 0) | (keep_indices >= gradients.shape[0])):
            self.statusBar().showMessage("Classification indices are out of range. Compute gradients again.")
            return
        if not self._can_classify_gradients():
            self.statusBar().showMessage("The selected classification axes are not available for the current gradients.")
            return

        try:
            from window.plot_gradient import GradientClassificationDialog, GradientScatterDialog
        except Exception:
            try:
                from mrsi_viewer.window.plot_gradient import GradientClassificationDialog, GradientScatterDialog
            except Exception as exc:
                self.statusBar().showMessage(f"Gradient classification viewer unavailable: {exc}")
                return

        try:
            source_name = self._last_gradients.get("source_name", "matrix")
            hemisphere_mode = self._selected_gradient_classification_hemisphere_mode()
            surface_hemisphere_mode = "both" if hemisphere_mode == "separate" else hemisphere_mode
            surface_mesh = self._selected_gradient_classification_surface_mesh()
            scatter_rotation = self._selected_gradient_scatter_rotation()
            use_triangular_rgb = self._selected_gradient_triangular_rgb()
            classification_fit_mode = self._selected_gradient_classification_fit_mode()
            triangular_color_order = self._selected_gradient_triangular_color_order()
            x_axis = self._selected_gradient_classification_x_axis()
            y_axis = self._selected_gradient_classification_y_axis()
            x_spatial_index, y_spatial_index, x_spatial_label, y_spatial_label = self._classification_spatial_indices(
                x_axis,
                y_axis,
            )
            class_component = int(
                self._normalize_gradient_classification_component(
                    self._selected_gradient_classification_component(),
                    max_components=gradients.shape[1],
                )
            ) - 1
            class_component = max(0, min(class_component, gradients.shape[1] - 1))
            required_projection_count = 1
            if x_axis == "gradient2" or y_axis == "gradient2":
                required_projection_count = max(required_projection_count, 2)
            if not use_triangular_rgb:
                required_projection_count = max(required_projection_count, class_component + 1)
            projected_data = self._ensure_projected_gradient_data(required_projection_count)
            if projected_data is None:
                raise RuntimeError("No projected fsaverage data are available for classification.")
            classification_cmap_name = self._selected_gradient_classification_colormap()
            classification_cmap = self._selected_surface_colormap(classification_cmap_name)
            axis_gradients = np.asarray(gradients[keep_indices, :], dtype=float)
            spatial_embedding_override = None
            if x_axis == "spatial" or y_axis == "spatial":
                spatial_embedding_override = self._classification_spatial_embedding(
                    axis_gradients,
                    align_to_gradients=True,
                )
            x_values, x_volume, x_label = self._classification_axis_payload(
                x_axis,
                axis_gradients,
                projected_data,
                spatial_index=x_spatial_index,
                spatial_label=x_spatial_label,
                spatial_embedding_override=spatial_embedding_override,
            )
            y_values, y_volume, y_label = self._classification_axis_payload(
                y_axis,
                axis_gradients,
                projected_data,
                spatial_index=y_spatial_index,
                spatial_label=y_spatial_label,
                spatial_embedding_override=spatial_embedding_override,
            )
            if x_axis == "spatial" and y_axis in {"gradient1", "gradient2"}:
                x_values, x_volume = self._rescale_classification_axis_to_range(
                    x_values,
                    x_volume,
                    y_values,
                )
            if y_axis == "spatial" and x_axis in {"gradient1", "gradient2"}:
                y_values, y_volume = self._rescale_classification_axis_to_range(
                    y_values,
                    y_volume,
                    x_values,
                )
            class_component_values = np.asarray(axis_gradients[:, class_component], dtype=float)
            finite_mask = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(class_component_values)
            if not np.any(finite_mask):
                self.statusBar().showMessage(f"No finite values available for {y_label} vs {x_label} classification.")
                return
            projection_labels = np.asarray(self._last_gradients.get("projection_labels"), dtype=int).reshape(-1)
            if projection_labels.shape[0] != x_values.shape[0]:
                self.statusBar().showMessage(
                    "Classification labels are out of sync with the projected nodes. Compute gradients again."
                )
                return
            finite_mask &= self._gradient_projection_hemisphere_mask(
                hemisphere_mode,
                projection_labels,
            )
            if not np.any(finite_mask):
                self.statusBar().showMessage(
                    f"No {hemisphere_mode.upper()} parcels are available for {y_label} vs {x_label} classification."
                )
                return
            parcel_names = _to_string_list(self._last_gradients.get("parcel_names"))
            point_labels = []
            for idx, label in enumerate(projection_labels.tolist()):
                label_text = ""
                if parcel_names and idx < len(parcel_names):
                    label_text = str(parcel_names[idx] or "").strip()
                if not label_text:
                    label_text = f"Parcel {int(label)}"
                point_labels.append(label_text)
            point_labels = np.asarray(point_labels, dtype=object)
            hemisphere_codes_full = np.asarray(self._gradient_projection_hemisphere_codes(), dtype=int).reshape(-1)
            if hemisphere_codes_full.shape != projection_labels.shape:
                raise RuntimeError("Hemisphere membership is out of sync with the projected labels.")

            ignore_lh_name = self._selected_gradient_classification_ignore_lh_parcel()
            ignore_rh_name = self._selected_gradient_classification_ignore_rh_parcel()
            if ignore_lh_name or ignore_rh_name:
                ignore_mask = np.zeros(projection_labels.shape, dtype=bool)
                normalized_labels = np.asarray([str(text or "").strip().lower() for text in point_labels.tolist()], dtype=object)
                if ignore_lh_name:
                    target = str(ignore_lh_name).strip().lower()
                    ignore_mask |= ((hemisphere_codes_full == 0) | (hemisphere_codes_full == 2)) & (normalized_labels == target)
                if ignore_rh_name:
                    target = str(ignore_rh_name).strip().lower()
                    ignore_mask |= ((hemisphere_codes_full == 1) | (hemisphere_codes_full == 2)) & (normalized_labels == target)
                finite_mask &= ~ignore_mask
            else:
                ignore_mask = np.zeros(projection_labels.shape, dtype=bool)
            if not np.any(finite_mask):
                self.statusBar().showMessage(
                    f"No parcels remain for {y_label} vs {x_label} after applying the ignore-parcel selection."
                )
                return
            scatter_title = f"{y_label} vs {x_label} - {source_name}"
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to prepare classification axes: {exc}")
            return

        edge_pairs = np.zeros((0, 2), dtype=int)
        adjacency_note = None
        try:
            edge_pairs, adjacency_note = self._classification_scatter_edge_pairs(
                projection_labels,
                finite_mask,
            )
        except Exception as exc:
            adjacency_note = f"Adjacency skipped: {exc}"

        scatter_error = None
        surface_error = None
        try:
            scatter_projection_labels = np.asarray(projection_labels[finite_mask], dtype=int)
            scatter_point_labels = np.asarray(point_labels[finite_mask], dtype=object)
            scatter_hemisphere_codes = np.asarray(hemisphere_codes_full[finite_mask], dtype=int)
            scatter_export_metadata = {
                "source_name": source_name,
                "source_dir": str(self._last_gradients.get("source_dir", self._default_results_dir())),
                "parc_path": str(self._last_gradients.get("template_path", "") or ""),
                "template_path": str(self._last_gradients.get("template_path", "") or ""),
                "adjacency_path": str(self._gradient_classification_adjacency_path or ""),
                "gradient1_values": np.asarray(axis_gradients[:, 0], dtype=float)[finite_mask],
                "gradient2_values": (
                    np.asarray(axis_gradients[:, 1], dtype=float)[finite_mask]
                    if axis_gradients.ndim == 2 and axis_gradients.shape[1] >= 2
                    else np.full(scatter_projection_labels.shape, np.nan, dtype=float)
                ),
            }
            gradients_pair_export = np.full((scatter_projection_labels.size, 2), np.nan, dtype=float)
            if axis_gradients.ndim == 2 and axis_gradients.shape[1] >= 1:
                gradients_pair_export[:, 0] = np.asarray(axis_gradients[:, 0], dtype=float)[finite_mask]
            if axis_gradients.ndim == 2 and axis_gradients.shape[1] >= 2:
                gradients_pair_export[:, 1] = np.asarray(axis_gradients[:, 1], dtype=float)[finite_mask]
            scatter_export_metadata["gradients_pair"] = gradients_pair_export

            gradients_avg_export = np.empty((0, 0), dtype=float)
            bundle = self._gradient_precomputed_bundle
            if isinstance(bundle, dict):
                gradients_avg_raw = bundle.get("gradients_avg")
                if gradients_avg_raw is not None:
                    gradients_avg_array = np.asarray(gradients_avg_raw, dtype=float)
                    if gradients_avg_array.ndim == 2 and gradients_avg_array.shape[1] == finite_mask.shape[0]:
                        gradients_avg_export = np.asarray(gradients_avg_array[:, finite_mask], dtype=float)
            scatter_export_metadata["gradients_avg"] = gradients_avg_export

            covars_row = {}
            participant = ""
            session = ""
            group_value = ""
            modality_value = ""
            precomputed_row_index = self._last_gradients.get("precomputed_row_index", None)
            if isinstance(bundle, dict) and precomputed_row_index is not None:
                try:
                    row_index = int(precomputed_row_index)
                    participant, session = self._gradient_precomputed_row_pair(bundle, row_index)
                    bundle_rows = list(bundle.get("covars_rows") or [])
                    if 0 <= row_index < len(bundle_rows) and isinstance(bundle_rows[row_index], dict):
                        covars_row = dict(bundle_rows[row_index])
                except Exception:
                    covars_row = {}
                group_value = str(covars_row.get("group", "") or "").strip()
                modality_value = str(covars_row.get("modality", "") or "").strip()
            else:
                entry_id = self._last_gradients.get("matrix_entry_id")
                entry = self._entries.get(entry_id) if entry_id in self._entries else None
                source_path = self._gradient_entry_source_path(entry)
                if source_path is not None and source_path.exists():
                    try:
                        with np.load(source_path, allow_pickle=True) as npz:
                            group_value = self._npz_optional_scalar_text(npz, "group")
                            modality_value = self._npz_optional_scalar_text(npz, "modality")
                    except Exception:
                        pass
            scatter_export_metadata["subject_id"] = participant
            scatter_export_metadata["session_id"] = session
            scatter_export_metadata["group"] = group_value
            scatter_export_metadata["modality"] = modality_value
            scatter_export_metadata["covars_row"] = covars_row

            def _project_paths_callback(payload):
                self._project_classification_paths_to_brain(
                    payload,
                    scatter_projection_labels,
                    scatter_point_labels,
                    hemisphere_mode=hemisphere_mode,
                    source_name=source_name,
                    x_label=x_label,
                    y_label=y_label,
                )

            self._gradient_scatter_dialog = GradientScatterDialog(
                x_values[finite_mask],
                y_values[finite_mask],
                color_values=class_component_values[finite_mask],
                gradient1_values=np.asarray(axis_gradients[:, 0], dtype=float)[finite_mask],
                point_labels=point_labels[finite_mask],
                point_ids=scatter_projection_labels,
                title=scatter_title,
                x_label=x_label,
                y_label=y_label,
                color_label=f"Gradient {class_component + 1}",
                cmap=classification_cmap,
                cmap_name=classification_cmap_name,
                theme_name=self._theme_name,
                rotation_preset=scatter_rotation,
                use_triangular_rgb=use_triangular_rgb,
                rgb_fit_mode=classification_fit_mode,
                triangular_color_order=triangular_color_order,
                edge_pairs=edge_pairs,
                point_group_codes=scatter_hemisphere_codes,
                hemisphere_mode=hemisphere_mode,
                project_paths_callback=_project_paths_callback,
                export_metadata=scatter_export_metadata,
                parent=self,
            )
            self._gradient_scatter_dialog.resize(860, 760)
            self._gradient_scatter_dialog.show()
        except Exception as exc:
            scatter_error = str(exc)

        try:
            support_mask_data = self._last_gradients.get("support_mask")
            if np.any(ignore_mask):
                ignored_labels = np.asarray(projection_labels[ignore_mask], dtype=int)
                if ignored_labels.size > 0:
                    _template_img, template_data = self._gradient_template_img_and_data()
                    ignored_voxels = np.isin(np.asarray(template_data, dtype=int), ignored_labels)
                    x_volume = np.asarray(x_volume, dtype=float).copy()
                    y_volume = np.asarray(y_volume, dtype=float).copy()
                    x_volume[ignored_voxels] = np.nan
                    y_volume[ignored_voxels] = np.nan
                    if support_mask_data is not None:
                        support_mask_data = np.asarray(support_mask_data, dtype=float).copy()
                        support_mask_data[ignored_voxels] = 0.0

            if use_triangular_rgb:
                self._gradient_classification_dialog = GradientClassificationDialog.from_array(
                    x_volume,
                    y_volume,
                    affine=self._last_gradients.get("affine"),
                    x_values=x_values[finite_mask],
                    y_values=y_values[finite_mask],
                    support_mask_data=support_mask_data,
                    title=f"{y_label} vs {x_label} Classification - {source_name}",
                    x_label=x_label,
                    y_label=y_label,
                    theme_name=self._theme_name,
                    hemisphere_mode=surface_hemisphere_mode,
                    fsaverage_mesh=surface_mesh,
                    rotation_preset=scatter_rotation,
                    rgb_fit_mode=classification_fit_mode,
                    triangular_color_order=triangular_color_order,
                    parent=self,
                )
            else:
                if projected_data.ndim == 4:
                    classification_projection = np.asarray(projected_data[..., class_component], dtype=float)
                else:
                    classification_projection = np.asarray(projected_data, dtype=float)
                if np.any(ignore_mask):
                    ignored_labels = np.asarray(projection_labels[ignore_mask], dtype=int)
                    if ignored_labels.size > 0:
                        _template_img, template_data = self._gradient_template_img_and_data()
                        ignored_voxels = np.isin(np.asarray(template_data, dtype=int), ignored_labels)
                        classification_projection = np.asarray(classification_projection, dtype=float).copy()
                        classification_projection[ignored_voxels] = np.nan
                self._gradient_classification_dialog = GradientSurfaceDialog.from_array(
                    classification_projection,
                    affine=self._last_gradients.get("affine"),
                    title=f"Gradient {class_component + 1} Classification - {source_name}",
                    cmap=classification_cmap,
                    cmap_name=classification_cmap_name,
                    theme_name=self._theme_name,
                    hemisphere_mode=surface_hemisphere_mode,
                    fsaverage_mesh=surface_mesh,
                    parent=self,
                )
            screen = QApplication.primaryScreen()
            if screen is not None:
                geom = screen.availableGeometry()
                width = max(int(geom.width() * 0.9), 1200)
                height = min(max(int(geom.height() * 0.82), 760), int(geom.height() * 0.95))
            else:
                width = 1500
                height = 900
            self._gradient_classification_dialog.resize(width, height)
            self._gradient_classification_dialog.show()
        except Exception as exc:
            surface_error = str(exc)

        adjacency_suffix = ""
        if adjacency_note:
            note_text = str(adjacency_note).strip()
            if note_text:
                if note_text[-1] not in ".!?":
                    note_text += "."
                adjacency_suffix = f" {note_text}"

        if scatter_error and surface_error:
            self.statusBar().showMessage(
                f"Failed to open classification scatter and fsaverage windows: {scatter_error}; {surface_error}{adjacency_suffix}"
            )
            return
        if scatter_error:
            self.statusBar().showMessage(
                f"Opened classified fsaverage viewer, but the scatter window failed: {scatter_error}{adjacency_suffix}"
            )
            return
        if surface_error:
            self.statusBar().showMessage(
                f"Opened classification scatter, but the fsaverage window failed: {surface_error}{adjacency_suffix}"
            )
            return
        if use_triangular_rgb:
            self.statusBar().showMessage(
                f"Opened classification scatter and fsaverage viewers.{adjacency_suffix}"
            )
        else:
            self.statusBar().showMessage(
                f"Opened classification scatter and fsaverage viewers using Gradient {class_component + 1}.{adjacency_suffix}"
            )

    def _project_classification_paths_to_brain(
        self,
        payload,
        scatter_projection_labels,
        scatter_point_labels,
        *,
        hemisphere_mode="both",
        source_name="matrix",
        x_label="Gradient 2",
        y_label="Gradient 1",
    ) -> None:
        project_payload = dict(payload or {})
        group_payloads = [
            dict(group_payload)
            for group_payload in list(project_payload.get("group_paths", []))
            if (
                len(list(dict(group_payload).get("optimal_full_path", []))) >= 2
                or len(list(dict(group_payload).get("subc_optimal_path", []))) >= 2
            )
        ]
        optimal_full_path = [int(node) for node in list(project_payload.get("optimal_full_path", []))]
        if (
            not group_payloads
            and len(optimal_full_path) < 2
            and not any(len(list(dict(group).get("subc_optimal_path", []))) >= 2 for group in list(project_payload.get("group_paths", [])))
        ):
            self.statusBar().showMessage("No complete ordered path is available at the current slider radius.")
            return

        scatter_projection_labels = np.asarray(scatter_projection_labels, dtype=int).reshape(-1)
        scatter_point_labels = np.asarray(scatter_point_labels, dtype=object).reshape(-1)
        if scatter_projection_labels.size == 0:
            self.statusBar().showMessage("No classification parcels are available for 3D path projection.")
            return

        if np.any((np.asarray(optimal_full_path, dtype=int) < 0) | (np.asarray(optimal_full_path, dtype=int) >= scatter_projection_labels.size)):
            self.statusBar().showMessage("The selected scatter path is out of range for the current parcels.")
            return

        try:
            template_img, template_data = self._gradient_template_img_and_data()
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to load the parcellation template for 3D projection: {exc}")
            return

        try:
            from dipy.viz import window
            from mrsitoolbox.graphplot.netplot import NetPlot
        except Exception as exc:
            self.statusBar().showMessage(f"Fury path viewer unavailable: {exc}")
            return

        try:
            centroids_mni = np.asarray(
                nettools.compute_centroids(
                    template_img,
                    labels=np.asarray(scatter_projection_labels, dtype=int),
                    world=False,
                ),
                dtype=float,
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to compute parcel centroids for 3D projection: {exc}")
            return

        if centroids_mni.shape != (scatter_projection_labels.size, 3):
            self.statusBar().showMessage("Centroid count mismatch for 3D path projection.")
            return

        point_colors = np.asarray(project_payload.get("point_colors", []), dtype=float)
        if point_colors.shape != (scatter_projection_labels.size, 3):
            point_colors = np.full((scatter_projection_labels.size, 3), 0.35, dtype=float)

        def _path_coords(path_nodes):
            node_indices = np.asarray([int(node) for node in list(path_nodes or [])], dtype=int)
            if node_indices.size < 2:
                return None
            if np.any((node_indices < 0) | (node_indices >= centroids_mni.shape[0])):
                return None
            coords = np.asarray(centroids_mni[node_indices, :], dtype=float)
            if coords.ndim != 2 or coords.shape[1] != 3:
                return None
            if not np.all(np.isfinite(coords)):
                return None
            return coords

        def _rgb_basis_color(channel):
            mapping = {
                "R": np.asarray((1.0, 0.0, 0.0), dtype=float),
                "G": np.asarray((0.0, 1.0, 0.0), dtype=float),
                "B": np.asarray((0.0, 0.0, 1.0), dtype=float),
            }
            return np.asarray(mapping.get(str(channel).strip().upper(), (0.5, 0.5, 0.5)), dtype=float)

        def _pair_channel_color(first, second):
            pair = frozenset((str(first).strip().upper(), str(second).strip().upper()))
            mapping = {
                frozenset(("R", "B")): np.asarray((0.58, 0.28, 0.82), dtype=float),  # violet
                frozenset(("R", "G")): np.asarray((1.00, 0.55, 0.00), dtype=float),  # orange
                frozenset(("G", "B")): np.asarray((0.10, 0.76, 0.72), dtype=float),  # turquoise
            }
            if pair in mapping:
                return np.asarray(mapping[pair], dtype=float)
            return np.clip(
                0.5 * (_rgb_basis_color(first) + _rgb_basis_color(second)),
                0.0,
                1.0,
            )

        channel_order = [
            str(channel)
            for channel in str(project_payload.get("channel_order", "")).strip()
        ]
        energy_scaling = dict(project_payload.get("energy_width_scaling", {}))
        try:
            base_edge_width = max(0.05, float(project_payload.get("edge_linewidth", 0.45)))
        except Exception:
            base_edge_width = 0.45
        width_mode = str(project_payload.get("width_scaling_mode", "exp")).strip().lower()
        if width_mode not in {"exp", "linear", "log"}:
            width_mode = "exp"
        try:
            width_strength = max(0.05, float(project_payload.get("width_scaling_strength", 2.0)))
        except Exception:
            width_strength = 2.0

        def _path_width_from_energy(energy, family_type):
            scaling_value = energy_scaling.get(str(family_type), {})
            scaling = dict(scaling_value) if isinstance(scaling_value, dict) else {}
            default_width = max(1.2, base_edge_width * 6.0)
            if not scaling:
                return default_width
            try:
                energy_value = float(energy)
                emin = float(scaling.get("min"))
                emax = float(scaling.get("max"))
            except Exception:
                return default_width
            if not np.isfinite(energy_value) or not np.isfinite(emin) or not np.isfinite(emax):
                return default_width
            if np.isclose(emax, emin):
                norm = 0.5
            else:
                norm = float(np.clip((energy_value - emin) / (emax - emin), 0.0, 1.0))
            if width_mode == "linear":
                mapped = norm
            elif width_mode == "log":
                mapped = float(np.log1p(width_strength * norm) / np.log1p(width_strength))
            else:
                denominator = float(np.expm1(width_strength))
                if np.isclose(denominator, 0.0):
                    mapped = norm
                else:
                    mapped = float(np.expm1(width_strength * norm) / denominator)
            return max(default_width * 0.7, base_edge_width * (4.5 + 4.5 * mapped))

        def _lookup_path_energy(group_payload, family_key, path_nodes, fallback_key=None):
            nodes_key = tuple(int(node) for node in list(path_nodes or []))
            if len(nodes_key) < 2:
                return None
            candidate_paths = list(group_payload.get(family_key, []))
            candidate_energies = np.asarray(
                group_payload.get(
                    "ctx_path_energies" if family_key == "all_full_paths" else "subc_path_energies",
                    [],
                ),
                dtype=float,
            ).reshape(-1)
            for idx, candidate in enumerate(candidate_paths):
                candidate_key = tuple(int(node) for node in list(candidate or []))
                if candidate_key == nodes_key and idx < candidate_energies.size and np.isfinite(candidate_energies[idx]):
                    return float(candidate_energies[idx])
            if fallback_key is not None:
                try:
                    fallback = group_payload.get(fallback_key)
                    if fallback is not None and np.isfinite(float(fallback)):
                        return float(fallback)
                except Exception:
                    return None
            return None

        def _ctx_segment_records(group_payload, full_path_nodes):
            nodes = [int(node) for node in list(full_path_nodes or [])]
            anchors = {str(key): int(value) for key, value in dict(group_payload.get("anchors", {})).items()}
            order = [str(channel) for channel in channel_order if str(channel) in anchors]
            if len(nodes) < 2 or len(order) < 2:
                return []

            def _record(first, second, segment_nodes):
                color = _pair_channel_color(first, second)
                return {
                    "nodes": [int(node) for node in list(segment_nodes or [])],
                    "color": color,
                }

            if len(order) == 2:
                return [_record(order[0], order[1], nodes)]

            middle_anchor = int(anchors[order[1]])
            try:
                split_index = next(
                    idx
                    for idx, node in enumerate(nodes)
                    if int(node) == middle_anchor and 0 < idx < len(nodes) - 1
                )
            except StopIteration:
                return [_record(order[0], order[-1], nodes)]
            return [
                _record(order[0], order[1], nodes[: split_index + 1]),
                _record(order[1], order[2], nodes[split_index:]),
            ]

        draw_paths = []
        draw_colors = []
        draw_widths = []
        if not group_payloads and len(optimal_full_path) >= 2:
            group_payloads = [project_payload]

        anchor_channels = ("R", "G", "B")
        anchor_centroids = []
        anchor_node_colors = []
        anchor_node_labels = []
        seen_anchor_indices = set()
        union_nodes = set()
        show_all_paths = bool(project_payload.get("show_all_ordered_paths"))
        for group_payload in group_payloads:
            seen_paths = set()
            ctx_paths = (
                list(group_payload.get("all_full_paths", []))
                if show_all_paths
                else [group_payload.get("optimal_full_path", [])]
            )
            for path_nodes in ctx_paths:
                nodes = tuple(int(node) for node in list(path_nodes or []))
                if len(nodes) < 2:
                    continue
                width = _lookup_path_energy(
                    group_payload,
                    "all_full_paths",
                    nodes,
                    fallback_key="ctx_optimal_path_energy",
                )
                path_width = _path_width_from_energy(width, "ctx")
                for record in _ctx_segment_records(group_payload, nodes):
                    segment_nodes = tuple(int(node) for node in list(record.get("nodes", [])))
                    if len(segment_nodes) < 2:
                        continue
                    coords = _path_coords(segment_nodes)
                    if coords is None:
                        continue
                    seen_paths.add(segment_nodes)
                    draw_paths.append(coords)
                    draw_colors.append(np.asarray(record.get("color", (0.0, 0.0, 0.0)), dtype=float).reshape(3))
                    draw_widths.append(path_width)
                union_nodes.update(int(node) for node in nodes)

            subc_color = np.asarray(group_payload.get("subc_color", (0.0, 0.0, 0.0)), dtype=float).reshape(-1)
            if subc_color.shape != (3,):
                subc_color = np.asarray((0.0, 0.0, 0.0), dtype=float)
            subc_paths = list(group_payload.get("subc_paths", [])) if show_all_paths else [group_payload.get("subc_optimal_path", [])]
            for path_nodes in subc_paths:
                nodes = tuple(int(node) for node in list(path_nodes or []))
                if len(nodes) < 2 or nodes in seen_paths:
                    continue
                coords = _path_coords(nodes)
                if coords is None:
                    continue
                seen_paths.add(nodes)
                draw_paths.append(coords)
                draw_colors.append(np.asarray(subc_color, dtype=float))
                draw_widths.append(
                    _path_width_from_energy(
                        _lookup_path_energy(
                            group_payload,
                            "subc_paths",
                            nodes,
                            fallback_key="subc_optimal_path_energy",
                        ),
                        "subc",
                    )
                )
                union_nodes.update(int(node) for node in nodes)

            anchors = dict(group_payload.get("anchors", {}))
            for channel in anchor_channels:
                anchor_idx = anchors.get(channel)
                if anchor_idx is None:
                    continue
                anchor_idx = int(anchor_idx)
                if anchor_idx in seen_anchor_indices or not (0 <= anchor_idx < centroids_mni.shape[0]):
                    continue
                seen_anchor_indices.add(anchor_idx)
                anchor_centroids.append(centroids_mni[anchor_idx])
                if anchor_idx < point_colors.shape[0]:
                    anchor_node_colors.append(point_colors[anchor_idx])
                else:
                    anchor_node_colors.append((0.0, 0.0, 0.0))
                label_text = str(scatter_point_labels[anchor_idx]).strip() if anchor_idx < scatter_point_labels.shape[0] else ""
                anchor_node_labels.append(label_text or f"Parcel {int(scatter_projection_labels[anchor_idx])}")
            subc_anchor = group_payload.get("subc_anchor")
            if subc_anchor is not None:
                subc_anchor = int(subc_anchor)
                if subc_anchor not in seen_anchor_indices and 0 <= subc_anchor < centroids_mni.shape[0]:
                    seen_anchor_indices.add(subc_anchor)
                    anchor_centroids.append(centroids_mni[subc_anchor])
                    if subc_anchor < point_colors.shape[0]:
                        anchor_node_colors.append(point_colors[subc_anchor])
                    else:
                        anchor_node_colors.append((0.0, 0.0, 0.0))
                    label_text = str(scatter_point_labels[subc_anchor]).strip() if subc_anchor < scatter_point_labels.shape[0] else ""
                    anchor_node_labels.append(label_text or f"Parcel {int(scatter_projection_labels[subc_anchor])}")

        if not draw_paths:
            self.statusBar().showMessage("The selected ordered path cannot be projected to the 3D brain.")
            return

        path_node_indices = np.asarray(sorted(int(node) for node in union_nodes), dtype=int)
        if path_node_indices.size:
            path_node_indices = path_node_indices[
                (path_node_indices >= 0) & (path_node_indices < centroids_mni.shape[0])
            ]
        nonpath_mask = np.ones(centroids_mni.shape[0], dtype=bool)
        if path_node_indices.size:
            nonpath_mask[path_node_indices] = False
        nonpath_indices = np.flatnonzero(nonpath_mask)

        try:
            netplot = NetPlot(window)
            netplot.scene.background((1, 1, 1))
            normalized_hemisphere_mode = self._normalize_gradient_hemisphere_mode(hemisphere_mode)
            brain_hemi = None if normalized_hemisphere_mode in {"both", "separate"} else normalized_hemisphere_mode
            netplot.add_brain(
                netplot.mni_template,
                hemisphere=brain_hemi,
                label_image=np.asarray(template_data, dtype=int),
                parcel_labels_list=None,
                opacity=0.12,
            )
            if nonpath_indices.size:
                netplot.add_nodes(
                    centroids_mni[nonpath_indices, :],
                    node_radius=0.95,
                    node_color=np.tile(np.asarray([[0.6, 0.6, 0.6]], dtype=float), (nonpath_indices.size, 1)),
                    node_labels=None,
                    node_opacity=0.22,
                )
            if path_node_indices.size:
                netplot.add_nodes(
                    centroids_mni[path_node_indices, :],
                    node_radius=1.15,
                    node_color=np.asarray(point_colors[path_node_indices, :], dtype=float),
                    node_labels=None,
                    node_opacity=0.95,
                )
            netplot.add_paths(
                draw_paths,
                path_color=np.asarray(draw_colors, dtype=float),
                path_width=np.asarray(draw_widths, dtype=float),
                path_opacity=1.0,
            )
            if anchor_centroids:
                netplot.add_nodes(
                    np.asarray(anchor_centroids, dtype=float),
                    node_radius=2.3,
                    node_color=np.asarray(anchor_node_colors, dtype=float),
                    node_labels=anchor_node_labels,
                )
            window.show(netplot.scene)
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to project the selected path set to the 3D brain: {exc}")
            return

        path_label = " -> ".join(list(project_payload.get("channel_order", "RBG")))
        self.statusBar().showMessage(
            f"Opened 3D path projection for {path_label} on {source_name} ({y_label} vs {x_label})."
        )

    @staticmethod
    def _gradient_rotation_angles(preset: str):
        mapping = {
            "Default": (0.0, 0.0, 0.0),
            "X +90": (90.0, 0.0, 0.0),
            "X -90": (-90.0, 0.0, 0.0),
            "Y +90": (0.0, 90.0, 0.0),
            "Y -90": (0.0, -90.0, 0.0),
            "Y 180": (0.0, 180.0, 0.0),
            "Z +90": (0.0, 0.0, 90.0),
            "Z -90": (0.0, 0.0, -90.0),
        }
        return mapping.get(str(preset or "Default"), (0.0, 0.0, 0.0))

    @staticmethod
    def _actor_bounds_center(actor_obj):
        bounds = actor_obj.GetBounds() if actor_obj is not None else None
        if not bounds or len(bounds) != 6:
            return None
        if not np.all(np.isfinite(bounds)):
            return None
        return (
            float((bounds[0] + bounds[1]) * 0.5),
            float((bounds[2] + bounds[3]) * 0.5),
            float((bounds[4] + bounds[5]) * 0.5),
        )

    def _apply_gradient_component_rotation(self, actors, preset: str, fallback_center) -> None:
        rx, ry, rz = self._gradient_rotation_angles(preset)
        if np.isclose(rx, 0.0) and np.isclose(ry, 0.0) and np.isclose(rz, 0.0):
            return
        origin = None
        for actor_obj in actors:
            origin = self._actor_bounds_center(actor_obj)
            if origin is not None:
                break
        if origin is None:
            center = np.asarray(fallback_center, dtype=float).reshape(3)
            origin = (float(center[0]), float(center[1]), float(center[2]))
        for actor_obj in actors:
            if actor_obj is None:
                continue
            actor_obj.SetOrigin(*origin)
            if not np.isclose(rx, 0.0):
                actor_obj.RotateX(rx)
            if not np.isclose(ry, 0.0):
                actor_obj.RotateY(ry)
            if not np.isclose(rz, 0.0):
                actor_obj.RotateZ(rz)

    def _render_gradients_network(self) -> None:
        if not self._last_gradients:
            self.statusBar().showMessage("No gradients computed yet.")
            return

        gradients = np.asarray(self._last_gradients.get("gradients"), dtype=float)
        keep_indices = np.asarray(self._last_gradients.get("keep_indices"), dtype=int)
        projection_labels = np.asarray(self._last_gradients.get("projection_labels"), dtype=int)
        if gradients.ndim != 2 or keep_indices.size == 0 or projection_labels.size == 0:
            self.statusBar().showMessage("Gradient node data unavailable. Compute gradients again.")
            return
        if keep_indices.ndim != 1 or projection_labels.ndim != 1 or keep_indices.size != projection_labels.size:
            self.statusBar().showMessage("Gradient node mapping is invalid. Compute gradients again.")
            return
        if np.max(keep_indices) >= gradients.shape[0]:
            self.statusBar().showMessage("Gradient node indices are out of range. Compute gradients again.")
            return

        template_path_raw = str(self._last_gradients.get("template_path") or "").strip()
        template_img = self._active_parcellation_img
        if template_path_raw:
            template_path = Path(template_path_raw)
            if self._active_parcellation_path is None or Path(self._active_parcellation_path) != template_path:
                try:
                    import nibabel as nib

                    template_img = nib.load(str(template_path))
                except Exception as exc:
                    self.statusBar().showMessage(f"Failed to load parcellation for network render: {exc}")
                    return
        if template_img is None:
            self.statusBar().showMessage("No parcellation available for network render.")
            return

        try:
            from dipy.viz import window, actor
            from mrsitoolbox.graphplot.netplot import NetPlot
        except Exception as exc:
            self.statusBar().showMessage(f"Fury network viewer unavailable: {exc}")
            return

        try:
            centroids_mni = np.asarray(
                nettools.compute_centroids(
                    template_img,
                    labels=projection_labels.astype(int),
                    world=False,
                ),
                dtype=float,
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to compute parcel centroids: {exc}")
            return

        if centroids_mni.shape[0] != projection_labels.shape[0]:
            self.statusBar().showMessage("Centroid count mismatch for network render.")
            return

        cmap = self._selected_surface_colormap()
        if not callable(cmap):
            try:
                import matplotlib.cm as mpl_cm

                cmap = mpl_cm.get_cmap(str(cmap))
            except Exception as exc:
                self.statusBar().showMessage(f"Invalid colormap for network render: {exc}")
                return
        n_grad = int(self._last_gradients.get("n_grad", gradients.shape[1] if gradients.ndim == 2 else 1))
        source_name = str(self._last_gradients.get("source_name", "matrix"))
        hemisphere_mode = self._selected_gradient_hemisphere_mode()
        rotation_presets = self._current_gradient_rotation_presets()
        selected_component = self._normalize_gradient_network_component(
            self._selected_gradient_network_component(),
            max_components=n_grad,
        )
        if selected_component == "all":
            component_indices = list(range(n_grad))
        else:
            component_indices = [int(selected_component) - 1]
        opened_components = []
        netplot = NetPlot(window)
        netplot.scene.background((1, 1, 1))
        template_shape = tuple(int(v) for v in np.asarray(netplot.mni_template.shape[:3], dtype=int))
        n_panels = max(1, len(component_indices))
        n_cols = max(1, min(3, n_panels))
        n_rows = max(1, int(math.ceil(n_panels / n_cols)))
        x_step = float(template_shape[0] + 36)
        y_step = float(template_shape[1] + 48)
        title_z = float(template_shape[2] * 0.5)
        hemisphere_mask = self._gradient_projection_hemisphere_mask(
            hemisphere_mode,
            projection_labels,
        )
        brain_hemi = None if hemisphere_mode in {"both", "separate"} else hemisphere_mode

        for panel_idx, comp_idx in enumerate(component_indices):
            component_values = np.asarray(gradients[keep_indices, comp_idx], dtype=float)
            finite_mask = (
                np.isfinite(component_values)
                & np.all(np.isfinite(centroids_mni), axis=1)
                & hemisphere_mask
            )
            if not np.any(finite_mask):
                continue

            component_centroids = centroids_mni[finite_mask]
            component_values = component_values[finite_mask]
            abs_values = np.abs(component_values)
            if abs_values.size == 0:
                continue

            vmax_abs = float(np.nanmax(abs_values))
            if not np.isfinite(vmax_abs) or np.isclose(vmax_abs, 0.0):
                radii = np.full(component_values.shape[0], 3.0, dtype=float)
            else:
                radii = 2.5 + 2.5 * (abs_values / vmax_abs)

            finite_values = component_values[np.isfinite(component_values)]
            if finite_values.size == 0:
                continue
            vmin = float(np.nanmin(finite_values))
            vmax = float(np.nanmax(finite_values))
            if np.isclose(vmin, vmax):
                vmin -= 1.0
                vmax += 1.0

            colors = np.asarray(
                cmap(Normalize(vmin=vmin, vmax=vmax)(component_values))[:, :3],
                dtype=float,
            )
            row = int(panel_idx // n_cols)
            col = int(panel_idx % n_cols)
            offset = np.array((col * x_step, row * y_step, 0.0), dtype=float)
            fallback_center = (
                float(offset[0] + template_shape[0] * 0.5),
                float(offset[1] + template_shape[1] * 0.5),
                float(offset[2] + template_shape[2] * 0.5),
            )
            try:
                brain_actor = netplot.add_brain(
                    netplot.mni_template,
                    hemisphere=brain_hemi,
                    opacity=0.16,
                    offset=offset,
                )
                node_actor = netplot.add_nodes(
                    component_centroids,
                    node_radius=radii,
                    node_color=colors,
                    node_labels=None,
                    offset=offset,
                    return_actor=True,
                )
                self._apply_gradient_component_rotation(
                    (brain_actor, node_actor),
                    rotation_presets[comp_idx] if comp_idx < len(rotation_presets) else "Default",
                    fallback_center=fallback_center,
                )
                title_actor = actor.text_3d(
                    f"C{comp_idx + 1} ({rotation_presets[comp_idx] if comp_idx < len(rotation_presets) else 'Default'})",
                    position=(
                        float(offset[0] + template_shape[0] * 0.42),
                        float(offset[1] + template_shape[1] + 12.0),
                        title_z,
                    ),
                    color=(0.0, 0.0, 0.0),
                    font_size=10,
                    justification="center",
                )
                netplot.scene.add(title_actor)
            except Exception as exc:
                self.statusBar().showMessage(
                    f"Failed to build network component {comp_idx + 1}: {exc}"
                )
                return
            opened_components.append(comp_idx + 1)

        if not opened_components:
            self.statusBar().showMessage("No finite gradient nodes available for network render.")
            return
        netplot.scene.ResetCamera()
        netplot.scene.ResetCameraClippingRange()
        win_width = int(max(900, min(1800, n_cols * 420)))
        win_height = int(max(700, min(1400, n_rows * 420)))
        component_text = ", ".join(str(idx) for idx in opened_components)
        self.statusBar().showMessage(
            f"Opening Fury network viewer for components {component_text}."
        )
        QApplication.processEvents()
        try:
            window.show(
                netplot.scene,
                title=f"Gradients Network - {source_name}",
                size=(win_width, win_height),
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to render network viewer: {exc}")
            return
        self.statusBar().showMessage("Closed Fury network viewer.")

    def _clear_plot(self) -> None:
        self.figure.clear()
        self._current_matrix = None
        self._current_parcel_labels = None
        self._current_parcel_names = None
        self._current_axes = None
        self._reset_zoom_selector()
        self._matrix_full_xlim = None
        self._matrix_full_ylim = None
        self._hover_vline = None
        self._hover_hline = None
        if hasattr(self, "zoom_reset_button"):
            self.zoom_reset_button.setEnabled(False)
        self._set_plot_title("")
        self._reset_gradients_output()
        self._update_nbs_prepare_button()
        self._update_write_to_file_button()
        self._update_view_labels_button()
        self.hover_label.setText("")
        self.canvas.draw_idle()

    def _hide_hover_crosshair(self) -> bool:
        changed = False
        for line in (self._hover_vline, self._hover_hline):
            if line is not None and line.get_visible():
                line.set_visible(False)
                changed = True
        return changed

    def _update_hover_crosshair(self, x_value: float, y_value: float) -> bool:
        changed = False
        if self._hover_vline is not None:
            self._hover_vline.set_xdata([x_value, x_value])
            if not self._hover_vline.get_visible():
                self._hover_vline.set_visible(True)
            changed = True
        if self._hover_hline is not None:
            self._hover_hline.set_ydata([y_value, y_value])
            if not self._hover_hline.get_visible():
                self._hover_hline.set_visible(True)
            changed = True
        return changed

    def _on_hover(self, event) -> None:
        if (
            self._current_matrix is None
            or self._current_axes is None
            or event.inaxes != self._current_axes
            or event.xdata is None
            or event.ydata is None
        ):
            if self._hide_hover_crosshair():
                self.canvas.draw_idle()
            self.hover_label.setText("")
            return
        row = int(event.ydata)
        col = int(event.xdata)
        if (
            row < 0
            or col < 0
            or row >= self._current_matrix.shape[0]
            or col >= self._current_matrix.shape[1]
        ):
            if self._hide_hover_crosshair():
                self.canvas.draw_idle()
            self.hover_label.setText("")
            return
        self._update_hover_crosshair(event.xdata, event.ydata)
        self.canvas.draw_idle()
        row_label = str(row)
        col_label = str(col)
        if self._current_parcel_labels and row < len(self._current_parcel_labels):
            row_label = self._current_parcel_labels[row]
        if self._current_parcel_labels and col < len(self._current_parcel_labels):
            col_label = self._current_parcel_labels[col]
        row_name = ""
        col_name = ""
        if self._current_parcel_names and row < len(self._current_parcel_names):
            row_name = self._current_parcel_names[row]
        if self._current_parcel_names and col < len(self._current_parcel_names):
            col_name = self._current_parcel_names[col]
        value_text = _format_value(self._current_matrix[row, col])
        parts = [f"Row {row}: {row_label}", f"Col {col}: {col_label}", f"Value: {value_text}"]
        if row_name or col_name:
            parts.append(f"Names: {row_name} | {col_name}")
        self.hover_label.setText("  ".join(parts))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_plot_title_label()

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if not url.isLocalFile():
                    continue
                local_path = Path(url.toLocalFile())
                if local_path.is_dir() or local_path.suffix.lower() == ".npz":
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        file_paths = []
        folder_paths = []
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            local_path = Path(url.toLocalFile())
            if local_path.is_dir():
                folder_paths.append(local_path)
            else:
                file_paths.append(str(local_path))

        if file_paths:
            self._add_files(file_paths)
        for folder_path in folder_paths:
            self._open_batch_import_dialog(folder_path)
        if not file_paths and not folder_paths:
            event.ignore()
            return
        event.acceptProposedAction()


def main() -> int:
    diag_path = _install_runtime_diagnostics()
    app = QApplication(sys.argv)
    _write_diagnostic_line("QApplication created")
    try:
        app.aboutToQuit.connect(lambda: _write_diagnostic_line("QApplication.aboutToQuit emitted"))
    except Exception:
        pass
    if sys.platform.startswith("linux"):
        app.setApplicationName("donald")
        if hasattr(app, "setDesktopFileName"):
            app.setDesktopFileName("donald")
    elif sys.platform == "darwin":
        app.setApplicationName("Donald")
        if hasattr(app, "setApplicationDisplayName"):
            app.setApplicationDisplayName("Donald")
    app_icon = _load_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    splash_candidates = [
        ROOTDIR / "assets" / "splash.png",
        ROOTDIR / "icons" / "conviewer_2.png",
        ROOTDIR / "icons" / "conviewer.png",
    ]
    splash_pix = QPixmap()
    for candidate in splash_candidates:
        if candidate.exists():
            splash_pix = QPixmap(str(candidate))
            if not splash_pix.isNull():
                break

    splash = None
    if not splash_pix.isNull():
        if hasattr(Qt, "WindowType"):
            splash_hint = Qt.WindowType.WindowStaysOnTopHint
        else:
            splash_hint = Qt.WindowStaysOnTopHint
        if hasattr(Qt, "AlignmentFlag"):
            splash_align = Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter
        else:
            splash_align = Qt.AlignBottom | Qt.AlignHCenter
        splash = QSplashScreen(splash_pix, splash_hint)
        splash.show()
        app.processEvents()
        splash.showMessage("Loading resources...", splash_align, QColor("white"))
        app.processEvents()
        splash.showMessage("Building UI...", splash_align, QColor("white"))
        app.processEvents()
        splash_deadline = time.monotonic() + 3.0
        while time.monotonic() < splash_deadline:
            app.processEvents()
            time.sleep(0.02)

    window = ConnectomeViewer()
    _write_diagnostic_line(f"DONALD constructed: {window!r}")
    try:
        window.destroyed.connect(lambda *_args: _write_diagnostic_line("DONALD destroyed"))
    except Exception:
        pass
    if not str(window._results_dir_default or "").strip() and splash is not None:
        splash.close()
        splash = None
    if not window.ensure_initial_configuration():
        _write_diagnostic_line("Initial configuration canceled")
        if splash is not None:
            splash.close()
        window.close()
        return 0
    if splash is not None:
        splash.showMessage("Starting Donald...", splash_align, QColor("white"))
        app.processEvents()
    window.showMaximized()
    _write_diagnostic_line(f"Viewer shown; diagnostics logging to {diag_path}")
    if splash is not None:
        splash.finish(window)
    exit_code = app.exec()
    _write_diagnostic_line(f"QApplication exited with code {exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
