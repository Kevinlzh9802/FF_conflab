import sys
import os
import re
from datetime import datetime

# Require tqdm for clean multi-line progress display
try:
    from tqdm.auto import tqdm as _tqdm
except Exception as e:  # pragma: no cover
    raise ImportError(
        "tqdm is required for display_frame_results. Install with: pip install tqdm"
    ) from e

# Simple tee that mirrors writes to both the original stream and a log file.
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

class _TeeStream:
    def __init__(self, stream, fh, strip_ansi: bool = True, map_cr_to_nl: bool = True):
        self._stream = stream
        self._fh = fh
        self._strip_ansi = strip_ansi
        self._map_cr_to_nl = map_cr_to_nl

    def write(self, data):
        if not isinstance(data, str):
            data = str(data)
        # Write to terminal
        self._stream.write(data)
        # Normalize control chars and strip ANSI for file
        out = data
        plain = ANSI_RE.sub("", out) if self._strip_ansi else out
        # Skip logging writes that are only carriage returns (cursor moves)
        if plain and set(plain) <= {"\r"}:
            self.flush()
            return
        # TODO: still some blank lines in logs
        if self._map_cr_to_nl:
            out = out.replace("\r\n", "\n").replace("\r", "\n")
        if self._strip_ansi:
            out = ANSI_RE.sub("", out)
        self._fh.write(out)
        # Flush both to keep logs up to date
        self.flush()

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._fh.flush()
        except Exception:
            pass

_ORIG_STDOUT = None
_ORIG_STDERR = None
_LOG_FH = None

def start_logging(log_path: str) -> str:
    """Start tee logging of stdout/stderr to the given file.

    - Creates parent directories as needed.
    - Replaces sys.stdout and sys.stderr with tee streams that also write to file.
    - Normalizes carriage returns and strips ANSI for the file so tqdm output is readable.
    """
    global _ORIG_STDOUT, _ORIG_STDERR, _LOG_FH
    if _LOG_FH is not None:
        return log_path
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    _LOG_FH = open(log_path, "a", encoding="utf-8")
    _ORIG_STDOUT = sys.stdout
    _ORIG_STDERR = sys.stderr
    sys.stdout = _TeeStream(_ORIG_STDOUT, _LOG_FH, strip_ansi=True, map_cr_to_nl=True)
    sys.stderr = _TeeStream(_ORIG_STDERR, _LOG_FH, strip_ansi=True, map_cr_to_nl=True)
    return log_path

def stop_logging():
    """Restore stdout/stderr and close the log file."""
    global _ORIG_STDOUT, _ORIG_STDERR, _LOG_FH
    if _LOG_FH is None:
        return
    if _ORIG_STDOUT is not None:
        sys.stdout = _ORIG_STDOUT
    if _ORIG_STDERR is not None:
        sys.stderr = _ORIG_STDERR
    try:
        _LOG_FH.flush()
        _LOG_FH.close()
    finally:
        _LOG_FH = None
        _ORIG_STDOUT = None
        _ORIG_STDERR = None


def log_only(text: str, end: str = "\n") -> None:
    """Write text only to the log file (not to the terminal).

    Requires start_logging() to have been called. Silently no-ops if not.
    Normalizes CRLF to LF and strips ANSI sequences.
    """
    global _LOG_FH
    if _LOG_FH is None:
        return
    if not isinstance(text, str):
        text = str(text)
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = ANSI_RE.sub("", out)
    _LOG_FH.write(out)
    if end:
        _LOG_FH.write(end)
    _LOG_FH.flush()

def display_frame_results(idx_frame: int, total_frames: int, groups, GTgroups) -> None:
    """Render a 3-line, in-place status that continuously updates using tqdm.

    Line 1: Frame n/N with tqdm progress bar
    Line 2: FOUND: [found groups]
    Line 3: GT:    [GT groups]
    """
    found_txt = " |".join(str(g) for g in (groups or [])) or "No Groups"
    gt_txt = " |".join(str(g) for g in (GTgroups or [])) or "No Groups"

    # tqdm stacked bars (positions 0..2)
    if not hasattr(display_frame_results, "_bars"):
        # Initialize three bars on first call
        p0 = _tqdm(total=total_frames, position=0, dynamic_ncols=True, file=sys.stdout,
                   leave=True, bar_format="Frame {n_fmt}/{total_fmt} {bar} {percentage:3.0f}%")
        p1 = _tqdm(total=1, position=1, dynamic_ncols=True, leave=True, file=sys.stdout,
                   bar_format="FOUND: {desc}")
        p2 = _tqdm(total=1, position=2, dynamic_ncols=True, leave=True, file=sys.stdout,
                   bar_format="GT:    {desc}")
        display_frame_results._bars = (p0, p1, p2)  # type: ignore[attr-defined]

    p0, p1, p2 = display_frame_results._bars  # type: ignore[attr-defined]

    # Update progress and lines
    p0.n = min(idx_frame, total_frames)
    p0.refresh()
    # Avoid double refresh: set descriptions without auto-refresh, then refresh once
    p1.set_description_str(found_txt, refresh=False)
    p2.set_description_str(gt_txt, refresh=False)
    p1.refresh()
    p2.refresh()

    if idx_frame >= total_frames:
        # Close bars and clear handle for next run
        # p2.close(); p1.close(); p0.close()
        delattr(display_frame_results, "_bars")  # type: ignore[attr-defined]

def get_log_path(config):
    # Resolve log path: if a .log file path is given, use its parent as directory.
    # The actual file name is GCFF-<start_time>.log with Windows-safe characters.
    start_dt = datetime.now()
    ts_display = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    ts_fname = start_dt.strftime("%Y-%m-%d_%H-%M-%S")  # replace ':' to be Windows-safe
    log_arg = config.paths.get('log', "./experiments/logs/")
    log_path_is_file = str(log_arg).lower().endswith('.log')
    log_dir = os.path.dirname(log_arg) if log_path_is_file else log_arg
    if not log_dir:
        log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"GCFF_{ts_fname}.log")

    return ts_display, log_path