# Require tqdm for clean multi-line progress display
try:
    from tqdm.auto import tqdm as _tqdm
except Exception as e:  # pragma: no cover
    raise ImportError(
        "tqdm is required for display_frame_results. Install with: pip install tqdm"
    ) from e

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
        p0 = _tqdm(total=total_frames, position=0, dynamic_ncols=True,
                   leave=True, bar_format="Frame {n_fmt}/{total_fmt} {bar} {percentage:3.0f}%")
        p1 = _tqdm(total=1, position=1, dynamic_ncols=True, leave=True,
                   bar_format="FOUND: {desc}")
        p2 = _tqdm(total=1, position=2, dynamic_ncols=True, leave=True,
                   bar_format="GT:    {desc}")
        display_frame_results._bars = (p0, p1, p2)  # type: ignore[attr-defined]

    p0, p1, p2 = display_frame_results._bars  # type: ignore[attr-defined]

    # Update progress and lines
    p0.n = min(idx_frame, total_frames)
    p0.refresh()
    p1.set_description_str(found_txt)
    p1.refresh()
    p2.set_description_str(gt_txt)
    p2.refresh()

    if idx_frame >= total_frames:
        # p2.close(); p1.close(); p0.close()
        delattr(display_frame_results, "_bars")  # type: ignore[attr-defined]
