"""Per-dataset EDA pages for the dataset cards (docs/assets/eda/<name>.html).

One compact figure per dataset:
- up to 3 sample test channels (downsampled for very long series) with ground-truth
  anomaly shading,
- an anomaly event-length histogram,
- a small stats table (dims, lengths, anomaly rate, events).

Generated only for datasets present under the local data/ directory — cards for
restricted or undownloaded datasets keep their textual description only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tsad_forge.data.schema import TSADDataset, label_events
from tsad_forge.viz.charts import _INK, _INK2, _base_layout

_SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100"]

# dataset name -> (loader kwargs, human note shown under the title)
EDA_TARGETS: dict[str, dict] = {
    "synthetic": {},
    "smd": {"machine": "machine-1-1"},
    "psm": {},
    "skab": {"experiment": "valve1/0"},
    "nab": {"rel_path": "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"},
    "mgab": {"series": 1},
    "mba": {},
}


def _downsample(x: np.ndarray, max_points: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Stride-downsample for plotting; returns (index, values)."""
    idx = np.arange(len(x))
    if len(x) <= max_points:
        return idx, x
    stride = len(x) // max_points + 1
    return idx[::stride], x[::stride]


def dataset_eda(ds: TSADDataset, out_dir: Path, name: str) -> Path:
    events = label_events(ds.labels)
    lengths = [e - s for s, e in events]
    n_channels = min(ds.n_dims, 3)

    fig = make_subplots(
        rows=n_channels + 1,
        cols=1,
        shared_xaxes=False,
        row_heights=[0.8 / n_channels] * n_channels + [0.2],
        vertical_spacing=0.30 / (n_channels + 1),
        subplot_titles=[f"test channel {d}" for d in range(n_channels)]
        + ["anomaly event length distribution (steps)"],
    )
    for d in range(n_channels):
        xi, yi = _downsample(ds.test[:, d])
        fig.add_trace(
            go.Scatter(
                x=xi,
                y=yi,
                name=f"channel {d}",
                showlegend=False,
                line={"color": _SERIES_COLORS[d % 3], "width": 1},
            ),
            row=d + 1,
            col=1,
        )
        for s, e in events:
            fig.add_vrect(
                x0=s,
                x1=max(e, s + max(len(ds.test) // 1000, 1)),
                fillcolor="rgba(208,59,59,0.14)",
                line_width=0,
                row=d + 1,
                col=1,
            )
    if len(lengths) >= 5 and len(set(lengths)) == 1:
        # all events share one length — a histogram degenerates into a single
        # full-width block with fractional ticks; a count bar reads better
        fig.add_trace(
            go.Bar(
                x=[f"{lengths[0]} steps"],
                y=[len(lengths)],
                marker={"color": "#2a78d6"},
                showlegend=False,
                width=0.3,
            ),
            row=n_channels + 1,
            col=1,
        )
        fig.update_yaxes(title_text="event count", row=n_channels + 1, col=1)
    elif len(lengths) >= 5:
        fig.add_trace(
            go.Histogram(
                x=lengths,
                nbinsx=min(30, len(lengths)),
                marker={"color": "#2a78d6"},
                showlegend=False,
            ),
            row=n_channels + 1,
            col=1,
        )
    elif lengths:  # a histogram of 1-4 events is unreadable — show one bar per event
        fig.add_trace(
            go.Bar(
                x=[f"event {i + 1}" for i in range(len(lengths))],
                y=lengths,
                marker={"color": "#2a78d6"},
                showlegend=False,
                width=0.4,
            ),
            row=n_channels + 1,
            col=1,
        )
        fig.update_yaxes(title_text="length (steps)", row=n_channels + 1, col=1)
    stats = (
        f"dims={ds.n_dims} · train={len(ds.train):,} steps · test={len(ds.test):,} steps · "
        f"anomaly rate={ds.anomaly_rate:.3f} · events={len(events)}"
        + (
            f" · event length min/median/max = {min(lengths)}/{int(np.median(lengths))}/{max(lengths)}"
            if lengths
            else ""
        )
    )
    height = 220 * n_channels + 280
    _base_layout(
        fig, f"EDA — {ds.meta.get('name', name)} (red shading = labeled anomalies)", height=height
    )
    fig.update_annotations(font={"size": 12, "color": _INK})
    fig.add_annotation(  # stats line just under the main title, above the subplots
        text=stats,
        xref="paper",
        yref="paper",
        x=0,
        y=1.0 + 24 / height,
        yanchor="bottom",
        showarrow=False,
        font={"size": 11, "color": _INK2},
        align="left",
    )
    fig.update_layout(margin={"l": 70, "r": 40, "t": 96, "b": 60})

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.html"
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


def generate_eda(
    out_dir: str | Path = "docs/assets/eda", data_dir: str | Path = "data"
) -> list[Path]:
    from tsad_forge.data.registry import load_dataset

    out_dir = Path(out_dir)
    paths = []
    for name, kw in EDA_TARGETS.items():
        kwargs = dict(kw)
        if name != "synthetic":
            kwargs["data_dir"] = data_dir
        try:
            ds = load_dataset(name, **kwargs)
        except (FileNotFoundError, KeyError):
            print(f"  [eda] skip {name}: data not present locally")
            continue
        paths.append(dataset_eda(ds, out_dir, name))
    print(f"generated {len(paths)} EDA pages under {out_dir}")
    return paths
