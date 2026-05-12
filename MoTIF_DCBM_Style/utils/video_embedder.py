from __future__ import annotations

from pathlib import Path

from ablation_toolkit.utils._reexport import load_module_from_path, reexport_public


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "VideoCBM" / "AgentMoTIF" / "utils" / "video_embedder.py"
MODULE = load_module_from_path("ablation_toolkit_upstream_video_embedder", SOURCE)
reexport_public(MODULE, globals())
