"""Interpret the provenance fields saved in LK recording metadata."""

BACKEND_SOURCES = {
    "brainflow": "hardware", "lsl": "lsl", "replay": "replay",
    "synthetic": "synthetic", "synthetic_mi": "synthetic",
}
MODEL_DATA_ORIGINS = {"hardware": "hardware_recorded", "lsl": "lsl_recorded", "synthetic": "synthetic"}


def recording_source(metadata: dict) -> str:
    configuration = metadata.get("configuration", {})
    provenance = metadata.get("provenance", {})
    backends = (configuration.get("backend"), provenance.get("backend"))
    sources = {
        value for value in (
            provenance.get("source_kind"), provenance.get("kind"), metadata.get("source_kind"),
            *(BACKEND_SOURCES.get(backend) for backend in backends),
        ) if value and value != "unspecified"
    }
    if len(sources) > 1:
        raise ValueError(f"Conflicting source provenance: {', '.join(sorted(sources))}")
    source = next(iter(sources), "unspecified")
    if source not in {*MODEL_DATA_ORIGINS, "replay", "unspecified"}:
        raise ValueError(f"Unknown source provenance: {source!r}")
    if source in {"hardware", "lsl"} and "scripted_simulation" in (
        provenance.get("label_source"), configuration.get("label_source"),
    ):
        raise ValueError("Recording has simulated labels marked as hardware acquisition")
    return source


def recording_task(metadata: dict) -> str:
    configuration = metadata.get("configuration", {})
    tasks = {task for task in (configuration.get("task"), metadata.get("task"))
             if task and task != "unspecified"}
    if len(tasks) > 1:
        raise ValueError(f"Conflicting recording tasks: {', '.join(sorted(tasks))}")
    return next(iter(tasks), "unspecified")


def require_motor_imagery(metadata: dict) -> None:
    task = recording_task(metadata)
    if task == "motor_imagery":
        return
    if task != "unspecified":
        raise ValueError(f"MI training requires motor_imagery, got {task!r}")
    # The original collect_lk_mi CLI saved its imagery window, but no task.
    # Synthetic MI fixtures written directly have only the specialized backend.
    configuration = metadata.get("configuration", {})
    provenance = metadata.get("provenance", {})
    imagery_seconds = configuration.get("imagery_seconds")
    if "synthetic_mi" in (configuration.get("backend"), provenance.get("backend")) or (
        type(imagery_seconds) in (float, int) and 0 < imagery_seconds < float("inf")
    ):
        return
    raise ValueError("Recording task is unspecified and no LK motor imagery protocol is recorded")
