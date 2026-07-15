from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.datasets.landmark_utils import (
    associate_landmarks_to_vertices,
    build_landmark_tooth_records,
    compute_tooth_centers,
    landmarks_to_nested_dict,
    load_landmarks,
    remap_landmarks_to_sample,
)
from src.datasets.labels import (
    ARCH_CLASS_TO_FDI,
    CLASS_TO_FDI,
    FDI_TO_ARCH_CLASS,
    FDI_TO_CLASS,
    map_fdi_to_arch_class,
    map_fdi_to_class,
)
from src.preprocessing.normalize import (
    normalize_landmarks,
    normalize_points,
    normalize_tooth_centers,
)
from src.preprocessing.normals import compute_vertex_normals, normalize_vectors
from src.preprocessing.sampling import sample_vertex_indices
from src.utils.io import load_mesh, load_teeth3ds_annotation


@dataclass(frozen=True)
class RawScanPaths:
    scan_id: str
    patient_id: str
    jaw: str
    mesh_path: Path
    annotation_path: Path | None = None
    landmark_path: Path | None = None


def discover_raw_scans(
    raw_root: str | Path, landmark_root: str | Path | None = None
) -> list[RawScanPaths]:
    raw_root = Path(raw_root)
    landmark_root = Path(landmark_root) if landmark_root else None
    meshes = sorted(raw_root.rglob("*.obj"))

    records = []
    for mesh_path in meshes:
        scan_id = _scan_id_from_mesh(mesh_path)
        jaw = _infer_jaw(mesh_path)
        patient_id = _infer_patient_id(scan_id, jaw)
        annotation_path = mesh_path.with_suffix(".json")
        if not annotation_path.is_file():
            annotation_path = None
        landmark_path = _find_landmark_path(landmark_root, jaw, patient_id, scan_id)

        records.append(
            RawScanPaths(
                scan_id=scan_id,
                patient_id=patient_id,
                jaw=jaw,
                mesh_path=mesh_path,
                annotation_path=annotation_path,
                landmark_path=landmark_path,
            )
        )
    return records


def load_raw_scan(paths: RawScanPaths) -> dict[str, Any]:
    mesh = load_mesh(paths.mesh_path)
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    normals = mesh["normals"]
    normal_source = "obj"
    if normals is None or normals.shape != vertices.shape:
        normals = compute_vertex_normals(vertices, faces)
        normal_source = "computed"
    else:
        normals = normalize_vectors(normals)

    labels, instances = _load_labels_and_instances(paths)
    if labels is not None:
        labels = _validate_annotation_length(
            labels, len(vertices), "labels", paths.scan_id
        )
    if instances is not None:
        instances = _validate_annotation_length(
            instances, len(vertices), "instances", paths.scan_id
        )
    _validate_tooth_annotation_consistency(labels, instances, paths.scan_id)

    landmarks = load_landmarks(paths.landmark_path)
    if landmarks:
        landmarks = associate_landmarks_to_vertices(
            landmarks, vertices, labels, instances
        )

    return {
        "scan_id": paths.scan_id,
        "patient_id": paths.patient_id,
        "jaw": paths.jaw,
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "normal_source": normal_source,
        "labels": labels,
        "instances": instances,
        "landmarks": landmarks,
        "paths": paths,
    }


def build_processed_sample(
    raw_scan: dict[str, Any],
    num_points: int | None = 60000,
    seed: int | None = None,
    require_labels: bool = True,
) -> dict[str, Any]:
    vertices = raw_scan["vertices"]
    normals = raw_scan["normals"]
    labels = raw_scan["labels"]
    instances = raw_scan["instances"]

    if labels is None:
        if require_labels:
            raise ValueError(f"Missing labels for scan {raw_scan['scan_id']}")
    elif instances is None:
        raise ValueError(f"Missing instances for scan {raw_scan['scan_id']}")

    pos_norm_all, center, scale = normalize_points(vertices)
    tooth_centers_raw = (
        compute_tooth_centers(vertices, labels) if labels is not None else {}
    )
    tooth_centers_norm = normalize_tooth_centers(tooth_centers_raw, center, scale)

    source_indices = sample_vertex_indices(
        len(vertices),
        num_points,
        seed=seed,
        points=pos_norm_all,
    )

    pos_raw = vertices[source_indices].astype(np.float32)
    pos = pos_norm_all[source_indices].astype(np.float32)
    normal = normals[source_indices].astype(np.float32)
    y_fdi = labels[source_indices].astype(np.int64) if labels is not None else None
    y_binary = (y_fdi > 0).astype(np.int64) if y_fdi is not None else None
    y_fdi_class = map_fdi_to_class(y_fdi) if y_fdi is not None else None
    y_arch_class = map_fdi_to_arch_class(y_fdi) if y_fdi is not None else None
    y_instance = (
        instances[source_indices].astype(np.int64) if instances is not None else None
    )

    landmarks = raw_scan.get("landmarks") or []
    landmarks = normalize_landmarks(landmarks, center, scale)
    landmarks = remap_landmarks_to_sample(landmarks, source_indices)
    landmarks_raw = landmarks_to_nested_dict(landmarks, coord_key="coord")
    landmarks_norm = landmarks_to_nested_dict(
        landmarks,
        coord_key="coord_norm",
        distance_key="nearest_distance_norm",
        distance_output_key="nearest_distance_norm",
    )
    landmark_to_tooth = build_landmark_tooth_records(landmarks)

    return {
        "scan_id": raw_scan["scan_id"],
        "patient_id": raw_scan["patient_id"],
        "jaw": raw_scan["jaw"],
        "pos_raw": pos_raw,
        "pos": pos,
        "normal": normal,
        "normal_source": raw_scan["normal_source"],
        "y_binary": y_binary,
        "y_fdi": y_fdi,
        "y_fdi_class": y_fdi_class,
        "y_arch_class": y_arch_class,
        "y_instance": y_instance,
        "fdi_to_class": dict(FDI_TO_CLASS),
        "class_to_fdi": dict(CLASS_TO_FDI),
        "fdi_to_arch_class": dict(FDI_TO_ARCH_CLASS),
        "arch_class_to_fdi": dict(ARCH_CLASS_TO_FDI[raw_scan["jaw"]]),
        "source_indices": source_indices.astype(np.int64),
        "landmarks_raw": landmarks_raw,
        "landmarks_norm": landmarks_norm,
        "landmark_to_tooth": landmark_to_tooth,
        "tooth_centers_raw": tooth_centers_raw,
        "tooth_centers_norm": tooth_centers_norm,
        "center": center.astype(np.float32),
        "scale": float(scale),
    }


def _load_labels_and_instances(
    paths: RawScanPaths,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if paths.annotation_path is None:
        return None, None

    annotation = load_teeth3ds_annotation(paths.annotation_path)
    if annotation["id_patient"] != paths.patient_id:
        raise ValueError(
            f"{paths.scan_id}: annotation id_patient={annotation['id_patient']} "
            f"does not match patient_id={paths.patient_id}"
        )
    if annotation["jaw"] != paths.jaw:
        raise ValueError(
            f"{paths.scan_id}: annotation jaw={annotation['jaw']} does not match path jaw={paths.jaw}"
        )

    return annotation["labels"], annotation["instances"]


def _validate_annotation_length(
    values: np.ndarray, expected: int, name: str, scan_id: str
) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if len(values) != expected:
        raise ValueError(
            f"{scan_id}: {name} length {len(values)} does not match vertices {expected}"
        )
    return values.astype(np.int64)


def _validate_tooth_annotation_consistency(
    labels: np.ndarray | None,
    instances: np.ndarray | None,
    scan_id: str,
) -> None:
    if labels is None and instances is None:
        return
    if labels is None or instances is None:
        raise ValueError(f"{scan_id}: labels and instances must be present together")

    fdi_to_instances: dict[int, set[int]] = {}
    instance_to_fdis: dict[int, set[int]] = {}
    mixed_zero_pairs: set[tuple[int, int]] = set()
    for fdi, inst in zip(labels.tolist(), instances.tolist()):
        fdi = int(fdi)
        inst = int(inst)
        if (fdi <= 0) != (inst <= 0):
            mixed_zero_pairs.add((fdi, inst))
            continue
        if fdi <= 0:
            continue
        fdi_to_instances.setdefault(fdi, set()).add(inst)
        instance_to_fdis.setdefault(inst, set()).add(fdi)

    fdi_conflicts = {
        fdi: sorted(values)
        for fdi, values in fdi_to_instances.items()
        if len(values) > 1
    }
    instance_conflicts = {
        inst: sorted(values)
        for inst, values in instance_to_fdis.items()
        if len(values) > 1
    }
    if fdi_conflicts or instance_conflicts or mixed_zero_pairs:
        parts = []
        if fdi_conflicts:
            parts.append(f"fdi_to_instances={fdi_conflicts}")
        if instance_conflicts:
            parts.append(f"instance_to_fdis={instance_conflicts}")
        if mixed_zero_pairs:
            parts.append(f"mixed_zero_pairs={sorted(mixed_zero_pairs)}")
        raise ValueError(
            f"{scan_id}: inconsistent tooth annotations ({'; '.join(parts)})"
        )


def _scan_id_from_mesh(mesh_path: Path) -> str:
    return mesh_path.stem


def _infer_jaw(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    stem = path.stem.lower()
    folder_jaw = (
        "upper" if "upper" in parts else "lower" if "lower" in parts else "unknown"
    )
    file_jaw = (
        "upper"
        if stem.endswith("_upper")
        else "lower"
        if stem.endswith("_lower")
        else "unknown"
    )

    if folder_jaw != "unknown" and file_jaw != "unknown" and folder_jaw != file_jaw:
        raise ValueError(
            f"Jaw conflict for {path}: folder={folder_jaw}, file={file_jaw}"
        )
    if file_jaw != "unknown":
        return file_jaw
    return folder_jaw


def _infer_patient_id(scan_id: str, jaw: str) -> str:
    text = scan_id
    for suffix in ("_upper", "_lower"):
        if text.lower().endswith(suffix):
            return text[: -len(suffix)]
    return text


def _find_landmark_path(
    landmark_root: Path | None, jaw: str, patient_id: str, scan_id: str
) -> Path | None:
    if landmark_root is None or jaw == "unknown":
        return None
    candidate = landmark_root / jaw / patient_id / f"{scan_id}__kpt.json"
    if candidate.is_file():
        return candidate
    return None
