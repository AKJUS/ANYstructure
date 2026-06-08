"""Infer flat stiffened plate fields from CalculiX/PrePoMax shell models.

This module is intentionally small and geometry-first.  The first supported
workflow reads the CalculiX ``.inp`` shell mesh, groups coplanar connected shell
elements into surface patches, and infers flat-panel bays plus shell-plate
stiffener webs/flanges.  ``.frd`` support is currently limited to lightweight
result-block discovery so the geometry interpreter can be paired with result
files without committing to stress recovery semantics yet.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


Point3D = tuple[float, float, float]
Vector3D = tuple[float, float, float]


@dataclass(frozen=True)
class ShellElement:
    """One shell element from a CalculiX input deck."""

    element_id: int
    node_ids: tuple[int, ...]
    element_type: str = ""
    elset: str | None = None

    @property
    def corner_node_ids(self) -> tuple[int, ...]:
        element_type = self.element_type.upper()
        if element_type.startswith("S8") and len(self.node_ids) >= 4:
            return self.node_ids[:4]
        if element_type.startswith("S6") and len(self.node_ids) >= 3:
            return self.node_ids[:3]
        return self.node_ids[:4]


@dataclass(frozen=True)
class ShellSection:
    """Shell-section metadata from ``*Shell section`` cards."""

    elset: str | None
    material: str | None
    thickness_m: float | None
    offset: str | None = None


@dataclass(frozen=True)
class FeShellModel:
    """Parsed shell model used by the plate-field interpreter."""

    nodes: dict[int, Point3D]
    shell_elements: dict[int, ShellElement]
    elsets: dict[str, tuple[int, ...]] = field(default_factory=dict)
    shell_sections: tuple[ShellSection, ...] = ()
    source_path: str | None = None


@dataclass(frozen=True)
class FrdStressResult:
    """Expanded CalculiX FRD shell result data needed for stress reduction."""

    path: str
    nodes: dict[int, Point3D]
    element_nodes: dict[int, tuple[int, ...]]
    components: tuple[str, ...]
    nodal_stress: dict[int, tuple[float, ...]]
    units: str = "Pa"


@dataclass(frozen=True)
class SurfacePatch:
    """A connected coplanar shell-element component."""

    patch_id: str
    element_ids: tuple[int, ...]
    normal: Vector3D
    offset: float
    bbox: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    area: float
    centroid: Point3D


@dataclass(frozen=True)
class InferredMember:
    """A shell-plate web and optional flange interpreted as one member."""

    member_id: str
    role: str
    section_type: str
    web_patch_id: str
    flange_patch_id: str | None
    direction: Vector3D
    station: float
    web_height_m: float
    flange_width_m: float | None
    web_thickness_m: float | None = None
    flange_thickness_m: float | None = None
    thickness_source: str | None = None
    confidence: float = 1.0
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlateField:
    """One inferred plate bay between adjacent stiffener/girder web lines."""

    field_id: str
    base_patch_id: str
    element_ids: tuple[int, ...]
    bbox: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    span_m: float
    spacing_m: float
    transverse_bounds: tuple[float, float]
    attached_member_ids: tuple[str, ...]
    members: tuple[InferredMember, ...] = ()
    shell_section_thickness_m: float | None = None
    confidence: float = 1.0
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class PanelStress:
    """PULS/ANYstructure stress input reduced from FE nodal stresses."""

    field_id: str
    sigma_x1_mpa: float
    sigma_x2_mpa: float
    sigma_y1_mpa: float
    sigma_y2_mpa: float
    tau_xy_mpa: float
    sample_count: int
    reduction: str
    source_units: str = "Pa"
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class FeaBucklingPanel:
    """One GUI/API selectable buckling panel discovered from FE results."""

    field_id: str
    field: PlateField
    stress: PanelStress | None
    anystructure_input: dict[str, Any]
    plot_bounds: tuple[float, float, float, float]
    buckling_result: dict[str, Any] | None = None
    usage_factor: float | None = None


@dataclass(frozen=True)
class FeaBucklingSession:
    """Complete FE-result buckling import used by both API and GUI workflows."""

    inp_path: str
    frd_path: str | None
    model: FeShellModel
    fields: tuple[PlateField, ...]
    panels: tuple[FeaBucklingPanel, ...]
    frd_summary: dict[str, Any] | None = None
    diagnostics: tuple[str, ...] = ()

    @property
    def field_count(self) -> int:
        return len(self.fields)

    @property
    def panel_count(self) -> int:
        return len(self.panels)

    def panel(self, field_id: str) -> FeaBucklingPanel:
        for panel in self.panels:
            if panel.field_id == field_id:
                return panel
        raise KeyError(field_id)

    def usage_factors(self) -> dict[str, float]:
        return {
            panel.field_id: panel.usage_factor
            for panel in self.panels
            if panel.usage_factor is not None
        }

    def summary(self) -> dict[str, Any]:
        payload = _summary_payload(self.model, self.fields, self.frd_summary)
        payload["inp_path"] = self.inp_path
        payload["frd_path"] = self.frd_path
        surface_records = {record["field_id"]: record for record in panel_3d_records(self.model, self.fields)}
        payload["panels"] = [
            {
                "field_id": panel.field_id,
                "plot_bounds": list(panel.plot_bounds),
                "surface_3d": surface_records.get(panel.field_id),
                "usage_factor": panel.usage_factor,
                "anystructure_input": panel.anystructure_input,
                "stress": None if panel.stress is None else summarize_panel_stresses([panel.stress])[0],
                "buckling_result": panel.buckling_result,
            }
            for panel in self.panels
        ]
        payload["diagnostics"] = list(self.diagnostics)
        return payload


@dataclass(frozen=True)
class _PatchInference:
    base_patch: SurfacePatch
    members: tuple[InferredMember, ...]
    stiffeners: tuple[InferredMember, ...]
    girders: tuple[InferredMember, ...]
    base_normal: Vector3D
    member_direction: Vector3D
    transverse_direction: Vector3D


def read_calculix_inp(path: str | os.PathLike[str]) -> FeShellModel:
    """Read shell nodes/elements, elsets, and shell-section metadata from ``.inp``."""

    path = str(path)
    nodes: dict[int, Point3D] = {}
    shell_elements: dict[int, ShellElement] = {}
    elsets: dict[str, list[int]] = {}
    shell_sections: list[ShellSection] = []

    mode: str | None = None
    attrs: dict[str, str | bool] = {}
    pending_shell_section: dict[str, str | None] | None = None

    with open(path, "r", encoding="utf-8", errors="ignore") as inp_file:
        for raw_line in inp_file:
            line = raw_line.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*"):
                mode = None
                attrs = _parse_keyword_attributes(line)
                keyword = _keyword_name(line)
                pending_shell_section = None
                if keyword == "node":
                    mode = "node"
                elif keyword == "element":
                    mode = "element"
                elif keyword == "elset":
                    mode = "elset"
                    name = str(attrs.get("elset", "")).strip()
                    if name:
                        elsets.setdefault(name, [])
                elif keyword == "shell section":
                    mode = "shell_section"
                    pending_shell_section = {
                        "elset": _optional_attr(attrs, "elset"),
                        "material": _optional_attr(attrs, "material"),
                        "offset": _optional_attr(attrs, "offset"),
                    }
                continue

            if mode == "node":
                parts = _csv_parts(line)
                if len(parts) >= 4:
                    nodes[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
            elif mode == "element":
                parts = _csv_parts(line)
                if len(parts) >= 4:
                    element_id = int(parts[0])
                    element_type = str(attrs.get("type", ""))
                    elset = _optional_attr(attrs, "elset")
                    shell_elements[element_id] = ShellElement(
                        element_id=element_id,
                        node_ids=tuple(int(item) for item in parts[1:]),
                        element_type=element_type,
                        elset=elset,
                    )
                    if elset:
                        elsets.setdefault(elset, []).append(element_id)
            elif mode == "elset":
                name = str(attrs.get("elset", "")).strip()
                if not name:
                    continue
                values = [int(float(part)) for part in _csv_parts(line)]
                if "generate" in attrs and len(values) >= 2:
                    step = values[2] if len(values) >= 3 else 1
                    elsets.setdefault(name, []).extend(range(values[0], values[1] + 1, step))
                else:
                    elsets.setdefault(name, []).extend(values)
            elif mode == "shell_section" and pending_shell_section is not None:
                parts = _csv_parts(line)
                thickness = _safe_float(parts[0]) if parts else None
                shell_sections.append(
                    ShellSection(
                        elset=pending_shell_section.get("elset"),
                        material=pending_shell_section.get("material"),
                        offset=pending_shell_section.get("offset"),
                        thickness_m=thickness,
                    )
                )
                pending_shell_section = None
                mode = None

    return FeShellModel(
        nodes=nodes,
        shell_elements=shell_elements,
        elsets={name: tuple(values) for name, values in elsets.items()},
        shell_sections=tuple(shell_sections),
        source_path=path,
    )


def read_calculix_frd_summary(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return lightweight metadata and result-block discovery for a CalculiX ``.frd`` file."""

    path = str(path)
    result_blocks: list[dict[str, Any]] = []
    current_block: dict[str, Any] | None = None
    current_step: int | None = None
    node_count: int | None = None
    material_names: list[str] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as frd_file:
        for line_number, raw_line in enumerate(frd_file, start=1):
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if stripped.startswith("1PSTEP"):
                numbers = [int(value) for value in re.findall(r"[-+]?\d+", stripped)]
                current_step = numbers[0] if numbers else None
            elif stripped.startswith("2C") and node_count is None:
                numbers = [int(value) for value in re.findall(r"[-+]?\d+", stripped[2:])]
                if numbers:
                    node_count = numbers[0]
            elif stripped.startswith("1UMAT"):
                name = stripped[5:].strip()
                if name:
                    material_names.append(name)
            elif stripped.startswith("-4"):
                parts = stripped.split()
                if len(parts) >= 2:
                    current_block = {
                        "name": parts[1],
                        "step": current_step,
                        "line_number": line_number,
                        "components": [],
                    }
                    result_blocks.append(current_block)
            elif stripped.startswith("-5") and current_block is not None:
                parts = stripped.split()
                if len(parts) >= 2:
                    current_block["components"].append(parts[1])

    return {
        "path": path,
        "file_size": os.path.getsize(path),
        "node_count": node_count,
        "materials": material_names,
        "result_blocks": result_blocks,
    }


def read_calculix_frd_stress(path: str | os.PathLike[str]) -> FrdStressResult:
    """Read expanded FRD nodes/connectivity and the first ``STRESS`` result block."""

    path = str(path)
    nodes: dict[int, Point3D] = {}
    element_nodes: dict[int, tuple[int, ...]] = {}
    components: list[str] = []
    nodal_stress: dict[int, tuple[float, ...]] = {}

    mode: str | None = None
    current_element_id: int | None = None
    current_element_nodes: list[int] = []

    def finish_element() -> None:
        nonlocal current_element_id, current_element_nodes
        if current_element_id is not None:
            element_nodes[current_element_id] = tuple(current_element_nodes)
        current_element_id = None
        current_element_nodes = []

    with open(path, "r", encoding="utf-8", errors="ignore") as frd_file:
        for raw_line in frd_file:
            stripped = raw_line.strip()
            if not stripped:
                continue

            if stripped.startswith("2C"):
                finish_element()
                mode = "nodes"
                continue
            if stripped.startswith("3C"):
                finish_element()
                mode = "elements"
                continue
            if stripped.startswith("-4"):
                finish_element()
                parts = stripped.split()
                mode = "stress_header" if len(parts) >= 2 and parts[1].upper() == "STRESS" else None
                if mode == "stress_header":
                    components = []
                continue
            if stripped.startswith("-3"):
                finish_element()
                mode = None
                continue

            if mode == "nodes" and stripped.startswith("-1"):
                values = _frd_numbers_after_marker(raw_line)
                if len(values) >= 4:
                    nodes[int(values[0])] = (float(values[1]), float(values[2]), float(values[3]))
            elif mode == "elements" and stripped.startswith("-1"):
                finish_element()
                values = _frd_numbers_after_marker(raw_line)
                if values:
                    current_element_id = int(values[0])
            elif mode == "elements" and stripped.startswith("-2"):
                values = _frd_numbers_after_marker(raw_line)
                current_element_nodes.extend(int(value) for value in values)
            elif mode == "stress_header" and stripped.startswith("-5"):
                parts = stripped.split()
                if len(parts) >= 2:
                    components.append(parts[1].upper())
            elif mode in {"stress_header", "stress_data"} and stripped.startswith("-1"):
                mode = "stress_data"
                values = _frd_numbers_after_marker(raw_line)
                if len(values) >= 1 + len(components):
                    node_id = int(values[0])
                    nodal_stress[node_id] = tuple(float(value) for value in values[1 : 1 + len(components)])

    return FrdStressResult(
        path=path,
        nodes=nodes,
        element_nodes=element_nodes,
        components=tuple(components),
        nodal_stress=nodal_stress,
    )


def reduce_field_stresses(
    model: FeShellModel,
    fields: Sequence[PlateField],
    frd_stress: FrdStressResult,
    *,
    transverse_edge_fraction: float = 0.2,
) -> list[PanelStress]:
    """Reduce FRD shell stresses to one ANYstructure/PULS stress set per field.

    The reduction follows the PULS S3/U3 nominal-load shape: axial stress and
    shear are uniform nominal values, while transverse stress may vary linearly
    between the two transverse sides.  CalculiX stresses are interpreted as
    tension-positive Pa; returned normal stresses are compression-positive MPa.
    """

    patches = detect_surface_patches(model)
    if not patches:
        return []
    inference = _infer_members_from_patches(model, patches)
    base_normal = inference.base_normal
    base_offset = inference.base_patch.offset
    member_direction = inference.member_direction
    transverse_direction = inference.transverse_direction

    panel_stresses: list[PanelStress] = []
    for field_item in fields:
        samples = _field_stress_samples(
            field_item,
            frd_stress,
            base_normal,
            base_offset,
            member_direction,
            transverse_direction,
        )
        if not samples:
            panel_stresses.append(
                PanelStress(
                    field_id=field_item.field_id,
                    sigma_x1_mpa=0.0,
                    sigma_x2_mpa=0.0,
                    sigma_y1_mpa=0.0,
                    sigma_y2_mpa=0.0,
                    tau_xy_mpa=0.0,
                    sample_count=0,
                    reduction="no FRD stress samples",
                    diagnostics=("no stress samples found for field element ids",),
                )
            )
            continue

        sigma_x = [-sample[1] / 1.0e6 for sample in samples]
        sigma_y = [-sample[2] / 1.0e6 for sample in samples]
        tau_xy = [sample[3] / 1.0e6 for sample in samples]
        transverse_values = [sample[0] for sample in samples]
        lower_transverse, upper_transverse = field_item.transverse_bounds
        edge_width = max((upper_transverse - lower_transverse) * transverse_edge_fraction, 1.0e-9)
        lower_indices = [
            index for index, value in enumerate(transverse_values)
            if value <= lower_transverse + edge_width
        ]
        upper_indices = [
            index for index, value in enumerate(transverse_values)
            if value >= upper_transverse - edge_width
        ]
        if not lower_indices or not upper_indices:
            ordered = sorted(range(len(samples)), key=lambda index: transverse_values[index])
            take = max(1, int(math.ceil(len(ordered) * transverse_edge_fraction)))
            lower_indices = ordered[:take]
            upper_indices = ordered[-take:]

        sigma_x_nominal = _mean(sigma_x)
        panel_stresses.append(
            PanelStress(
                field_id=field_item.field_id,
                sigma_x1_mpa=sigma_x_nominal,
                sigma_x2_mpa=sigma_x_nominal,
                sigma_y1_mpa=_mean(sigma_y[index] for index in lower_indices),
                sigma_y2_mpa=_mean(sigma_y[index] for index in upper_indices),
                tau_xy_mpa=_mean(tau_xy),
                sample_count=len(samples),
                reduction="nominal membrane: mean axial/shear, transverse side-band means",
                diagnostics=(
                    "FRD stress tensors projected to inferred panel axes",
                    "normal stresses converted from FE tension-positive Pa to compression-positive MPa",
                    "mid-surface shell result nodes preferred to avoid bending peak stress input",
                ),
            )
        )
    return panel_stresses


def summarize_panel_stresses(panel_stresses: Sequence[PanelStress]) -> list[dict[str, Any]]:
    """Flatten reduced panel stresses for JSON/CSV-style inspection."""

    return [
        {
            "field_id": stress.field_id,
            "sigma_x1_mpa": stress.sigma_x1_mpa,
            "sigma_x2_mpa": stress.sigma_x2_mpa,
            "sigma_y1_mpa": stress.sigma_y1_mpa,
            "sigma_y2_mpa": stress.sigma_y2_mpa,
            "tau_xy_mpa": stress.tau_xy_mpa,
            "sample_count": stress.sample_count,
            "reduction": stress.reduction,
            "source_units": stress.source_units,
            "diagnostics": list(stress.diagnostics),
        }
        for stress in panel_stresses
    ]


def calculate_field_buckling(
    fields: Sequence[PlateField],
    panel_stresses: Sequence[PanelStress],
    *,
    calculation_method: str = "SemiAnalytical S3/U3",
    buckling_acceptance: str = "ultimate",
    pressure_mpa: float = 0.0,
    material_yield_mpa: float = 355.0,
    elastic_modulus_mpa: float = 210000.0,
    material_factor: float = 1.15,
    poisson: float = 0.3,
) -> list[dict[str, Any]]:
    """Run ANYstructure buckling checks for fields using reduced FE stresses."""

    from anystruct.api import FlatStru

    stresses_by_field = {stress.field_id: stress for stress in panel_stresses}
    results: list[dict[str, Any]] = []
    for field_item in fields:
        stress = stresses_by_field.get(field_item.field_id)
        if stress is None:
            results.append(
                {
                    "field_id": field_item.field_id,
                    "available": False,
                    "error": "missing reduced panel stress",
                }
            )
            continue

        domain = _flat_structure_domain_for_field(field_item)
        try:
            panel = FlatStru(domain)
            panel.set_material(
                mat_yield=material_yield_mpa,
                emodule=elastic_modulus_mpa,
                material_factor=material_factor,
                poisson=poisson,
            )
            panel.set_plate_geometry(
                spacing=field_item.spacing_m * 1000.0,
                thickness=(field_item.shell_section_thickness_m or 0.0) * 1000.0,
                span=field_item.span_m * 1000.0,
            )
            panel.set_stresses(
                pressure=pressure_mpa,
                sigma_x1=stress.sigma_x1_mpa,
                sigma_x2=stress.sigma_x2_mpa,
                sigma_y1=stress.sigma_y1_mpa,
                sigma_y2=stress.sigma_y2_mpa,
                tau_xy=stress.tau_xy_mpa,
            )
            stiffener = _first_member_by_role(field_item, "stiffener")
            if stiffener is not None and domain != "Flat plate, unstiffened":
                panel.set_stiffener(
                    hw=stiffener.web_height_m * 1000.0,
                    tw=(stiffener.web_thickness_m or 0.0) * 1000.0,
                    bf=(stiffener.flange_width_m or 0.0) * 1000.0,
                    tf=(stiffener.flange_thickness_m or 0.0) * 1000.0,
                    stf_type=stiffener.section_type,
                    spacing=field_item.spacing_m * 1000.0,
                )
            girder = _first_member_by_role(field_item, "girder")
            if girder is not None and domain == "Flat plate, stiffened with girder":
                panel.set_girder(
                    hw=girder.web_height_m * 1000.0,
                    tw=(girder.web_thickness_m or 0.0) * 1000.0,
                    bf=(girder.flange_width_m or 0.0) * 1000.0,
                    tf=(girder.flange_thickness_m or 0.0) * 1000.0,
                    stf_type=girder.section_type,
                    spacing=field_item.span_m * 1000.0,
                )
            panel.set_puls_parameters(sp_or_up="UP" if stiffener is None else "SP", puls_boundary="Int")
            panel.set_buckling_parameters(
                calculation_method=calculation_method,
                buckling_acceptance=buckling_acceptance,
            )
            buckling_result = panel.get_buckling_results(calculation_method=calculation_method)
            result_available = (
                bool(buckling_result.get("available", True))
                if isinstance(buckling_result, dict)
                else True
            )
            results.append(
                {
                    "field_id": field_item.field_id,
                    "domain": domain,
                    "available": result_available,
                    "calculation_method": calculation_method,
                    "buckling_acceptance": buckling_acceptance,
                    "stress": summarize_panel_stresses([stress])[0],
                    "result": buckling_result,
                }
            )
        except Exception as err:
            results.append(
                {
                    "field_id": field_item.field_id,
                    "domain": domain,
                    "available": False,
                    "calculation_method": calculation_method,
                    "buckling_acceptance": buckling_acceptance,
                    "stress": summarize_panel_stresses([stress])[0],
                    "error": str(err),
                }
            )
    return results


def create_fea_buckling_session(
    inp_path: str | os.PathLike[str],
    frd_path: str | os.PathLike[str] | None = None,
    *,
    calculation_method: str = "SemiAnalytical S3/U3",
    buckling_acceptance: str = "ultimate",
    pressure_mpa: float = 0.0,
    material_yield_mpa: float = 355.0,
    elastic_modulus_mpa: float = 210000.0,
    material_factor: float = 1.15,
    poisson: float = 0.3,
    run_buckling: bool = True,
) -> FeaBucklingSession:
    """Create a selectable FE-result buckling session for API and GUI callers."""

    inp_path = str(inp_path)
    frd_path_text = None if frd_path is None else str(frd_path)
    model = read_calculix_inp(inp_path)
    fields = tuple(infer_plate_fields(model))
    frd_summary = read_calculix_frd_summary(frd_path_text) if frd_path_text else None
    diagnostics: list[str] = []

    if frd_path_text:
        frd_stress = read_calculix_frd_stress(frd_path_text)
        panel_stresses = tuple(reduce_field_stresses(model, fields, frd_stress))
    else:
        panel_stresses = ()
        diagnostics.append("no FRD result file supplied; panel stresses set to defaults")

    buckling_results: tuple[dict[str, Any], ...] = ()
    if run_buckling and panel_stresses:
        buckling_results = tuple(
            calculate_field_buckling(
                fields,
                panel_stresses,
                calculation_method=calculation_method,
                buckling_acceptance=buckling_acceptance,
                pressure_mpa=pressure_mpa,
                material_yield_mpa=material_yield_mpa,
                elastic_modulus_mpa=elastic_modulus_mpa,
                material_factor=material_factor,
                poisson=poisson,
            )
        )

    plot_records = panel_plot_records(model, fields)
    plot_by_field = {record["field_id"]: record for record in plot_records}
    stress_by_field = {stress.field_id: stress for stress in panel_stresses}
    result_by_field = {str(result.get("field_id")): result for result in buckling_results if result.get("field_id")}

    panels = tuple(
        FeaBucklingPanel(
            field_id=field_item.field_id,
            field=field_item,
            stress=stress_by_field.get(field_item.field_id),
            anystructure_input=anystructure_input_for_field(
                field_item,
                stress_by_field.get(field_item.field_id),
                pressure_mpa=pressure_mpa,
                material_yield_mpa=material_yield_mpa,
                elastic_modulus_mpa=elastic_modulus_mpa,
                material_factor=material_factor,
                poisson=poisson,
                calculation_method=calculation_method,
                buckling_acceptance=buckling_acceptance,
            ),
            plot_bounds=tuple(plot_by_field.get(field_item.field_id, {}).get("bounds", (0.0, 0.0, 0.0, 0.0))),
            buckling_result=result_by_field.get(field_item.field_id),
            usage_factor=_selected_uf_from_buckling_result(result_by_field.get(field_item.field_id, {})),
        )
        for field_item in fields
    )

    return FeaBucklingSession(
        inp_path=inp_path,
        frd_path=frd_path_text,
        model=model,
        fields=fields,
        panels=panels,
        frd_summary=frd_summary,
        diagnostics=tuple(diagnostics),
    )


def anystructure_input_for_field(
    field_item: PlateField,
    stress: PanelStress | None = None,
    *,
    pressure_mpa: float = 0.0,
    material_yield_mpa: float = 355.0,
    elastic_modulus_mpa: float = 210000.0,
    material_factor: float = 1.15,
    poisson: float = 0.3,
    calculation_method: str = "SemiAnalytical S3/U3",
    buckling_acceptance: str = "ultimate",
) -> dict[str, Any]:
    """Return normal ANYstructure input values inferred for one FE panel."""

    stiffener = _first_member_by_role(field_item, "stiffener")
    girder = _first_member_by_role(field_item, "girder")
    section_member = stiffener or girder
    section_type = "FB" if section_member is None else section_member.section_type
    panel_stress = stress or PanelStress(
        field_id=field_item.field_id,
        sigma_x1_mpa=0.0,
        sigma_x2_mpa=0.0,
        sigma_y1_mpa=0.0,
        sigma_y2_mpa=0.0,
        tau_xy_mpa=0.0,
        sample_count=0,
        reduction="default zero stress",
    )
    return {
        "field_id": field_item.field_id,
        "calculation_domain": _flat_structure_domain_for_field(field_item),
        "geometry": {
            "span_mm": field_item.span_m * 1000.0,
            "spacing_mm": field_item.spacing_m * 1000.0,
            "plate_thickness_mm": (field_item.shell_section_thickness_m or 0.0) * 1000.0,
        },
        "section": {
            "type": section_type,
            "web_height_mm": 0.0 if section_member is None else section_member.web_height_m * 1000.0,
            "web_thickness_mm": 0.0 if section_member is None else (section_member.web_thickness_m or 0.0) * 1000.0,
            "flange_width_mm": 0.0 if section_member is None else (section_member.flange_width_m or 0.0) * 1000.0,
            "flange_thickness_mm": 0.0 if section_member is None else (section_member.flange_thickness_m or 0.0) * 1000.0,
            "source_member_id": None if section_member is None else section_member.member_id,
        },
        "girder": None if girder is None else {
            "type": girder.section_type,
            "web_height_mm": girder.web_height_m * 1000.0,
            "web_thickness_mm": (girder.web_thickness_m or 0.0) * 1000.0,
            "flange_width_mm": (girder.flange_width_m or 0.0) * 1000.0,
            "flange_thickness_mm": (girder.flange_thickness_m or 0.0) * 1000.0,
            "source_member_id": girder.member_id,
        },
        "material": {
            "yield_mpa": material_yield_mpa,
            "elastic_modulus_mpa": elastic_modulus_mpa,
            "material_factor": material_factor,
            "poisson": poisson,
        },
        "stresses": {
            "pressure_mpa": pressure_mpa,
            "sigma_x1_mpa": panel_stress.sigma_x1_mpa,
            "sigma_x2_mpa": panel_stress.sigma_x2_mpa,
            "sigma_y1_mpa": panel_stress.sigma_y1_mpa,
            "sigma_y2_mpa": panel_stress.sigma_y2_mpa,
            "tau_xy_mpa": panel_stress.tau_xy_mpa,
            "sample_count": panel_stress.sample_count,
            "reduction": panel_stress.reduction,
        },
        "buckling": {
            "calculation_method": calculation_method,
            "buckling_acceptance": buckling_acceptance,
            "puls_boundary": "Int",
            "puls_sp_or_up": "UP" if stiffener is None else "SP",
            "puls_up_boundary": "SSSS",
            "stiffener_end_support": "Continuous",
        },
    }


def panel_plot_records(
    model: FeShellModel,
    fields: Sequence[PlateField],
    field_values: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Return 2D plot rectangles for clickable GUI rendering of FE panels."""

    patches = detect_surface_patches(model)
    inference = _infer_members_from_patches(model, patches) if patches else None
    member_direction = inference.member_direction if inference is not None else (1.0, 0.0, 0.0)
    transverse_direction = inference.transverse_direction if inference is not None else (0.0, 1.0, 0.0)
    records: list[dict[str, Any]] = []
    for index, field_item in enumerate(fields):
        bounds = _field_plot_bounds(field_item, member_direction, transverse_direction)
        value = None if field_values is None else field_values.get(field_item.field_id)
        records.append(
            {
                "field_id": field_item.field_id,
                "index": index,
                "bounds": bounds,
                "value": value,
                "span_m": field_item.span_m,
                "spacing_m": field_item.spacing_m,
            }
        )
    return records


def panel_3d_records(
    model: FeShellModel,
    fields: Sequence[PlateField],
    field_values: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Return 3D buckling-panel polygons.

    Unlike ``panel_plot_records`` this does not flatten panels into one plane.
    Each field is represented by one coarse panel surface, so panels at
    different elevations, perpendicular panels, and skewed panels remain
    separated in model coordinates without exposing the FE mesh.
    """

    records: list[dict[str, Any]] = []
    for index, field_item in enumerate(fields):
        polygons = [_field_representative_panel_polygon(model, field_item)]
        if not polygons:
            polygons = [_field_bbox_representative_polygon(field_item)]
        points = [point for polygon in polygons for point in polygon]
        bbox = _bbox(points) if points else field_item.bbox
        value = None if field_values is None else field_values.get(field_item.field_id)
        records.append(
            {
                "field_id": field_item.field_id,
                "index": index,
                "polygons": polygons,
                "bbox": bbox,
                "centroid": _mean_point(points) if points else (
                    (bbox[0][0] + bbox[0][1]) / 2.0,
                    (bbox[1][0] + bbox[1][1]) / 2.0,
                    (bbox[2][0] + bbox[2][1]) / 2.0,
                ),
                "normal": _field_representative_normal(model, field_item),
                "value": value,
                "span_m": field_item.span_m,
                "spacing_m": field_item.spacing_m,
            }
        )
    return records


def _field_representative_panel_polygon(model: FeShellModel, field_item: PlateField) -> list[Point3D]:
    """Return one oriented rectangle covering the FE elements in a panel field."""

    element_polygons = _field_element_polygons(model, field_item)
    points = [point for polygon in element_polygons for point in polygon]
    if len(points) < 3:
        return _field_bbox_representative_polygon(field_item)

    normal = _field_representative_normal(model, field_item)
    centroid = _mean_point(points)
    bbox = _bbox(points)
    axis_vectors = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    projected_axes: list[tuple[float, Vector3D]] = []
    for axis_index, axis_vector in enumerate(axis_vectors):
        projected = _project_to_plane(axis_vector, normal)
        projected_axes.append((bbox[axis_index][1] - bbox[axis_index][0], projected))
    projected_axes = [
        (extent, axis)
        for extent, axis in sorted(projected_axes, key=lambda item: item[0], reverse=True)
        if _length(axis) > 1.0e-12
    ]
    if projected_axes:
        u_axis = projected_axes[0][1]
    else:
        u_axis = _normalise(_subtract(points[1], points[0]))
    v_axis = _normalise(_cross(normal, u_axis))
    if _length(v_axis) <= 1.0e-12:
        return _field_bbox_representative_polygon(field_item)

    u_values = [_dot(_subtract(point, centroid), u_axis) for point in points]
    v_values = [_dot(_subtract(point, centroid), v_axis) for point in points]
    u_min, u_max = min(u_values), max(u_values)
    v_min, v_max = min(v_values), max(v_values)

    def corner(u_value: float, v_value: float) -> Point3D:
        return (
            centroid[0] + u_axis[0] * u_value + v_axis[0] * v_value,
            centroid[1] + u_axis[1] * u_value + v_axis[1] * v_value,
            centroid[2] + u_axis[2] * u_value + v_axis[2] * v_value,
        )

    return [
        corner(u_min, v_min),
        corner(u_max, v_min),
        corner(u_max, v_max),
        corner(u_min, v_max),
    ]


def plot_plate_fields_3d(
    model: FeShellModel,
    fields: Sequence[PlateField] | None = None,
    *,
    output_path: str | os.PathLike[str] | None = None,
    field_values: dict[str, float] | None = None,
    value_label: str = "UF",
    cmap_name: str = "tab20",
    annotate: bool = True,
    dpi: int = 180,
) -> Any:
    """Plot discovered buckling panels as 3D shell-panel surfaces only."""

    from matplotlib import colors as matplotlib_colors
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fields = list(infer_plate_fields(model) if fields is None else fields)
    records = panel_3d_records(model, fields, field_values=field_values)
    fig = plt.figure(figsize=(9.0, 5.5), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    scalar_values = _finite_field_values(field_values)
    if scalar_values:
        scalar_cmap_name = "RdYlGn_r" if cmap_name == "tab20" else cmap_name
        cmap = plt.get_cmap(scalar_cmap_name)
        value_min = min(min(scalar_values.values()), 0.0)
        value_max = max(max(scalar_values.values()), 1.0)
        if math.isclose(value_min, value_max):
            value_max = value_min + 1.0
        norm = matplotlib_colors.Normalize(vmin=value_min, vmax=value_max)
    else:
        cmap = plt.get_cmap(cmap_name, max(len(records), 1))
        norm = None

    all_points: list[Point3D] = []
    for record in records:
        value = record.get("value")
        if norm is not None:
            color = cmap(norm(value)) if value is not None and math.isfinite(float(value)) else "0.82"
        else:
            color = cmap(record["index"] % max(len(records), 1))
        collection = Poly3DCollection(
            record["polygons"],
            facecolor=color,
            edgecolor="black",
            linewidth=0.35,
            alpha=0.78,
        )
        ax.add_collection3d(collection)
        all_points.extend(point for polygon in record["polygons"] for point in polygon)
        if annotate:
            centroid = record["centroid"]
            label = record["field_id"].replace("field_", "")
            if value is not None and math.isfinite(float(value)):
                label = f"{label}\n{float(value):.2f}"
            ax.text(centroid[0], centroid[1], centroid[2], label, fontsize=7, ha="center", va="center")

    _set_equal_3d_limits(ax, all_points)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(f"Discovered buckling panels ({len(records)})" if norm is None else f"Buckling panels by {value_label}")
    if norm is not None:
        scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_map.set_array([])
        colorbar = fig.colorbar(scalar_map, ax=ax, fraction=0.036, pad=0.04)
        colorbar.set_label(value_label)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def analyze_frd_buckling(
    inp_path: str | os.PathLike[str],
    frd_path: str | os.PathLike[str],
    **buckling_kwargs: Any,
) -> dict[str, Any]:
    """Read INP/FRD, infer fields, reduce stresses, and run buckling checks."""

    model = read_calculix_inp(inp_path)
    fields = infer_plate_fields(model)
    frd_summary = read_calculix_frd_summary(frd_path)
    frd_stress = read_calculix_frd_stress(frd_path)
    panel_stresses = reduce_field_stresses(model, fields, frd_stress)
    buckling_results = calculate_field_buckling(fields, panel_stresses, **buckling_kwargs)
    payload = _summary_payload(model, fields, frd_summary)
    payload["panel_stresses"] = summarize_panel_stresses(panel_stresses)
    payload["buckling_results"] = buckling_results
    return payload


def detect_surface_patches(model: FeShellModel, decimals: int = 6) -> list[SurfacePatch]:
    """Group shell elements into connected coplanar surface patches."""

    grouped: dict[tuple[Vector3D, float], list[int]] = defaultdict(list)
    for element in model.shell_elements.values():
        points = [model.nodes[node_id] for node_id in element.corner_node_ids]
        normal = _canonical_normal(_element_normal(points), decimals=decimals)
        offset = round(_dot(normal, points[0]), decimals)
        grouped[(normal, offset)].append(element.element_id)

    patches: list[SurfacePatch] = []
    for group_index, ((normal, offset), element_ids) in enumerate(
        sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
    ):
        components = _connected_components_by_corner_edges(model, element_ids)
        for component_index, component in enumerate(components):
            patches.append(
                _make_surface_patch(
                    model,
                    patch_id=f"patch_{group_index + 1:03d}_{component_index + 1:03d}",
                    element_ids=tuple(sorted(component)),
                    normal=normal,
                    offset=offset,
                )
            )
    return patches


def infer_plate_fields(model: FeShellModel) -> list[PlateField]:
    """Infer flat plate fields and shell-plate stiffener sections from a shell model."""

    patches = detect_surface_patches(model)
    if not patches:
        return []

    inference = _infer_members_from_patches(model, patches)
    base_patch = inference.base_patch
    members = tuple(sorted(inference.members, key=lambda member: (member.role, member.station)))
    stiffeners = tuple(sorted(inference.stiffeners, key=lambda member: member.station))
    girders = tuple(sorted(inference.girders, key=lambda member: member.station))
    member_by_id = {member.member_id: member for member in members}
    base_section_thickness = _shell_section_thickness_for_patch(model, base_patch)

    if not stiffeners:
        return [
            PlateField(
                field_id="field_001",
                base_patch_id=base_patch.patch_id,
                element_ids=base_patch.element_ids,
                bbox=base_patch.bbox,
                span_m=_extent_along_patch(base_patch, inference.member_direction),
                spacing_m=_extent_along_patch(base_patch, inference.transverse_direction),
                transverse_bounds=_projected_bounds(base_patch, inference.transverse_direction),
                attached_member_ids=(),
                shell_section_thickness_m=base_section_thickness,
                diagnostics=("unstiffened base plate",),
            )
        ]

    fields: list[PlateField] = []
    base_centroids = {
        element_id: _element_centroid(model, model.shell_elements[element_id])
        for element_id in base_patch.element_ids
    }
    longitudinal_bounds = _split_bounds_with_members(base_patch, inference.member_direction, girders)

    field_index = 1
    for lower_stiffener, upper_stiffener in zip(stiffeners[:-1], stiffeners[1:]):
        lower_transverse = min(lower_stiffener.station, upper_stiffener.station)
        upper_transverse = max(lower_stiffener.station, upper_stiffener.station)
        transverse_tolerance = max((upper_transverse - lower_transverse) * 1.0e-6, 1.0e-9)
        for lower_longitudinal, upper_longitudinal, boundary_girders in longitudinal_bounds:
            longitudinal_tolerance = max((upper_longitudinal - lower_longitudinal) * 1.0e-6, 1.0e-9)
            attached_ids = [lower_stiffener.member_id, upper_stiffener.member_id]
            attached_ids.extend(member.member_id for member in boundary_girders)
            field_members = tuple(member_by_id[member_id] for member_id in dict.fromkeys(attached_ids))
            field_element_ids = tuple(
                sorted(
                    element_id
                    for element_id, centroid in base_centroids.items()
                    if lower_transverse + transverse_tolerance
                    <= _dot(centroid, inference.transverse_direction)
                    <= upper_transverse - transverse_tolerance
                    and lower_longitudinal + longitudinal_tolerance
                    <= _dot(centroid, inference.member_direction)
                    <= upper_longitudinal - longitudinal_tolerance
                )
            )
            field_bbox = _bbox_for_elements(model, field_element_ids) if field_element_ids else base_patch.bbox
            fields.append(
                PlateField(
                    field_id=f"field_{field_index:03d}",
                    base_patch_id=base_patch.patch_id,
                    element_ids=field_element_ids,
                    bbox=field_bbox,
                    span_m=upper_longitudinal - lower_longitudinal,
                    spacing_m=upper_transverse - lower_transverse,
                    transverse_bounds=(lower_transverse, upper_transverse),
                    attached_member_ids=tuple(dict.fromkeys(attached_ids)),
                    members=field_members,
                    shell_section_thickness_m=base_section_thickness,
                    diagnostics=("bounded by adjacent stiffener stations and longitudinal girder/boundary",),
                )
            )
            field_index += 1

    return fields


def summarize_plate_fields(fields: Sequence[PlateField]) -> list[dict[str, Any]]:
    """Flatten inferred plate fields for JSON/CSV-style inspection."""

    summary: list[dict[str, Any]] = []
    for field_item in fields:
        members = [
            {
                "member_id": member.member_id,
                "role": member.role,
                "section_type": member.section_type,
                "web_height_m": member.web_height_m,
                "flange_width_m": member.flange_width_m,
                "web_thickness_m": member.web_thickness_m,
                "flange_thickness_m": member.flange_thickness_m,
                "thickness_source": member.thickness_source,
                "station_m": member.station,
                "web_patch_id": member.web_patch_id,
                "flange_patch_id": member.flange_patch_id,
                "confidence": member.confidence,
                "diagnostics": list(member.diagnostics),
            }
            for member in field_item.members
        ]
        summary.append(
            {
                "field_id": field_item.field_id,
                "base_patch_id": field_item.base_patch_id,
                "element_count": len(field_item.element_ids),
                "span_m": field_item.span_m,
                "spacing_m": field_item.spacing_m,
                "transverse_bounds": list(field_item.transverse_bounds),
                "attached_member_ids": list(field_item.attached_member_ids),
                "shell_section_thickness_m": field_item.shell_section_thickness_m,
                "bbox": [list(bounds) for bounds in field_item.bbox],
                "confidence": field_item.confidence,
                "diagnostics": list(field_item.diagnostics),
                "members": members,
            }
        )
    return summary


def plot_plate_fields(
    model: FeShellModel,
    fields: Sequence[PlateField] | None = None,
    *,
    output_path: str | os.PathLike[str] | None = None,
    annotate: bool = True,
    show_members: bool = True,
    field_values: dict[str, float] | None = None,
    value_label: str = "UF",
    cmap_name: str = "tab20",
    dpi: int = 180,
) -> Any:
    """Plot inferred buckling panels in the base-plate plane.

    Axes are the inferred member direction and transverse direction.  By
    default each ``PlateField`` gets a distinct diagnostic color; when
    ``field_values`` is supplied, those scalar values drive the panel colors and
    a colorbar is added.
    """

    from matplotlib import colors as matplotlib_colors
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    fields = list(infer_plate_fields(model) if fields is None else fields)
    patches = detect_surface_patches(model)
    inference = _infer_members_from_patches(model, patches) if patches else None

    fig, ax = plt.subplots(figsize=_plot_figure_size(fields), dpi=dpi)
    scalar_values = _finite_field_values(field_values)
    if scalar_values:
        scalar_cmap_name = "RdYlGn_r" if cmap_name == "tab20" else cmap_name
        cmap = plt.get_cmap(scalar_cmap_name)
        value_min = min(min(scalar_values.values()), 0.0)
        value_max = max(max(scalar_values.values()), 1.0)
        if math.isclose(value_min, value_max):
            value_max = value_min + 1.0
        norm = matplotlib_colors.Normalize(vmin=value_min, vmax=value_max)
    else:
        cmap = plt.get_cmap(cmap_name, max(len(fields), 1))
        norm = None

    for index, field_item in enumerate(fields):
        lower_longitudinal, upper_longitudinal, lower_transverse, upper_transverse = _field_plot_bounds(
            field_item,
            inference.member_direction if inference is not None else (1.0, 0.0, 0.0),
            inference.transverse_direction if inference is not None else (0.0, 1.0, 0.0),
        )
        width = upper_longitudinal - lower_longitudinal
        height = upper_transverse - lower_transverse
        field_value = None if field_values is None else field_values.get(field_item.field_id)
        if norm is not None:
            color = cmap(norm(field_value)) if field_value is not None and math.isfinite(field_value) else "0.82"
        else:
            color = cmap(index % max(len(fields), 1))
        ax.add_patch(
            Rectangle(
                (lower_longitudinal, lower_transverse),
                width,
                height,
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.72,
            )
        )
        if annotate and width > 0.0 and height > 0.0:
            label = field_item.field_id.replace("field_", "")
            if field_value is not None and math.isfinite(field_value):
                label = f"{label}\n{field_value:.2f}"
            ax.text(
                lower_longitudinal + width / 2.0,
                lower_transverse + height / 2.0,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    if inference is not None and show_members:
        _plot_member_lines(ax, inference)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("member direction [m]")
    ax.set_ylabel("transverse direction [m]")
    ax.set_title(f"Discovered buckling panels ({len(fields)})" if norm is None else f"Buckling panels by {value_label}")
    ax.grid(True, color="0.88", linewidth=0.5)
    ax.autoscale_view()
    if norm is not None:
        scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_map.set_array([])
        colorbar = fig.colorbar(scalar_map, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(value_label)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def _summary_payload(model: FeShellModel, fields: Sequence[PlateField], frd_summary: dict[str, Any] | None) -> dict[str, Any]:
    members = _unique_members(fields)
    stiffeners = [member for member in members if member.role == "stiffener"]
    girders = [member for member in members if member.role == "girder"]
    flanges = [member for member in members if member.flange_patch_id is not None]
    section_types = sorted({member.section_type for member in members})
    spacings = [field_item.spacing_m for field_item in fields if field_item.spacing_m is not None]
    return {
        "source": model.source_path,
        "node_count": len(model.nodes),
        "element_count": len(model.shell_elements),
        "field_count": len(fields),
        "member_count": len(members),
        "web_count": len(members),
        "stiffener_count": len(stiffeners),
        "girder_count": len(girders),
        "flange_count": len(flanges),
        "section_types": section_types,
        "median_spacing_m": _median(spacings) if spacings else None,
        "shell_sections": [
            {
                "elset": section.elset,
                "material": section.material,
                "thickness_m": section.thickness_m,
                "offset": section.offset,
            }
            for section in model.shell_sections
        ],
        "frd": frd_summary,
        "fields": summarize_plate_fields(fields),
    }


def _infer_members_from_patches(model: FeShellModel, patches: Sequence[SurfacePatch]) -> _PatchInference:
    base_patch = _select_base_patch(patches)
    base_normal = base_patch.normal
    base_offset = base_patch.offset

    web_candidates = [
        patch
        for patch in patches
        if patch.patch_id != base_patch.patch_id
        and abs(_dot(patch.normal, base_normal)) < 0.25
        and _patch_crosses_base_plane(patch, base_normal, base_offset)
    ]

    web_families = _parallel_patch_families(web_candidates)
    stiffener_patches = web_families[0] if web_families else []
    girder_patches = [patch for family in web_families[1:] for patch in family]

    if stiffener_patches:
        transverse_direction = _normalise(
            tuple(sum(patch.normal[index] for patch in stiffener_patches) for index in range(3))
        )
        transverse_direction = _project_to_plane(transverse_direction, base_normal)
    else:
        transverse_direction = _fallback_base_transverse_direction(base_patch, base_normal)
    member_direction = _normalise(_cross(transverse_direction, base_normal))
    if _length(member_direction) <= 0.0:
        member_direction = _fallback_member_direction(base_patch, transverse_direction)

    flange_candidates = [
        patch
        for patch in patches
        if patch.patch_id != base_patch.patch_id
        and abs(_dot(patch.normal, base_normal)) > 0.95
        and _dot(patch.centroid, base_normal) > base_offset + 1.0e-8
    ]

    members: list[InferredMember] = []
    stiffeners = _members_from_web_patches(
        stiffener_patches,
        flange_candidates,
        base_normal,
        transverse_direction,
        member_direction,
        "stiffener",
        "stiffener",
        1,
        model,
    )
    girders = _members_from_web_patches(
        girder_patches,
        flange_candidates,
        base_normal,
        member_direction,
        transverse_direction,
        "girder",
        "girder",
        1,
        model,
    )
    members.extend(stiffeners)
    members.extend(girders)

    return _PatchInference(
        base_patch=base_patch,
        members=tuple(members),
        stiffeners=tuple(stiffeners),
        girders=tuple(girders),
        base_normal=base_normal,
        member_direction=member_direction,
        transverse_direction=transverse_direction,
    )


def _members_from_web_patches(
    web_patches: Sequence[SurfacePatch],
    flange_candidates: Sequence[SurfacePatch],
    base_normal: Vector3D,
    station_direction: Vector3D,
    member_direction: Vector3D,
    role: str,
    id_prefix: str,
    start_index: int,
    model: FeShellModel,
) -> list[InferredMember]:
    members: list[InferredMember] = []
    for offset, web_patch in enumerate(sorted(web_patches, key=lambda patch: _dot(patch.centroid, station_direction))):
        station = _dot(web_patch.centroid, station_direction)
        web_height = _extent_along_patch(web_patch, base_normal)
        flange_patch = _matching_flange_patch(web_patch, flange_candidates, base_normal, station_direction)
        flange_width: float | None = None
        section_type = "FB"
        diagnostics = ["section dimensions inferred from shell midsurface geometry"]
        if flange_patch is not None:
            flange_width = _extent_along_patch(flange_patch, station_direction)
            lower, upper = _projected_bounds(flange_patch, station_direction)
            centre_error = abs(station - (lower + upper) / 2.0)
            if centre_error <= max(flange_width * 0.05, 1.0e-6):
                section_type = "T"
            else:
                section_type = "L"
        else:
            diagnostics.append("no matching flange patch found")

        web_thickness = _shell_section_thickness_for_patch(model, web_patch)
        flange_thickness = None if flange_patch is None else _shell_section_thickness_for_patch(model, flange_patch)
        thickness_source = "shell section metadata" if web_thickness is not None or flange_thickness is not None else None
        if thickness_source:
            diagnostics.append("nominal thickness from shell section metadata")

        members.append(
            InferredMember(
                member_id=f"{id_prefix}_{start_index + offset:03d}",
                role=role,
                section_type=section_type,
                web_patch_id=web_patch.patch_id,
                flange_patch_id=None if flange_patch is None else flange_patch.patch_id,
                direction=member_direction,
                station=station,
                web_height_m=web_height,
                flange_width_m=flange_width,
                web_thickness_m=web_thickness,
                flange_thickness_m=flange_thickness,
                thickness_source=thickness_source,
                diagnostics=tuple(diagnostics),
            )
        )
    return members


def _select_base_patch(patches: Sequence[SurfacePatch]) -> SurfacePatch:
    horizontal = [patch for patch in patches if abs(patch.normal[2]) > 0.8]
    candidates = horizontal or list(patches)
    return sorted(candidates, key=lambda patch: (-patch.area, patch.bbox[2][0], patch.patch_id))[0]


def _parallel_patch_families(patches: Sequence[SurfacePatch]) -> list[list[SurfacePatch]]:
    """Return web patch families grouped by near-parallel normals."""

    families: list[list[SurfacePatch]] = []
    for patch in sorted(patches, key=lambda item: item.patch_id):
        for family in families:
            if abs(_dot(patch.normal, family[0].normal)) > 0.95:
                family.append(patch)
                break
        else:
            families.append([patch])
    families.sort(key=lambda family: (-len(family), -sum(patch.area for patch in family), family[0].patch_id))
    return families


def _split_bounds_with_members(
    base_patch: SurfacePatch,
    direction: Vector3D,
    boundary_members: Sequence[InferredMember],
) -> list[tuple[float, float, tuple[InferredMember, ...]]]:
    lower, upper = _projected_bounds(base_patch, direction)
    internal_members = [
        member
        for member in boundary_members
        if lower + 1.0e-8 < member.station < upper - 1.0e-8
    ]
    stations = [lower] + [member.station for member in sorted(internal_members, key=lambda item: item.station)] + [upper]
    result: list[tuple[float, float, tuple[InferredMember, ...]]] = []
    for first, second in zip(stations[:-1], stations[1:]):
        if second - first <= 1.0e-10:
            continue
        attached = tuple(
            member for member in internal_members
            if abs(member.station - first) <= 1.0e-8 or abs(member.station - second) <= 1.0e-8
        )
        result.append((first, second, attached))
    return result or [(lower, upper, ())]


def _matching_flange_patch(
    web_patch: SurfacePatch,
    flange_candidates: Sequence[SurfacePatch],
    base_normal: Vector3D,
    transverse_direction: Vector3D,
) -> SurfacePatch | None:
    station = _dot(web_patch.centroid, transverse_direction)
    web_top = max(_projected_bounds(web_patch, base_normal))
    best: tuple[float, SurfacePatch] | None = None
    for flange_patch in flange_candidates:
        lower, upper = _projected_bounds(flange_patch, transverse_direction)
        width = max(upper - lower, 1.0e-9)
        if lower - max(width * 0.1, 1.0e-6) <= station <= upper + max(width * 0.1, 1.0e-6):
            height_error = abs(_dot(flange_patch.centroid, base_normal) - web_top)
            score = height_error + abs(station - (lower + upper) / 2.0) * 0.01
            if best is None or score < best[0]:
                best = (score, flange_patch)
    return None if best is None else best[1]


def _patch_crosses_base_plane(patch: SurfacePatch, base_normal: Vector3D, base_offset: float) -> bool:
    lower, upper = _projected_bounds(patch, base_normal)
    tolerance = max((upper - lower) * 1.0e-6, 1.0e-8)
    return lower - tolerance <= base_offset <= upper + tolerance and upper - lower > tolerance


def _connected_components_by_corner_edges(model: FeShellModel, element_ids: Iterable[int]) -> list[list[int]]:
    edge_to_elements: dict[tuple[int, int], list[int]] = defaultdict(list)
    element_ids = list(element_ids)
    for element_id in element_ids:
        corners = list(model.shell_elements[element_id].corner_node_ids)
        for first, second in zip(corners, corners[1:] + corners[:1]):
            edge_to_elements[tuple(sorted((first, second)))].append(element_id)

    adjacency: dict[int, set[int]] = {element_id: set() for element_id in element_ids}
    for shared_elements in edge_to_elements.values():
        if len(shared_elements) < 2:
            continue
        for element_id in shared_elements:
            adjacency[element_id].update(other for other in shared_elements if other != element_id)

    components: list[list[int]] = []
    seen: set[int] = set()
    for element_id in element_ids:
        if element_id in seen:
            continue
        stack = [element_id]
        seen.add(element_id)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    return components


def _make_surface_patch(
    model: FeShellModel,
    patch_id: str,
    element_ids: tuple[int, ...],
    normal: Vector3D,
    offset: float,
) -> SurfacePatch:
    coords = [
        model.nodes[node_id]
        for element_id in element_ids
        for node_id in model.shell_elements[element_id].corner_node_ids
    ]
    bbox = _bbox(coords)
    area = sum(_element_area([model.nodes[node_id] for node_id in model.shell_elements[element_id].corner_node_ids])
               for element_id in element_ids)
    centroid = (
        sum(point[0] for point in coords) / len(coords),
        sum(point[1] for point in coords) / len(coords),
        sum(point[2] for point in coords) / len(coords),
    )
    return SurfacePatch(
        patch_id=patch_id,
        element_ids=element_ids,
        normal=normal,
        offset=offset,
        bbox=bbox,
        area=area,
        centroid=centroid,
    )


def _bbox_for_elements(model: FeShellModel, element_ids: Iterable[int]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    points = [
        model.nodes[node_id]
        for element_id in element_ids
        for node_id in model.shell_elements[element_id].corner_node_ids
    ]
    return _bbox(points)


def _field_element_polygons(model: FeShellModel, field_item: PlateField) -> list[list[Point3D]]:
    polygons: list[list[Point3D]] = []
    for element_id in field_item.element_ids:
        element = model.shell_elements.get(element_id)
        if element is None:
            continue
        polygon = [model.nodes[node_id] for node_id in element.corner_node_ids if node_id in model.nodes]
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons


def _field_representative_normal(model: FeShellModel, field_item: PlateField) -> Vector3D:
    for polygon in _field_element_polygons(model, field_item):
        normal = _element_normal(polygon)
        if _length(normal) > 0.0:
            return _normalise(normal)
    corners = _field_bbox_representative_polygon(field_item)
    normal = _element_normal(corners)
    return _normalise(normal) if _length(normal) > 0.0 else (0.0, 0.0, 1.0)


def _field_bbox_representative_polygon(field_item: PlateField) -> list[Point3D]:
    bbox = field_item.bbox
    x0, x1 = bbox[0]
    y0, y1 = bbox[1]
    z0, z1 = bbox[2]
    ranges = [abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)]
    collapsed_axis = min(range(3), key=lambda index: ranges[index])
    if collapsed_axis == 0:
        x = (x0 + x1) / 2.0
        return [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]
    if collapsed_axis == 1:
        y = (y0 + y1) / 2.0
        return [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
    z = (z0 + z1) / 2.0
    return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]


def _mean_point(points: Sequence[Point3D]) -> Point3D:
    if not points:
        return (0.0, 0.0, 0.0)
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
        sum(point[2] for point in points) / len(points),
    )


def _set_equal_3d_limits(ax: Any, points: Sequence[Point3D]) -> None:
    if not points:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_zlim(0.0, 1.0)
        return
    bbox = _bbox(points)
    centers = [(axis[0] + axis[1]) / 2.0 for axis in bbox]
    half_span = max((axis[1] - axis[0]) / 2.0 for axis in bbox)
    half_span = max(half_span, 0.5)
    pad = half_span * 0.08
    half_span += pad
    ax.set_xlim(centers[0] - half_span, centers[0] + half_span)
    ax.set_ylim(centers[1] - half_span, centers[1] + half_span)
    ax.set_zlim(centers[2] - half_span, centers[2] + half_span)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def _bbox(points: Sequence[Point3D]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    return tuple((min(point[index] for point in points), max(point[index] for point in points)) for index in range(3))  # type: ignore[return-value]


def _bbox_corners(
    bbox: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> list[Point3D]:
    return [(x, y, z) for x in bbox[0] for y in bbox[1] for z in bbox[2]]


def _shell_section_thickness_for_patch(model: FeShellModel, patch: SurfacePatch) -> float | None:
    if not model.shell_sections:
        return None
    patch_elements = set(patch.element_ids)
    for section in model.shell_sections:
        if not section.elset:
            continue
        section_elements = set(model.elsets.get(section.elset, ()))
        if patch_elements & section_elements:
            return section.thickness_m
    return model.shell_sections[0].thickness_m


def _unique_members(fields: Sequence[PlateField]) -> list[InferredMember]:
    by_id: dict[str, InferredMember] = {}
    for field_item in fields:
        for member in field_item.members:
            by_id[member.member_id] = member
    return [by_id[key] for key in sorted(by_id)]


def _buckling_uf_by_field(buckling_results: Sequence[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in buckling_results:
        field_id = item.get("field_id")
        if not field_id:
            continue
        uf = _selected_uf_from_buckling_result(item)
        if uf is not None:
            values[str(field_id)] = uf
    return values


def _selected_uf_from_buckling_result(item: dict[str, Any]) -> float | None:
    result = item.get("result")
    if not isinstance(result, dict):
        return None
    return _find_usage_factor(result)


def _find_usage_factor(value: object) -> float | None:
    preferred_keys = (
        "selected UF",
        "ultimate UF",
        "buckling UF",
        "Plate buckling",
        "Actual usage Factor",
        "actual usage factor",
        "usage_factor",
        "usage factor",
        "UF",
    )
    if isinstance(value, dict):
        lower_lookup = {str(key).lower(): key for key in value}
        for key in preferred_keys:
            actual_key = lower_lookup.get(key.lower())
            if actual_key is not None:
                uf = _first_finite_float(value.get(actual_key))
                if uf is not None:
                    return uf
        for nested in value.values():
            uf = _find_usage_factor(nested)
            if uf is not None:
                return uf
        return None
    return _first_finite_float(value)


def _first_finite_float(value: object) -> float | None:
    direct = _safe_float(value)
    if direct is not None:
        return direct
    if isinstance(value, (list, tuple)):
        for item in value:
            nested = _first_finite_float(item)
            if nested is not None:
                return nested
    return None


def _finite_field_values(field_values: dict[str, float] | None) -> dict[str, float]:
    if not field_values:
        return {}
    return {
        str(key): float(value)
        for key, value in field_values.items()
        if _safe_float(value) is not None
    }


def _field_stress_samples(
    field_item: PlateField,
    frd_stress: FrdStressResult,
    base_normal: Vector3D,
    base_offset: float,
    member_direction: Vector3D,
    transverse_direction: Vector3D,
) -> list[tuple[float, float, float, float]]:
    all_samples: list[tuple[float, float, float, float, float]] = []
    seen_node_ids: set[int] = set()
    for element_id in field_item.element_ids:
        for node_id in frd_stress.element_nodes.get(element_id, ()):
            if node_id in seen_node_ids or node_id not in frd_stress.nodal_stress:
                continue
            point = frd_stress.nodes.get(node_id)
            if point is None:
                continue
            seen_node_ids.add(node_id)
            sigma_x, sigma_y, tau_xy = _project_frd_stress(
                frd_stress.components,
                frd_stress.nodal_stress[node_id],
                member_direction,
                transverse_direction,
            )
            distance_to_mid_surface = abs(_dot(point, base_normal) - base_offset)
            all_samples.append((_dot(point, transverse_direction), sigma_x, sigma_y, tau_xy, distance_to_mid_surface))

    if not all_samples:
        return []

    mid_surface_tolerance = max((field_item.shell_section_thickness_m or 0.0) * 0.2, 1.0e-8)
    mid_surface_samples = [sample for sample in all_samples if sample[4] <= mid_surface_tolerance]
    selected = mid_surface_samples or all_samples
    return [sample[:4] for sample in selected]


def _project_frd_stress(
    components: Sequence[str],
    values: Sequence[float],
    member_direction: Vector3D,
    transverse_direction: Vector3D,
) -> tuple[float, float, float]:
    stress_by_component = {component.upper(): float(value) for component, value in zip(components, values)}
    sxx = stress_by_component.get("SXX", 0.0)
    syy = stress_by_component.get("SYY", 0.0)
    szz = stress_by_component.get("SZZ", 0.0)
    sxy = stress_by_component.get("SXY", 0.0)
    syz = stress_by_component.get("SYZ", 0.0)
    szx = stress_by_component.get("SZX", stress_by_component.get("SXZ", 0.0))
    tensor = (
        (sxx, sxy, szx),
        (sxy, syy, syz),
        (szx, syz, szz),
    )
    sigma_x = _quadratic_tensor_product(member_direction, tensor)
    sigma_y = _quadratic_tensor_product(transverse_direction, tensor)
    tau_xy = _mixed_tensor_product(member_direction, tensor, transverse_direction)
    return sigma_x, sigma_y, tau_xy


def _quadratic_tensor_product(vector: Vector3D, tensor: tuple[Vector3D, Vector3D, Vector3D]) -> float:
    return _mixed_tensor_product(vector, tensor, vector)


def _mixed_tensor_product(
    first: Vector3D,
    tensor: tuple[Vector3D, Vector3D, Vector3D],
    second: Vector3D,
) -> float:
    return sum(first[row] * tensor[row][col] * second[col] for row in range(3) for col in range(3))


def _flat_structure_domain_for_field(field_item: PlateField) -> str:
    has_stiffener = any(member.role == "stiffener" for member in field_item.members)
    has_girder = any(member.role == "girder" for member in field_item.members)
    if has_stiffener and has_girder:
        return "Flat plate, stiffened with girder"
    if has_stiffener:
        return "Flat plate, stiffened"
    return "Flat plate, unstiffened"


def _first_member_by_role(field_item: PlateField, role: str) -> InferredMember | None:
    members = [member for member in field_item.members if member.role == role]
    if not members:
        return None
    return sorted(members, key=lambda member: member.member_id)[0]


def _plot_figure_size(fields: Sequence[PlateField]) -> tuple[float, float]:
    if not fields:
        return (8.0, 4.5)
    bbox = _bbox([corner for field_item in fields for corner in _bbox_corners(field_item.bbox)])
    x_range = max(bbox[0][1] - bbox[0][0], 1.0)
    y_range = max(bbox[1][1] - bbox[1][0], 1.0)
    ratio = max(min(x_range / y_range, 3.0), 1.2)
    return (min(max(5.0 * ratio, 7.0), 14.0), 5.0)


def _field_plot_bounds(
    field_item: PlateField,
    member_direction: Vector3D,
    transverse_direction: Vector3D,
) -> tuple[float, float, float, float]:
    corners = _bbox_corners(field_item.bbox)
    longitudinal_values = [_dot(point, member_direction) for point in corners]
    transverse_values = [_dot(point, transverse_direction) for point in corners]
    return (
        min(longitudinal_values),
        max(longitudinal_values),
        min(transverse_values),
        max(transverse_values),
    )


def _plot_member_lines(ax: Any, inference: _PatchInference) -> None:
    lower_longitudinal, upper_longitudinal = _projected_bounds(inference.base_patch, inference.member_direction)
    lower_transverse, upper_transverse = _projected_bounds(inference.base_patch, inference.transverse_direction)
    for member in inference.stiffeners:
        ax.plot(
            [lower_longitudinal, upper_longitudinal],
            [member.station, member.station],
            color="black",
            linewidth=0.7,
            alpha=0.72,
        )
    for member in inference.girders:
        ax.plot(
            [member.station, member.station],
            [lower_transverse, upper_transverse],
            color="black",
            linewidth=1.4,
            linestyle="--",
            alpha=0.82,
        )


def _element_centroid(model: FeShellModel, element: ShellElement) -> Point3D:
    points = [model.nodes[node_id] for node_id in element.corner_node_ids]
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
        sum(point[2] for point in points) / len(points),
    )


def _element_normal(points: Sequence[Point3D]) -> Vector3D:
    if len(points) < 3:
        return (0.0, 0.0, 0.0)
    for idx in range(1, len(points) - 1):
        normal = _cross(_subtract(points[idx], points[0]), _subtract(points[idx + 1], points[0]))
        if _length(normal) > 1.0e-12:
            return _normalise(normal)
    return (0.0, 0.0, 0.0)


def _element_area(points: Sequence[Point3D]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx in range(1, len(points) - 1):
        area += 0.5 * _length(_cross(_subtract(points[idx], points[0]), _subtract(points[idx + 1], points[0])))
    return area


def _canonical_normal(normal: Vector3D, decimals: int = 6) -> Vector3D:
    normal = _normalise(normal)
    largest = max(range(3), key=lambda index: abs(normal[index]))
    if normal[largest] < 0.0:
        normal = tuple(-value for value in normal)  # type: ignore[assignment]
    return tuple(round(value, decimals) for value in normal)  # type: ignore[return-value]


def _projected_bounds(patch: SurfacePatch, direction: Vector3D) -> tuple[float, float]:
    corners = [
        (x, y, z)
        for x in patch.bbox[0]
        for y in patch.bbox[1]
        for z in patch.bbox[2]
    ]
    values = [_dot(point, direction) for point in corners]
    return min(values), max(values)


def _extent_along_patch(patch: SurfacePatch, direction: Vector3D) -> float:
    lower, upper = _projected_bounds(patch, direction)
    return upper - lower


def _extent_from_bbox(
    bbox: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    direction: Vector3D,
) -> float:
    corners = [(x, y, z) for x in bbox[0] for y in bbox[1] for z in bbox[2]]
    values = [_dot(point, direction) for point in corners]
    return max(values) - min(values)


def _fallback_base_transverse_direction(patch: SurfacePatch, base_normal: Vector3D) -> Vector3D:
    ranges = [patch.bbox[index][1] - patch.bbox[index][0] for index in range(3)]
    normal_axis = max(range(3), key=lambda index: abs(base_normal[index]))
    candidates = [index for index in range(3) if index != normal_axis]
    axis = min(candidates, key=lambda index: ranges[index])
    vector = [0.0, 0.0, 0.0]
    vector[axis] = 1.0
    return tuple(vector)  # type: ignore[return-value]


def _fallback_member_direction(patch: SurfacePatch, transverse_direction: Vector3D) -> Vector3D:
    ranges = [patch.bbox[index][1] - patch.bbox[index][0] for index in range(3)]
    transverse_axis = max(range(3), key=lambda index: abs(transverse_direction[index]))
    candidates = [index for index in range(3) if index != transverse_axis]
    axis = max(candidates, key=lambda index: ranges[index])
    vector = [0.0, 0.0, 0.0]
    vector[axis] = 1.0
    return tuple(vector)  # type: ignore[return-value]


def _subtract(first: Point3D, second: Point3D) -> Vector3D:
    return (first[0] - second[0], first[1] - second[1], first[2] - second[2])


def _cross(first: Vector3D, second: Vector3D) -> Vector3D:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot(first: Sequence[float], second: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(first, second))


def _length(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def _normalise(vector: Sequence[float]) -> Vector3D:
    length = _length(vector)
    if length <= 1.0e-15:
        return (0.0, 0.0, 0.0)
    return tuple(float(value) / length for value in vector)  # type: ignore[return-value]


def _project_to_plane(vector: Vector3D, normal: Vector3D) -> Vector3D:
    projected = tuple(vector[index] - _dot(vector, normal) * normal[index] for index in range(3))
    return _normalise(projected)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(float(value) for value in values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _parse_keyword_attributes(line: str) -> dict[str, str | bool]:
    parts = [part.strip() for part in line.split(",")]
    attrs: dict[str, str | bool] = {}
    for item in parts[1:]:
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            attrs[key.strip().lower()] = value.strip()
        else:
            attrs[item.strip().lower()] = True
    return attrs


def _keyword_name(line: str) -> str:
    return line.split(",", 1)[0].strip().lstrip("*").lower()


def _optional_attr(attrs: dict[str, str | bool], name: str) -> str | None:
    value = attrs.get(name.lower())
    if value is None or value is True:
        return None
    return str(value)


def _csv_parts(line: str) -> list[str]:
    return [part.strip() for part in line.split(",") if part.strip()]


_FRD_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[Ee][-+]?\d+)?|[-+]?\d+(?:[Ee][-+]?\d+)?")


def _frd_numbers_after_marker(line: str) -> list[float]:
    stripped = line.strip()
    return [float(match.group(0)) for match in _FRD_NUMBER_PATTERN.finditer(stripped[2:])]


def _safe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Infer flat stiffened plate fields from CalculiX shell files.")
    parser.add_argument("inp", help="CalculiX/PrePoMax .inp file")
    parser.add_argument("--frd", help="Optional CalculiX .frd result file")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a compact text summary")
    parser.add_argument("--plot", metavar="PNG", help="Save a colored 2D plot of the inferred buckling panels")
    parser.add_argument("--no-plot-labels", action="store_true", help="Do not annotate panel ids on the plot")
    parser.add_argument(
        "--plot-color-by",
        choices=("panel", "uf"),
        help="Color plot by panel id or buckling utilization factor; default is uf with --buckling, otherwise panel",
    )
    parser.add_argument("--stresses", action="store_true", help="Reduce FRD stresses to one stress set per field")
    parser.add_argument("--buckling", action="store_true", help="Run ANYstructure buckling checks from reduced FRD stresses")
    parser.add_argument(
        "--buckling-method",
        default="SemiAnalytical S3/U3",
        choices=("DNV-RP-C201 - prescriptive", "SemiAnalytical S3/U3", "ML-Numeric (PULS based)"),
        help="ANYstructure buckling method used with --buckling",
    )
    parser.add_argument(
        "--buckling-acceptance",
        default="ultimate",
        choices=("buckling", "ultimate"),
        help="Buckling result branch used with --buckling",
    )
    parser.add_argument("--pressure-mpa", type=float, default=0.0, help="Optional lateral pressure for buckling checks")
    args = parser.parse_args(argv)

    model = read_calculix_inp(args.inp)
    fields = infer_plate_fields(model)
    frd_summary = read_calculix_frd_summary(args.frd) if args.frd else None
    payload = _summary_payload(model, fields, frd_summary)
    panel_stresses: list[PanelStress] = []
    if args.stresses or args.buckling:
        if not args.frd:
            parser.error("--stresses/--buckling requires --frd")
        frd_stress = read_calculix_frd_stress(args.frd)
        panel_stresses = reduce_field_stresses(model, fields, frd_stress)
        payload["panel_stresses"] = summarize_panel_stresses(panel_stresses)
    if args.buckling:
        payload["buckling_results"] = calculate_field_buckling(
            fields,
            panel_stresses,
            calculation_method=args.buckling_method,
            buckling_acceptance=args.buckling_acceptance,
            pressure_mpa=args.pressure_mpa,
        )
    if args.plot:
        plot_path = Path(args.plot)
        if plot_path.parent:
            plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_color_by = args.plot_color_by or ("uf" if args.buckling else "panel")
        if plot_color_by == "uf" and not args.buckling:
            parser.error("--plot-color-by uf requires --buckling")
        field_values = _buckling_uf_by_field(payload["buckling_results"]) if plot_color_by == "uf" else None
        fig = plot_plate_fields(
            model,
            fields,
            output_path=plot_path,
            annotate=not args.no_plot_labels,
            field_values=field_values,
            value_label="UF",
        )
        from matplotlib import pyplot as plt

        plt.close(fig)
        payload["plot_path"] = str(plot_path)
        payload["plot_color_by"] = plot_color_by

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Nodes: {payload['node_count']}")
        print(f"Shell elements: {payload['element_count']}")
        print(f"Plate fields: {payload['field_count']}")
        print(f"Members/webs: {payload['web_count']}")
        print(f"Stiffeners: {payload['stiffener_count']}")
        print(f"Girders: {payload['girder_count']}")
        print(f"Flanges: {payload['flange_count']}")
        print(f"Section types: {', '.join(payload['section_types']) or 'n/a'}")
        if payload["median_spacing_m"] is not None:
            print(f"Median spacing: {payload['median_spacing_m']:.6g} m")
        if args.stresses or args.buckling:
            max_abs_sigma = max(
                (
                    abs(value)
                    for stress in panel_stresses
                    for value in (
                        stress.sigma_x1_mpa,
                        stress.sigma_x2_mpa,
                        stress.sigma_y1_mpa,
                        stress.sigma_y2_mpa,
                    )
                ),
                default=0.0,
            )
            print(f"Reduced stress panels: {len(panel_stresses)}")
            print(f"Max abs reduced normal stress: {max_abs_sigma:.6g} MPa")
        if args.buckling:
            available = [result for result in payload["buckling_results"] if result.get("available")]
            print(f"Buckling panels evaluated: {len(available)}")
        if args.plot:
            print(f"Panel plot: {payload['plot_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
