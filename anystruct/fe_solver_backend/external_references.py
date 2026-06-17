"""Generated external FE reference-case decks.

The helpers in this module create deterministic CalculiX/Abaqus-style input
decks from small FEModel cases.  They are intended as reproducible handoff
artifacts for external solver comparison; they do not require CalculiX to be
installed and they do not claim numerical agreement by themselves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .boundary import BoundaryCondition, LoadCase
from .elements import BeamElement, ShellElement
from .fe_core import FEModel, Material
from .mesh_gen import generate_simple_panel_mesh


DEFAULT_EXTERNAL_REFERENCE_PATH = Path("reports/external_references/external_reference_report.json")
_DOF_TO_CALCULIX = {"ux": 1, "uy": 2, "uz": 3, "rx": 4, "ry": 5, "rz": 6}


@dataclass(frozen=True)
class ExternalReferenceCase:
    """One generated external-reference input deck."""

    name: str
    kind: str
    inp_path: Path
    metadata_path: Path
    model_summary: Mapping[str, Any]
    load_summary: Mapping[str, Any]
    assumptions: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "inp_path": str(self.inp_path),
            "metadata_path": str(self.metadata_path),
            "model_summary": dict(self.model_summary),
            "load_summary": dict(self.load_summary),
            "assumptions": list(self.assumptions),
        }


def _fmt(value: float) -> str:
    return f"{float(value):.16g}"


def _material_block(materials: Mapping[str, Material]) -> List[str]:
    lines: List[str] = []
    for material in materials.values():
        lines.extend(
            [
                f"*MATERIAL, NAME={material.name}",
                "*ELASTIC",
                f"{_fmt(material.elastic_modulus)}, {_fmt(material.poisson_ratio)}",
            ]
        )
        if material.density:
            lines.extend(["*DENSITY", _fmt(material.density)])
    return lines


def _element_type(element: Any) -> str:
    if isinstance(element, ShellElement):
        return "S8" if len(element.node_ids) == 8 else "S4"
    if isinstance(element, BeamElement):
        return "B31"
    return "UNKNOWN"


def _element_sets_by_type_and_material(model: FEModel) -> Dict[Tuple[str, str], List[int]]:
    groups: Dict[Tuple[str, str], List[int]] = {}
    for element_id, element in model.mesh.elements.items():
        groups.setdefault((_element_type(element), element.material_name), []).append(int(element_id))
    return groups


def _section_blocks(model: FEModel) -> Tuple[List[str], List[str]]:
    lines: List[str] = []
    assumptions: List[str] = []
    for (element_type, material_name), element_ids in _element_sets_by_type_and_material(model).items():
        elset = f"E_{element_type}_{material_name}"
        ids = ", ".join(str(element_id) for element_id in element_ids)
        lines.append(f"*ELSET, ELSET={elset}")
        lines.append(ids)
        first = model.mesh.elements[element_ids[0]]
        if isinstance(first, ShellElement):
            lines.extend([f"*SHELL SECTION, ELSET={elset}, MATERIAL={material_name}", _fmt(first.thickness)])
        elif isinstance(first, BeamElement):
            area = float(first.cross_section.get("area", 0.01))
            side = np.sqrt(max(area, 1.0e-18))
            lines.extend([f"*BEAM SECTION, ELSET={elset}, MATERIAL={material_name}, SECTION=RECT", f"{_fmt(side)}, {_fmt(side)}"])
            assumptions.append(
                f"Beam element set {elset} is exported as an equivalent square RECT section preserving area; "
                "Iy/Iz/J exact matching is not represented in this v1 deck writer."
            )
    return lines, assumptions


def _boundary_block(model: FEModel) -> List[str]:
    lines: List[str] = []
    if not model.boundary_conditions:
        return lines
    lines.append("*BOUNDARY")
    for bc in model.boundary_conditions:
        for node_id in bc.node_ids:
            for dof_name, value in bc.dof_constraints.items():
                if abs(float(value)) > 0.0:
                    continue
                dof = _DOF_TO_CALCULIX.get(dof_name)
                if dof is not None:
                    lines.append(f"{int(node_id)}, {dof}, {dof}, 0.")
    return lines


def _load_block(load_case: Optional[LoadCase]) -> Tuple[List[str], Dict[str, Any]]:
    if load_case is None:
        return [], {"name": None, "nodal_loads": 0, "pressure_loads": 0}
    lines: List[str] = []
    if load_case.nodal_loads:
        lines.append("*CLOAD")
        for node_id, values in sorted(load_case.nodal_loads.items()):
            for idx, value in enumerate(np.asarray(values, dtype=float), start=1):
                if abs(float(value)) > 0.0:
                    lines.append(f"{int(node_id)}, {idx}, {_fmt(value)}")
    if load_case.pressure_loads:
        lines.append("*DLOAD")
        for element_id, pressure in sorted(load_case.pressure_loads.items()):
            lines.append(f"{int(element_id)}, P, {_fmt(pressure)}")
    if load_case.gravity is not None:
        gx, gy, gz = np.asarray(load_case.gravity, dtype=float).reshape(3)
        lines.append("*DLOAD")
        lines.append(f"ALL, GRAV, {_fmt(np.linalg.norm([gx, gy, gz]))}, {_fmt(gx)}, {_fmt(gy)}, {_fmt(gz)}")
    return lines, {
        "name": load_case.name,
        "nodal_loads": len(load_case.nodal_loads),
        "pressure_loads": len(load_case.pressure_loads),
        "has_gravity": load_case.gravity is not None,
    }


def write_calculix_input_deck(
    model: FEModel,
    load_case: Optional[LoadCase],
    output_path: Path | str,
    *,
    analysis: str = "static",
    metadata: Optional[Mapping[str, Any]] = None,
) -> ExternalReferenceCase:
    """Write a deterministic CalculiX-style input deck and sidecar metadata."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.apply_boundary_conditions()
    lines: List[str] = [
        "** Generated by fe_solver.external_references",
        f"** Model: {model.name}",
        "*NODE",
    ]
    for node_id, node in sorted(model.mesh.nodes.items()):
        lines.append(f"{int(node_id)}, {_fmt(node.x)}, {_fmt(node.y)}, {_fmt(node.z)}")
    for (element_type, _material_name), element_ids in sorted(_element_sets_by_type_and_material(model).items()):
        if element_type == "UNKNOWN":
            continue
        lines.append(f"*ELEMENT, TYPE={element_type}")
        for element_id in element_ids:
            element = model.mesh.elements[element_id]
            lines.append(f"{int(element_id)}, " + ", ".join(str(int(node_id)) for node_id in element.node_ids))
    lines.extend(_material_block(model.materials))
    section_lines, assumptions = _section_blocks(model)
    lines.extend(section_lines)
    lines.extend(_boundary_block(model))
    load_lines, load_summary = _load_block(load_case)
    if analysis == "buckling":
        lines.extend(["*STEP", "*BUCKLE", "5"])
    elif analysis == "frequency":
        lines.extend(["*STEP", "*FREQUENCY", "5"])
    else:
        lines.extend(["*STEP", "*STATIC"])
    lines.extend(load_lines)
    lines.extend(["*NODE FILE", "U", "*EL FILE", "S", "*END STEP", ""])
    output.write_text("\n".join(lines), encoding="utf-8")

    model_summary = {
        "name": model.name,
        "nodes": len(model.mesh.nodes),
        "elements": len(model.mesh.elements),
        "materials": sorted(model.materials),
        "analysis": analysis,
    }
    sidecar = output.with_suffix(".json")
    payload = {
        "name": output.stem,
        "kind": analysis,
        "model_summary": model_summary,
        "load_summary": load_summary,
        "assumptions": assumptions,
        "metadata": dict(metadata or {}),
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ExternalReferenceCase(
        name=output.stem,
        kind=analysis,
        inp_path=output,
        metadata_path=sidecar,
        model_summary=model_summary,
        load_summary=load_summary,
        assumptions=tuple(assumptions),
    )


def _pressure_plate_case(output_dir: Path) -> ExternalReferenceCase:
    model = generate_simple_panel_mesh(2.0, 1.0, 0.01, num_divisions_x=2, num_divisions_y=1)
    model.name = "external_pressure_plate_s4"
    load_case = LoadCase("pressure")
    for element_id in model.mesh.elements:
        load_case.add_pressure_load(element_id, 1000.0)
    return write_calculix_input_deck(
        model,
        load_case,
        output_dir / "pressure_plate_s4.inp",
        analysis="static",
        metadata={"purpose": "S4 pressure plate external reference deck"},
    )


def _beam_buckling_case(output_dir: Path) -> ExternalReferenceCase:
    model = FEModel("external_beam_column_buckling")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    section = {"area": 0.02, "Iy": 3.0e-6, "Iz": 5.0e-6, "J": 2.0e-6}
    for i in range(5):
        model.add_node(i + 1, float(i), 0.0, 0.0)
    for i in range(4):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "steel", section))
    all_nodes = list(model.mesh.nodes)
    model.add_boundary_condition(BoundaryCondition("suppress", all_nodes, {"ux": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0}))
    model.add_boundary_condition(BoundaryCondition("pins", [1, 5], {"uy": 0.0}))
    load_case = LoadCase("unit_compression")
    load_case.add_nodal_load(5, [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return write_calculix_input_deck(
        model,
        load_case,
        output_dir / "beam_column_buckling.inp",
        analysis="buckling",
        metadata={"purpose": "Beam-column buckling external reference deck"},
    )


def _cylindrical_shell_case(output_dir: Path) -> ExternalReferenceCase:
    model = FEModel("external_cylinder_s4_pressure")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    radius = 1.0
    height = 1.0
    n_circ = 8
    n_z = 2
    node_id = 1
    grid: Dict[Tuple[int, int], int] = {}
    for iz in range(n_z + 1):
        z = height * iz / n_z
        for it in range(n_circ):
            theta = 2.0 * np.pi * it / n_circ
            grid[(iz, it)] = node_id
            model.add_node(node_id, radius * np.cos(theta), radius * np.sin(theta), z)
            node_id += 1
    elem_id = 1
    for iz in range(n_z):
        for it in range(n_circ):
            n1 = grid[(iz, it)]
            n2 = grid[(iz, (it + 1) % n_circ)]
            n3 = grid[(iz + 1, (it + 1) % n_circ)]
            n4 = grid[(iz + 1, it)]
            model.add_element(elem_id, ShellElement(elem_id, [n1, n2, n3, n4], "steel", thickness=0.02))
            elem_id += 1
    bottom = [grid[(0, it)] for it in range(n_circ)]
    top = [grid[(n_z, it)] for it in range(n_circ)]
    model.add_boundary_condition(BoundaryCondition("bottom_uz", bottom, {"uz": 0.0}))
    model.add_boundary_condition(BoundaryCondition("top_uz", top, {"uz": 0.0}))
    model.add_boundary_condition(BoundaryCondition("reference_xy", [bottom[0]], {"ux": 0.0, "uy": 0.0}))
    load_case = LoadCase("internal_pressure")
    for element_id in model.mesh.elements:
        load_case.add_pressure_load(element_id, 1000.0)
    return write_calculix_input_deck(
        model,
        load_case,
        output_dir / "cylinder_s4_pressure.inp",
        analysis="static",
        metadata={"purpose": "Cylindrical shell pressure external reference deck"},
    )


def generate_external_reference_cases(output_dir: Path | str = Path("reports/external_references/decks")) -> List[ExternalReferenceCase]:
    """Generate the default external reference decks."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return [_pressure_plate_case(root), _beam_buckling_case(root), _cylindrical_shell_case(root)]


def generate_external_reference_report(output_dir: Path | str = Path("reports/external_references/decks")) -> Dict[str, Any]:
    cases = generate_external_reference_cases(output_dir)
    return {
        "status": "passed" if cases else "failed",
        "schema_version": 1,
        "cases": [case.to_dict() for case in cases],
        "known_limitations": [
            "Deck generation is a reproducible handoff for external validation; this report does not execute CalculiX.",
            "Beam sections are exported with a simple RECT approximation in v1.",
            "FRD/result parsing and pass/fail numerical comparison are a later batch after external solver execution is available.",
        ],
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = ["# External FE Reference Deck Report", "", f"- Status: {report.get('status')}", ""]
    lines.extend(["## Cases", ""])
    for case in report.get("cases", []):
        summary = case.get("model_summary", {})
        lines.append(f"### {case.get('name')}")
        lines.append(f"- Kind: {case.get('kind')}")
        lines.append(f"- Input: `{case.get('inp_path')}`")
        lines.append(f"- Nodes: {summary.get('nodes')}")
        lines.append(f"- Elements: {summary.get('elements')}")
        lines.append("")
    lines.extend(["## Known Limitations", ""])
    for item in report.get("known_limitations", []):
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def write_external_reference_report(
    output: Path | str = DEFAULT_EXTERNAL_REFERENCE_PATH,
    *,
    deck_dir: Path | str = Path("reports/external_references/decks"),
    markdown: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Generate decks and write a JSON/Markdown report."""

    report = generate_external_reference_report(deck_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown is not None:
        markdown_path = Path(markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_markdown(report), encoding="utf-8")
    return report
