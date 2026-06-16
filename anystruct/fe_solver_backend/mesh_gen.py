"""
Mesh generation from ANYstructure-style stiffened panel geometry.

The generated meshes are intentionally limited to the rectangular panel/stiffener
layout used by the current FE solver prototype.  The mesh builder applies support
conditions to full shell edges and uses explicit eccentric beam-shell MPC
couplings instead of shared-node or penalty placeholders.

Beam-shell coupling is shell-interpolated: each eccentric beam node is constrained
to the shell element underneath it using the shell shape functions.  This avoids
the earlier brittle requirement that each beam node must lie exactly on a shell
node row/column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .fe_core import FEModel, FEMesh, Material


@dataclass
class MeshConfig:
    """Configuration for mesh generation."""

    shell_num_divisions_x: int = 4
    shell_num_divisions_y: int = 4
    beam_num_divisions: int = 1

    use_coupling_elements: bool = True
    coupling_stiffness: float = 0.0
    use_shared_nodes: bool = False

    tolerance: float = 1.0e-6

    default_material: str = "steel"
    plate_material: str = "steel"
    stiffener_material: str = "steel"

    use_8node_shells: bool = False
    align_mesh_to_stiffeners: bool = False


@dataclass
class PanelGeometry:
    """Geometry and load metadata for a rectangular stiffened panel."""

    length: float = 0.0
    width: float = 0.0

    plate_thickness: float = 0.0
    plate_material: str = "steel"

    stiffener_type: str = "T-bar"
    stiffener_spacing: float = 0.0
    stiffener_height: float = 0.0
    stiffener_web_thickness: float = 0.0
    stiffener_flange_width: float = 0.0
    stiffener_flange_thickness: float = 0.0
    stiffener_material: str = "steel"
    num_stiffeners: int = 1

    in_plane_support: str = "Integrated"
    rotational_support: str = "SS"

    axial_stress: float = 0.0
    transverse_stress: float = 0.0
    shear_stress: float = 0.0
    pressure: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "length": self.length,
            "width": self.width,
            "plate_thickness": self.plate_thickness,
            "plate_material": self.plate_material,
            "stiffener_type": self.stiffener_type,
            "stiffener_spacing": self.stiffener_spacing,
            "stiffener_height": self.stiffener_height,
            "stiffener_web_thickness": self.stiffener_web_thickness,
            "stiffener_flange_width": self.stiffener_flange_width,
            "stiffener_flange_thickness": self.stiffener_flange_thickness,
            "stiffener_material": self.stiffener_material,
            "num_stiffeners": self.num_stiffeners,
            "in_plane_support": self.in_plane_support,
            "rotational_support": self.rotational_support,
            "axial_stress": self.axial_stress,
            "transverse_stress": self.transverse_stress,
            "shear_stress": self.shear_stress,
            "pressure": self.pressure,
        }

    @classmethod
    def from_anystructure(cls, anystructure_data: Any) -> "PanelGeometry":
        geometry = cls()
        if hasattr(anystructure_data, "Plate"):
            plate = anystructure_data.Plate
            geometry.length = getattr(plate, "span", 0.0)
            geometry.width = getattr(plate, "spacing", 0.0)
            geometry.plate_thickness = getattr(plate, "t", 0.0)
            geometry.plate_material = "steel"
        if hasattr(anystructure_data, "Stiffener"):
            stiffener = anystructure_data.Stiffener
            geometry.stiffener_type = getattr(stiffener, "stiffener_type", "T-bar")
            geometry.stiffener_spacing = getattr(stiffener, "spacing", 0.0)
            geometry.stiffener_height = getattr(stiffener, "hw", 0.0)
            geometry.stiffener_web_thickness = getattr(stiffener, "tw", 0.0)
            geometry.stiffener_flange_width = getattr(stiffener, "b", 0.0)
            geometry.stiffener_flange_thickness = getattr(stiffener, "tf", 0.0)
            geometry.stiffener_material = "steel"
        if hasattr(anystructure_data, "sigma_x1"):
            geometry.axial_stress = anystructure_data.sigma_x1
        if hasattr(anystructure_data, "sigma_y1"):
            geometry.transverse_stress = anystructure_data.sigma_y1
        if hasattr(anystructure_data, "tau_xy"):
            geometry.shear_stress = anystructure_data.tau_xy
        if hasattr(anystructure_data, "pressure"):
            geometry.pressure = anystructure_data.pressure
        return geometry


@dataclass
class StiffenerCrossSection:
    """Cross-section properties for a line stiffener."""

    area: float
    Iy: float
    Iz: float
    J: float
    shear_factor_y: float = 5.0 / 6.0
    shear_factor_z: float = 5.0 / 6.0
    c_y: float = 0.0
    c_z: float = 0.0
    torsion_modulus: float = 0.0

    @staticmethod
    def _composite_rectangles(
        rectangles: List[Tuple[float, float, float, float]],
    ) -> Tuple[float, float, float, float, float]:
        """
        Return A, Iy, Iz, c_y, c_z for rectangles described by
        y, z, width_y, height_z, with c_y/c_z the extreme fiber distances
        from the centroid.
        """
        areas = np.asarray([width * height for _y, _z, width, height in rectangles], dtype=float)
        total_area = max(float(np.sum(areas)), 1.0e-30)
        y_centroid = float(
            np.sum([area * rect[0] for area, rect in zip(areas, rectangles)]) / total_area
        )
        z_centroid = float(
            np.sum([area * rect[1] for area, rect in zip(areas, rectangles)]) / total_area
        )

        Iy = 0.0
        Iz = 0.0
        c_y = 0.0
        c_z = 0.0
        for area, (y, z, width, height) in zip(areas, rectangles):
            Iy += width * height**3 / 12.0 + area * (z - z_centroid) ** 2
            Iz += height * width**3 / 12.0 + area * (y - y_centroid) ** 2
            c_y = max(c_y, abs(y - y_centroid) + width / 2.0)
            c_z = max(c_z, abs(z - z_centroid) + height / 2.0)
        return total_area, Iy, Iz, c_y, c_z

    @classmethod
    def from_geometry(cls, stiffener_type: str, hw: float, tw: float, b: float, tf: float) -> "StiffenerCrossSection":
        # Open thin-walled torsion: J = sum(l*t^3)/3 and tau_max = T*t_max/J,
        # so the torsional section modulus is Wt = J / t_max.
        if stiffener_type == "T-bar":
            A, Iy, Iz, c_y, c_z = cls._composite_rectangles(
                [
                    (0.0, hw / 2.0, tw, hw),
                    (0.0, hw + tf / 2.0, b, tf),
                ]
            )
            J = (hw * tw**3 + b * tf**3) / 3.0
            t_max = max(tw, tf)
        elif stiffener_type in ("L-bulb", "Angle"):
            A, Iy, Iz, c_y, c_z = cls._composite_rectangles(
                [
                    (tw / 2.0, hw / 2.0, tw, hw),
                    (b / 2.0, hw + tf / 2.0, b, tf),
                ]
            )
            J = (hw * tw**3 + b * tf**3) / 3.0
            t_max = max(tw, tf)
        elif stiffener_type == "Flatbar":
            A, Iy, Iz, c_y, c_z = cls._composite_rectangles([(0.0, 0.0, b, tf)])
            J = b * tf**3 / 3.0
            t_max = min(b, tf)
        else:
            A, Iy, Iz, c_y, c_z = cls._composite_rectangles([(0.0, hw / 2.0, tw, hw)])
            J = hw * tw**3 / 3.0
            t_max = tw
        torsion_modulus = J / max(t_max, 1.0e-30)
        return cls(area=A, Iy=Iy, Iz=Iz, J=J, c_y=c_y, c_z=c_z, torsion_modulus=torsion_modulus)


class InterpolatedBeamShellMPCElement:
    """
    Duck-typed MPC-only element for eccentric beam-to-shell coupling.

    The first node is the beam slave node.  The remaining nodes are the shell
    master nodes of the shell element underneath the beam node.  Shape weights
    are evaluated at the projected beam-node location in the shell element.
    """

    def __init__(
        self,
        element_id: int,
        beam_node_id: int,
        shell_node_ids: List[int],
        shape_weights: np.ndarray,
        eccentricity: np.ndarray,
        material_name: str = "steel",
    ):
        self.element_id = element_id
        self.beam_node_id = beam_node_id
        self.shell_node_ids = list(shell_node_ids)
        self.shape_weights = np.asarray(shape_weights, dtype=float)
        self.eccentricity = np.asarray(eccentricity, dtype=float)
        self.material_name = material_name
        self.node_ids = [beam_node_id] + self.shell_node_ids
        self._stiffness_matrix = None
        self._mass_matrix = None

    @property
    def num_nodes(self) -> int:
        return 1 + len(self.shell_node_ids)

    @property
    def dofs_per_node(self) -> int:
        return 6

    @property
    def total_dofs(self) -> int:
        return self.num_nodes * self.dofs_per_node

    def get_node_coordinates(self, mesh: "FEMesh") -> np.ndarray:
        coords = []
        for node_id in self.node_ids:
            node = mesh.get_node(node_id)
            if node is None:
                raise ValueError(f"MPC element {self.element_id} references missing node {node_id}")
            coords.append(node.coords())
        return np.asarray(coords, dtype=float)

    def get_dof_mapping(self, mesh: "FEMesh") -> List[int]:
        dofs: List[int] = []
        for node_id in self.node_ids:
            node = mesh.get_node(node_id)
            if node is not None:
                dofs.extend(node.dofs)
        return dofs

    def compute_stiffness_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        # Constraint is enforced exactly by assembly.build_constraint_transformation().
        K = np.zeros((self.total_dofs, self.total_dofs), dtype=float)
        self._stiffness_matrix = K
        return K

    def compute_mass_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        return np.zeros((self.total_dofs, self.total_dofs), dtype=float)

    def compute_geometric_stiffness_matrix(self, mesh: "FEMesh", material: "Material", state: Any = None) -> np.ndarray:
        return np.zeros((self.total_dofs, self.total_dofs), dtype=float)

    def compute_stresses(
        self,
        mesh: "FEMesh",
        displacements: np.ndarray,
        material: "Material",
        return_global: bool = False,
    ) -> Dict[str, np.ndarray]:
        return {}

    def get_mpc_constraints(self, mesh: "FEMesh") -> List[Dict[str, Any]]:
        beam_node = mesh.get_node(self.beam_node_id)
        if beam_node is None:
            return []

        beam_dofs = beam_node.dofs
        rx, ry, rz = self.eccentricity
        translational_masters = [{}, {}, {}]
        rotational_masters = [{}, {}, {}]

        for shell_node_id, weight in zip(self.shell_node_ids, self.shape_weights):
            shell_node = mesh.get_node(shell_node_id)
            if shell_node is None or abs(float(weight)) == 0.0:
                continue
            s = shell_node.dofs
            w = float(weight)

            translational_masters[0][s[0]] = translational_masters[0].get(s[0], 0.0) + w
            translational_masters[0][s[4]] = translational_masters[0].get(s[4], 0.0) + w * rz
            translational_masters[0][s[5]] = translational_masters[0].get(s[5], 0.0) - w * ry

            translational_masters[1][s[1]] = translational_masters[1].get(s[1], 0.0) + w
            translational_masters[1][s[3]] = translational_masters[1].get(s[3], 0.0) - w * rz
            translational_masters[1][s[5]] = translational_masters[1].get(s[5], 0.0) + w * rx

            translational_masters[2][s[2]] = translational_masters[2].get(s[2], 0.0) + w
            translational_masters[2][s[3]] = translational_masters[2].get(s[3], 0.0) + w * ry
            translational_masters[2][s[4]] = translational_masters[2].get(s[4], 0.0) - w * rx

            rotational_masters[0][s[3]] = rotational_masters[0].get(s[3], 0.0) + w
            rotational_masters[1][s[4]] = rotational_masters[1].get(s[4], 0.0) + w
            rotational_masters[2][s[5]] = rotational_masters[2].get(s[5], 0.0) + w

        return [
            {"slave": beam_dofs[0], "masters": translational_masters[0], "value": 0.0, "label": f"interp_beam_shell_ux_{self.element_id}"},
            {"slave": beam_dofs[1], "masters": translational_masters[1], "value": 0.0, "label": f"interp_beam_shell_uy_{self.element_id}"},
            {"slave": beam_dofs[2], "masters": translational_masters[2], "value": 0.0, "label": f"interp_beam_shell_uz_{self.element_id}"},
            {"slave": beam_dofs[3], "masters": rotational_masters[0], "value": 0.0, "label": f"interp_beam_shell_rx_{self.element_id}"},
            {"slave": beam_dofs[4], "masters": rotational_masters[1], "value": 0.0, "label": f"interp_beam_shell_ry_{self.element_id}"},
            {"slave": beam_dofs[5], "masters": rotational_masters[2], "value": 0.0, "label": f"interp_beam_shell_rz_{self.element_id}"},
        ]


class RigidLidMPCElement:
    """
    Constraint-only rigid end diaphragm.

    The element ties an end ring to a free center reference node using rigid-body
    kinematics. It adds end-ring coupling without adding lid shell elements, so
    lid stresses and lid pressure loads are not recovered.
    """

    def __init__(
        self,
        element_id: int,
        center_node_id: int,
        ring_node_ids: List[int],
        material_name: str = "steel",
    ):
        self.element_id = int(element_id)
        self.center_node_id = int(center_node_id)
        self.ring_node_ids = [int(node_id) for node_id in ring_node_ids if int(node_id) != int(center_node_id)]
        self.material_name = material_name
        self.node_ids = [self.center_node_id] + self.ring_node_ids

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def dofs_per_node(self) -> int:
        return 6

    @property
    def total_dofs(self) -> int:
        return self.num_nodes * self.dofs_per_node

    def get_node_coordinates(self, mesh: "FEMesh") -> np.ndarray:
        coords = []
        for node_id in self.node_ids:
            node = mesh.get_node(node_id)
            if node is None:
                raise ValueError(f"Rigid lid element {self.element_id} references missing node {node_id}")
            coords.append(node.coords())
        return np.asarray(coords, dtype=float)

    def get_dof_mapping(self, mesh: "FEMesh") -> List[int]:
        return []

    def compute_stiffness_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        return np.zeros((0, 0), dtype=float)

    def compute_mass_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        return np.zeros((0, 0), dtype=float)

    def compute_geometric_stiffness_matrix(self, mesh: "FEMesh", material: "Material", state: Any = None) -> np.ndarray:
        return np.zeros((0, 0), dtype=float)

    def compute_stresses(
        self,
        mesh: "FEMesh",
        displacements: np.ndarray,
        material: "Material",
        return_global: bool = False,
    ) -> Dict[str, np.ndarray]:
        return {}

    @staticmethod
    def _nonzero_masters(items: Dict[int, float]) -> Dict[int, float]:
        return {int(dof): float(value) for dof, value in items.items() if abs(float(value)) > 0.0}

    def get_mpc_constraints(self, mesh: "FEMesh") -> List[Dict[str, Any]]:
        center = mesh.get_node(self.center_node_id)
        if center is None:
            return []
        center_dofs = center.dofs
        constraints: List[Dict[str, Any]] = []
        for ring_node_id in self.ring_node_ids:
            node = mesh.get_node(ring_node_id)
            if node is None:
                continue
            rx, ry, rz = (node.coords() - center.coords()).tolist()
            node_dofs = node.dofs
            translation_masters = (
                {center_dofs[0]: 1.0, center_dofs[4]: rz, center_dofs[5]: -ry},
                {center_dofs[1]: 1.0, center_dofs[3]: -rz, center_dofs[5]: rx},
                {center_dofs[2]: 1.0, center_dofs[3]: ry, center_dofs[4]: -rx},
            )
            for local_index, masters in enumerate(translation_masters):
                constraints.append(
                    {
                        "slave": node_dofs[local_index],
                        "masters": self._nonzero_masters(masters),
                        "value": 0.0,
                        "label": f"rigid_lid_{self.element_id}_u{local_index + 1}",
                    }
                )
            for local_index in range(3, 6):
                constraints.append(
                    {
                        "slave": node_dofs[local_index],
                        "masters": {center_dofs[local_index]: 1.0},
                        "value": 0.0,
                        "label": f"rigid_lid_{self.element_id}_r{local_index - 2}",
                    }
                )
        return constraints


def _safe_divisions(value: int) -> int:
    return max(int(value), 1)


def generate_stiffened_panel_mesh(panel: PanelGeometry, config: Optional[MeshConfig] = None) -> "FEModel":
    """Generate a rectangular stiffened panel mesh."""
    from .elements import BeamElement, QuadraticBeamElement, ShellElement
    from .fe_core import FEModel

    config = config or MeshConfig()
    if config.use_shared_nodes:
        raise ValueError(
            "use_shared_nodes=True is no longer supported for eccentric stiffeners. "
            "Use separate beam nodes with interpolated beam-shell MPC constraints instead."
        )

    model = FEModel(name=f"StiffenedPanel_{panel.length}x{panel.width}")
    model.add_material(
        name="steel",
        elastic_modulus=210.0e9,
        poisson_ratio=0.3,
        density=7850.0,
        yield_stress=235.0e6,
    )
    model.current_material = "steel"

    shell_nodes, shell_elements = _generate_shell_mesh(panel, config)
    beam_nodes, beam_elements = _generate_beam_mesh(panel, config)

    all_nodes = {**shell_nodes, **beam_nodes}
    for node_id, coords in all_nodes.items():
        model.add_node(node_id, coords[0], coords[1], coords[2])

    for elem_id, (node_ids, thickness) in shell_elements.items():
        model.add_element(elem_id, ShellElement(elem_id, node_ids, material_name="steel", thickness=thickness))

    for elem_id, (node_ids, element_data) in beam_elements.items():
        if len(node_ids) == 3:
            eccentricity = element_data.get("eccentricity", None)
            cross_section = {k: v for k, v in element_data.items() if k != "eccentricity"}
            element = QuadraticBeamElement(elem_id, node_ids, material_name="steel", cross_section=cross_section, eccentricity=eccentricity)
        else:
            cross_section = {k: v for k, v in element_data.items() if k != "eccentricity"}
            element = BeamElement(elem_id, node_ids, material_name="steel", cross_section=cross_section)
        model.add_element(elem_id, element)

    if config.use_coupling_elements:
        coupling_elements = _generate_coupling_elements(panel, config, shell_nodes, shell_elements, beam_nodes)
        if len(coupling_elements) != len(beam_nodes):
            raise ValueError(
                f"Only generated {len(coupling_elements)} beam-shell MPC couplings for {len(beam_nodes)} beam nodes. "
                "Check stiffener positions and panel dimensions."
            )
        for elem_id, data in coupling_elements.items():
            beam_node_id, shell_node_ids, shape_weights, eccentricity = data
            model.add_element(
                elem_id,
                InterpolatedBeamShellMPCElement(
                    elem_id,
                    beam_node_id=beam_node_id,
                    shell_node_ids=shell_node_ids,
                    shape_weights=shape_weights,
                    eccentricity=eccentricity,
                    material_name="steel",
                ),
            )

    _add_boundary_conditions(model, panel, shell_nodes, config)
    return model



def _generate_shell_mesh(
    panel: PanelGeometry,
    config: MeshConfig,
) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[int, Tuple[List[int], float]]]:
    """Generate 4-node or 8-node quadrilateral shell mesh for the plating."""
    nodes: Dict[int, Tuple[float, float, float]] = {}
    elements: Dict[int, Tuple[List[int], float]] = {}
    nx = _safe_divisions(config.shell_num_divisions_x)
    ny = _safe_divisions(config.shell_num_divisions_y)
    L = panel.length
    W = panel.width
    t = panel.plate_thickness

    if config.align_mesh_to_stiffeners:
        num_stiffeners = max(int(panel.num_stiffeners), 1)
        spacing = panel.stiffener_spacing if panel.stiffener_spacing > 0.0 else panel.width / (num_stiffeners + 1)
        y_stiffeners = [(s + 1) * spacing for s in range(num_stiffeners)]
        key_ys = [0.0] + y_stiffeners + [W]
        n_segments = len(key_ys) - 1

        if ny <= n_segments:
            segment_divs = [1] * n_segments
            ny = n_segments
        else:
            segment_divs = [1] * n_segments
            remaining = ny - n_segments
            widths = np.array([key_ys[k+1] - key_ys[k] for k in range(n_segments)], dtype=float)
            shares = widths / np.sum(widths) * remaining
            floored = np.floor(shares).astype(int)
            segment_divs = [divs + f for divs, f in zip(segment_divs, floored)]
            remaining -= np.sum(floored)
            fractional = shares - floored
            for idx in np.argsort(fractional)[::-1][:remaining]:
                segment_divs[idx] += 1
            ny = sum(segment_divs)

        y_grid = []
        for k in range(n_segments):
            y0_seg = key_ys[k]
            y1_seg = key_ys[k+1]
            divs = segment_divs[k]
            for d in range(divs):
                y_grid.append(y0_seg + d * (y1_seg - y0_seg) / divs)
        y_grid.append(W)
    else:
        y_grid = [j * W / ny for j in range(ny + 1)]

    if config.use_8node_shells:
        node_id = 1
        corner_nodes: Dict[Tuple[int, int], int] = {}
        for i in range(nx + 1):
            for j in range(ny + 1):
                corner_nodes[(i, j)] = node_id
                nodes[node_id] = (i * L / nx, y_grid[j], 0.0)
                node_id += 1
        h_mid_nodes: Dict[Tuple[int, int], int] = {}
        for i in range(nx):
            for j in range(ny + 1):
                h_mid_nodes[(i, j)] = node_id
                nodes[node_id] = ((i + 0.5) * L / nx, y_grid[j], 0.0)
                node_id += 1
        v_mid_nodes: Dict[Tuple[int, int], int] = {}
        for i in range(nx + 1):
            for j in range(ny):
                v_mid_nodes[(i, j)] = node_id
                nodes[node_id] = (i * L / nx, 0.5 * (y_grid[j] + y_grid[j + 1]), 0.0)
                node_id += 1
        elem_id = 1
        for i in range(nx):
            for j in range(ny):
                elements[elem_id] = (
                    [
                        corner_nodes[(i, j)],
                        corner_nodes[(i + 1, j)],
                        corner_nodes[(i + 1, j + 1)],
                        corner_nodes[(i, j + 1)],
                        h_mid_nodes[(i, j)],
                        v_mid_nodes[(i + 1, j)],
                        h_mid_nodes[(i, j + 1)],
                        v_mid_nodes[(i, j)],
                    ],
                    t,
                )
                elem_id += 1
    else:
        node_id = 1
        node_grid: Dict[Tuple[int, int], int] = {}
        for i in range(nx + 1):
            for j in range(ny + 1):
                node_grid[(i, j)] = node_id
                nodes[node_id] = (i * L / nx, y_grid[j], 0.0)
                node_id += 1
        elem_id = 1
        for i in range(nx):
            for j in range(ny):
                elements[elem_id] = (
                    [node_grid[(i, j)], node_grid[(i + 1, j)], node_grid[(i + 1, j + 1)], node_grid[(i, j + 1)]],
                    t,
                )
                elem_id += 1
    return nodes, elements


def _stiffener_cross_section_dict(panel: PanelGeometry) -> Dict[str, float]:
    cross_section = StiffenerCrossSection.from_geometry(
        panel.stiffener_type,
        panel.stiffener_height,
        panel.stiffener_web_thickness,
        panel.stiffener_flange_width,
        panel.stiffener_flange_thickness,
    )
    return {
        "area": cross_section.area,
        "Iy": cross_section.Iy,
        "Iz": cross_section.Iz,
        "J": cross_section.J,
        "shear_factor_y": cross_section.shear_factor_y,
        "shear_factor_z": cross_section.shear_factor_z,
        "c_y": cross_section.c_y,
        "c_z": cross_section.c_z,
        "torsion_modulus": cross_section.torsion_modulus,
        # Section local z (web direction): the panel lies in the global z=0
        # plane with stiffener webs pointing in +Z.  This pins the beam local
        # frame so Iy/Iz keep the meaning used by StiffenerCrossSection.
        "orientation": (0.0, 0.0, 1.0),
    }


def _generate_beam_mesh(
    panel: PanelGeometry,
    config: MeshConfig,
) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[int, Tuple[List[int], Dict[str, float]]]]:
    """Generate separate beam nodes/elements for longitudinal stiffeners."""
    nodes: Dict[int, Tuple[float, float, float]] = {}
    elements: Dict[int, Tuple[List[int], Dict[str, float]]] = {}
    cross_section = _stiffener_cross_section_dict(panel)
    n_div = _safe_divisions(config.beam_num_divisions)
    num_stiffeners = max(int(panel.num_stiffeners), 1)
    spacing = panel.stiffener_spacing if panel.stiffener_spacing > 0.0 else panel.width / (num_stiffeners + 1)
    eccentricity = np.array([0.0, 0.0, panel.stiffener_height], dtype=float)

    start_node_id = 10000
    start_elem_id = 20000
    for s in range(num_stiffeners):
        y_pos = (s + 1) * spacing
        for i in range(n_div + 1):
            node_id = start_node_id + s * (n_div + 1) + i
            nodes[node_id] = (i * panel.length / n_div, y_pos, panel.stiffener_height)
        for i in range(n_div):
            n1 = start_node_id + s * (n_div + 1) + i
            n2 = n1 + 1
            elem_id = start_elem_id + s * n_div + i
            data = dict(cross_section)
            data["eccentricity"] = eccentricity
            elements[elem_id] = ([n1, n2], data)
    return nodes, elements


def _generate_beam_mesh_shared_nodes(
    panel: PanelGeometry,
    config: MeshConfig,
    shell_nodes: Dict[int, Tuple[float, float, float]],
) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[int, Tuple[List[int], Dict[str, Any]]]]:
    """Deprecated helper kept only for import compatibility."""
    raise ValueError("Shared-node stiffener mesh is disabled; use interpolated eccentric MPC coupling instead.")


def _shape_functions_4node(xi: float, eta: float) -> np.ndarray:
    return np.array(
        [
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta),
        ],
        dtype=float,
    )


def _shape_functions_8node(xi: float, eta: float) -> np.ndarray:
    return np.array(
        [
            -0.25 * (1.0 - xi) * (1.0 - eta) * (1.0 + xi + eta),
            -0.25 * (1.0 + xi) * (1.0 - eta) * (1.0 - xi + eta),
            -0.25 * (1.0 + xi) * (1.0 + eta) * (1.0 - xi - eta),
            -0.25 * (1.0 - xi) * (1.0 + eta) * (1.0 + xi - eta),
            0.5 * (1.0 - xi**2) * (1.0 - eta),
            0.5 * (1.0 + xi) * (1.0 - eta**2),
            0.5 * (1.0 - xi**2) * (1.0 + eta),
            0.5 * (1.0 - xi) * (1.0 - eta**2),
        ],
        dtype=float,
    )


def _locate_shell_element_at_xy(
    x: float,
    y: float,
    shell_nodes: Dict[int, Tuple[float, float, float]],
    shell_elements: Dict[int, Tuple[List[int], float]],
    tolerance: float,
) -> Optional[Tuple[List[int], np.ndarray, np.ndarray]]:
    """Find the shell element containing an x/y point and return node ids, weights and shell point."""
    tol = max(float(tolerance), 1.0e-10)

    # Fast O(1) grid lookup path for regular rectangular grids
    try:
        xs = sorted(list(set(round(coords[0] / tol) * tol for coords in shell_nodes.values())))
        ys = sorted(list(set(round(coords[1] / tol) * tol for coords in shell_nodes.values())))
        xs_arr = np.array(xs, dtype=float)
        ys_arr = np.array(ys, dtype=float)
        nx = len(xs_arr) - 1
        ny = len(ys_arr) - 1

        if nx > 0 and ny > 0 and len(shell_elements) == nx * ny:
            i = np.searchsorted(xs_arr, x) - 1
            j = np.searchsorted(ys_arr, y) - 1
            i = max(0, min(i, nx - 1))
            j = max(0, min(j, ny - 1))

            candidate_id = 1 + i * ny + j
            if candidate_id in shell_elements:
                node_ids, _thickness = shell_elements[candidate_id]
                corner_ids = node_ids[:4]
                corner_coords = np.asarray([shell_nodes[nid] for nid in corner_ids], dtype=float)
                xmin, xmax = float(np.min(corner_coords[:, 0])), float(np.max(corner_coords[:, 0]))
                ymin, ymax = float(np.min(corner_coords[:, 1])), float(np.max(corner_coords[:, 1]))

                if xmin - tol <= x <= xmax + tol and ymin - tol <= y <= ymax + tol:
                    dx = xmax - xmin
                    dy = ymax - ymin
                    if dx > tol and dy > tol:
                        xi = 2.0 * (x - xmin) / dx - 1.0
                        eta = 2.0 * (y - ymin) / dy - 1.0
                        xi = float(np.clip(xi, -1.0, 1.0))
                        eta = float(np.clip(eta, -1.0, 1.0))
                        weights = _shape_functions_8node(xi, eta) if len(node_ids) == 8 else _shape_functions_4node(xi, eta)
                        shell_coords = np.asarray([shell_nodes[nid] for nid in node_ids], dtype=float)
                        shell_point = weights @ shell_coords
                        return list(node_ids), weights, shell_point
    except Exception:
        pass

    # Fallback to sequential search
    for node_ids, _thickness in shell_elements.values():
        corner_ids = node_ids[:4]
        corner_coords = np.asarray([shell_nodes[nid] for nid in corner_ids], dtype=float)
        xmin, xmax = float(np.min(corner_coords[:, 0])), float(np.max(corner_coords[:, 0]))
        ymin, ymax = float(np.min(corner_coords[:, 1])), float(np.max(corner_coords[:, 1]))
        if x < xmin - tol or x > xmax + tol or y < ymin - tol or y > ymax + tol:
            continue
        dx = xmax - xmin
        dy = ymax - ymin
        if abs(dx) <= tol or abs(dy) <= tol:
            continue
        xi = 2.0 * (x - xmin) / dx - 1.0
        eta = 2.0 * (y - ymin) / dy - 1.0
        xi = float(np.clip(xi, -1.0, 1.0))
        eta = float(np.clip(eta, -1.0, 1.0))
        weights = _shape_functions_8node(xi, eta) if len(node_ids) == 8 else _shape_functions_4node(xi, eta)
        shell_coords = np.asarray([shell_nodes[nid] for nid in node_ids], dtype=float)
        shell_point = weights @ shell_coords
        return list(node_ids), weights, shell_point
    return None


def _generate_coupling_elements(
    panel: PanelGeometry,
    config: MeshConfig,
    shell_nodes: Dict[int, Tuple[float, float, float]],
    shell_elements: Dict[int, Tuple[List[int], float]],
    beam_nodes: Dict[int, Tuple[float, float, float]],
) -> Dict[int, Tuple[int, List[int], np.ndarray, np.ndarray]]:
    """Generate interpolated eccentric beam-shell MPC coupling elements."""
    coupling_elements: Dict[int, Tuple[int, List[int], np.ndarray, np.ndarray]] = {}
    elem_id = 30000
    for beam_node_id, beam_coords_tuple in beam_nodes.items():
        beam_coords = np.asarray(beam_coords_tuple, dtype=float)
        located = _locate_shell_element_at_xy(beam_coords[0], beam_coords[1], shell_nodes, shell_elements, config.tolerance)
        if located is None:
            continue
        shell_node_ids, shape_weights, shell_point = located
        eccentricity = beam_coords - shell_point
        coupling_elements[elem_id] = (beam_node_id, shell_node_ids, shape_weights, eccentricity)
        elem_id += 1
    return coupling_elements


def _edge_node_sets(
    panel: PanelGeometry,
    nodes: Dict[int, Tuple[float, float, float]],
    tolerance: float,
) -> Dict[str, List[int]]:
    """Return all shell nodes lying on the four rectangular panel edges."""
    L = panel.length
    W = panel.width
    tol = max(float(tolerance), 1.0e-9, 1.0e-8 * max(abs(L), abs(W), 1.0))
    edge_nodes = {"x0": [], "xL": [], "y0": [], "yW": []}
    for node_id, coords in nodes.items():
        x, y, _ = coords
        if abs(x) <= tol:
            edge_nodes["x0"].append(node_id)
        if abs(x - L) <= tol:
            edge_nodes["xL"].append(node_id)
        if abs(y) <= tol:
            edge_nodes["y0"].append(node_id)
        if abs(y - W) <= tol:
            edge_nodes["yW"].append(node_id)
    edge_nodes["x0"].sort(key=lambda nid: nodes[nid][1])
    edge_nodes["xL"].sort(key=lambda nid: nodes[nid][1])
    edge_nodes["y0"].sort(key=lambda nid: nodes[nid][0])
    edge_nodes["yW"].sort(key=lambda nid: nodes[nid][0])
    edge_nodes["all"] = sorted(set(edge_nodes["x0"] + edge_nodes["xL"] + edge_nodes["y0"] + edge_nodes["yW"]))
    return edge_nodes


def _unique_node_ids(*node_lists: List[int]) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for node_list in node_lists:
        for node_id in node_list:
            if node_id not in seen:
                seen.add(node_id)
                ordered.append(node_id)
    return ordered


def _add_custom_support(model: "FEModel", name: str, node_ids: List[int], dof_constraints: Dict[str, float]) -> None:
    from .boundary import BoundaryCondition

    node_ids = _unique_node_ids(node_ids)
    if node_ids:
        model.add_boundary_condition(BoundaryCondition(name, node_ids, dof_constraints))


def _add_boundary_conditions(
    model: "FEModel",
    panel: PanelGeometry,
    shell_nodes: Dict[int, Tuple[float, float, float]],
    config: Optional[MeshConfig] = None,
) -> None:
    """Add edge-based support conditions to the model."""
    config = config or MeshConfig()
    edges = _edge_node_sets(panel, shell_nodes, config.tolerance)
    support = (panel.in_plane_support or "").strip().lower()
    rotational = (panel.rotational_support or "").strip().upper()

    longitudinal_edges = _unique_node_ids(edges["y0"], edges["yW"])
    transverse_edges = _unique_node_ids(edges["x0"], edges["xL"])
    all_edge_nodes = edges["all"]

    if support == "integrated":
        _add_custom_support(model, "Integrated_edge_translations", all_edge_nodes, {"ux": 0.0, "uy": 0.0, "uz": 0.0})
    elif support == "girder - long":
        _add_custom_support(model, "Longitudinal_girder_edges", longitudinal_edges, {"ux": 0.0, "uy": 0.0, "uz": 0.0})
        _add_custom_support(model, "Transverse_reference_ux", edges["x0"][:1], {"ux": 0.0})
    elif support == "girder - trans":
        _add_custom_support(model, "Transverse_girder_edges", transverse_edges, {"ux": 0.0, "uy": 0.0, "uz": 0.0})
        _add_custom_support(model, "Longitudinal_reference_uy", edges["y0"][:1], {"uy": 0.0})
    else:
        _add_custom_support(model, "Reference_out_of_plane", all_edge_nodes[:1], {"uz": 0.0})

    if rotational == "CL":
        _add_custom_support(model, "Clamped_edge_rotations", all_edge_nodes, {"rx": 0.0, "ry": 0.0, "rz": 0.0})
    elif rotational == "FS":
        _add_custom_support(model, "Fixed_simple_longitudinal_rotations", longitudinal_edges, {"rx": 0.0, "ry": 0.0, "rz": 0.0})


def generate_simple_panel_mesh(
    length: float,
    width: float,
    thickness: float,
    num_divisions_x: int = 4,
    num_divisions_y: int = 4,
    use_8node_elements: bool = False,
) -> "FEModel":
    """Generate a simple rectangular shell panel mesh for testing."""
    from .elements import ShellElement
    from .fe_core import FEModel

    config = MeshConfig(shell_num_divisions_x=num_divisions_x, shell_num_divisions_y=num_divisions_y, use_8node_shells=use_8node_elements)
    panel = PanelGeometry(length=length, width=width, plate_thickness=thickness)
    shell_nodes, shell_elements = _generate_shell_mesh(panel, config)
    model = FEModel(name=f"SimplePanel_{length}x{width}")
    model.add_material("steel", 210.0e9, 0.3)
    model.current_material = "steel"
    for node_id, coords in shell_nodes.items():
        model.add_node(node_id, coords[0], coords[1], coords[2])
    for elem_id, (node_ids, elem_thickness) in shell_elements.items():
        model.add_element(elem_id, ShellElement(elem_id, node_ids, material_name="steel", thickness=elem_thickness))
    _add_boundary_conditions(model, panel, shell_nodes, config)
    return model


def generate_beam_mesh(
    length: float,
    num_divisions: int = 10,
    cross_section: Optional[Dict[str, float]] = None,
) -> "FEModel":
    """Generate a simple cantilever beam mesh."""
    from .boundary import FixedSupport
    from .elements import BeamElement
    from .fe_core import FEModel

    model = FEModel(name=f"SimpleBeam_{length}")
    model.add_material("steel", 210.0e9, 0.3)
    model.current_material = "steel"
    if cross_section is None:
        cross_section = {"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}
    n_div = _safe_divisions(num_divisions)
    for i in range(n_div + 1):
        model.add_node(i + 1, i * length / n_div, 0.0, 0.0)
    for i in range(n_div):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], material_name="steel", cross_section=cross_section))
    model.add_boundary_condition(FixedSupport("Fixed_1", [1]))
    return model


def verify_mesh_quality(model: "FEModel") -> Dict[str, Any]:
    """Analyze and verify mesh quality metrics (aspect ratio, warping)."""
    from .elements import ShellElement

    aspect_ratios = []
    warps = []
    shell_count = 0

    for elem in model.mesh.elements.values():
        if not isinstance(elem, ShellElement):
            continue
        shell_count += 1
        coords = elem.get_node_coordinates(model.mesh)
        corner_coords = coords[:4]
        
        e1 = corner_coords[1] - corner_coords[0]
        e2 = corner_coords[2] - corner_coords[1]
        e3 = corner_coords[3] - corner_coords[2]
        e4 = corner_coords[0] - corner_coords[3]
        
        l1 = float(np.linalg.norm(e1))
        l2 = float(np.linalg.norm(e2))
        l3 = float(np.linalg.norm(e3))
        l4 = float(np.linalg.norm(e4))
        
        lengths = [l1, l2, l3, l4]
        max_l = max(lengths)
        min_l = min(lengths)
        
        ar = max_l / max(min_l, 1.0e-15)
        aspect_ratios.append(ar)
        
        # Calculate warp
        n_raw = np.cross(e1, corner_coords[2] - corner_coords[0])
        n_norm = np.linalg.norm(n_raw)
        if n_norm > 1.0e-15:
            n = n_raw / n_norm
            d = abs(float(np.dot(corner_coords[3] - corner_coords[0], n)))
            avg_l = sum(lengths) / 4.0
            warp = d / max(avg_l, 1.0e-15)
        else:
            warp = 0.0
        warps.append(warp)

    warnings = []
    max_ar = float(np.max(aspect_ratios)) if aspect_ratios else 1.0
    mean_ar = float(np.mean(aspect_ratios)) if aspect_ratios else 1.0
    max_warp = float(np.max(warps)) if warps else 0.0

    if max_ar > 5.0:
        warnings.append(
            f"High aspect ratio detected (max AR = {max_ar:.2f}). "
            "Highly stretched elements can reduce solver accuracy. Consider refining the mesh divisions."
        )
    if max_warp > 0.05:
        warnings.append(
            f"Significant element warp detected (max warp = {max_warp:.4f}). "
            "Warped shell elements can lose accuracy. Ensure plate geometries are flat or sufficiently refined."
        )

    return {
        "num_shell_elements": shell_count,
        "max_aspect_ratio": max_ar,
        "mean_aspect_ratio": mean_ar,
        "max_warp": max_warp,
        "warnings": warnings,
    }
