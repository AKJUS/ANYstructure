"""Runtime load-resultant helpers for ANYstructure FE solver backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .boundary import LoadCase
    from .fe_core import FEModel


@dataclass(frozen=True)
class LoadResultant:
    """Global resultant force and moment of a load vector."""

    force: np.ndarray
    moment: np.ndarray

    @property
    def force_norm(self) -> float:
        return float(np.linalg.norm(self.force))

    @property
    def moment_norm(self) -> float:
        return float(np.linalg.norm(self.moment))


def load_vector_resultant(model: "FEModel", load_vector: np.ndarray) -> LoadResultant:
    """Compute global force and moment resultants from a full load vector."""

    load_vector = np.asarray(load_vector, dtype=float).reshape(-1)
    force = np.zeros(3, dtype=float)
    moment = np.zeros(3, dtype=float)
    for node in model.mesh.nodes.values():
        node_force = load_vector[node.dofs[:3]]
        node_moment = load_vector[node.dofs[3:6]]
        r = node.coords()
        force += node_force
        moment += np.cross(r, node_force) + node_moment
    return LoadResultant(force=force, moment=moment)


def load_case_resultant(model: "FEModel", load_case: "LoadCase") -> LoadResultant:
    """Assemble a load case and return global force/moment resultants."""

    load_vector = load_case.get_load_vector(model.mesh, model.mesh.dof_manager, model.get_material)
    return load_vector_resultant(model, load_vector)
