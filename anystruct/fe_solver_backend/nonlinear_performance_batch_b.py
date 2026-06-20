"""Batch B optimizations for persistent nonlinear shell assembly.

This module completes the nonlinear-assembly redesign for ordinary Python runs
with Numba available.  Elastic shell groups use a dedicated in-place kernel that
writes element force and tangent entries directly into the persistent assembly
plan buffers.  The plastic shell path and all element formulations remain
unchanged.

The elastic path avoids, per Newton evaluation:

* plastic-strain and hardening-state work arrays,
* through-thickness layer-strain arrays,
* broadcast constitutive tensor batches,
* per-batch force/tangent result arrays, and
* Python reconstruction of new elastic state arrays.

The kernel relies on the shell transformation's existing 3x3 block-diagonal
layout (translations and rotations for every node).  Each local 3x3 tangent
block is transformed in place, so no second dense element tangent batch is
required.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from scipy import sparse

from .jit_compiler import njit, prange
from .elements import plane_stress_elastic_matrix
from . import nonlinear_performance as _performance


@njit(cache=True, parallel=True)
def _elastic_shell_batch_into_buffers(
    displacements: np.ndarray,
    dof_mappings: np.ndarray,
    T0_batch: np.ndarray,
    B_m_batch: np.ndarray,
    B_b_batch: np.ndarray,
    B_d_batch: np.ndarray,
    Gw_batch: np.ndarray,
    detw_batch: np.ndarray,
    B_s_batch: np.ndarray,
    detw_shear_batch: np.ndarray,
    membrane_matrix: np.ndarray,
    bending_matrix: np.ndarray,
    shear_matrix: np.ndarray,
    drilling_stiffness: float,
    force_positions: np.ndarray,
    tangent_positions: np.ndarray,
    force_values: np.ndarray,
    tangent_values: np.ndarray,
    u_local_work: np.ndarray,
    tangent: bool,
) -> None:
    """Evaluate elastic geometrically nonlinear shells into persistent buffers."""
    n_elem = dof_mappings.shape[0]
    n_dof = dof_mappings.shape[1]
    n_gp = detw_batch.shape[1]
    n_shear = detw_shear_batch.shape[1]

    for element_index in prange(n_elem):
        T0 = T0_batch[element_index]
        u_local = u_local_work[element_index]

        # Gather global element DOFs and transform to the reference local frame.
        for local_row in range(n_dof):
            value = 0.0
            for local_col in range(n_dof):
                value += (
                    T0[local_row, local_col]
                    * displacements[dof_mappings[element_index, local_col]]
                )
            u_local[local_row] = value

        # The persistent plan clears the flat buffers before this kernel.  Work
        # directly in the element's assigned slices.
        force_pos = force_positions[element_index]
        tangent_pos = tangent_positions[element_index]

        B_eff = np.empty((3, n_dof), dtype=np.float64)
        membrane_times_B = np.empty((3, n_dof), dtype=np.float64)
        bending_times_B = np.empty((3, n_dof), dtype=np.float64)

        for gp_index in range(n_gp):
            B_m = B_m_batch[element_index, gp_index]
            B_b = B_b_batch[element_index, gp_index]
            B_d = B_d_batch[element_index, gp_index]
            Gw = Gw_batch[element_index, gp_index]
            detw = detw_batch[element_index, gp_index]

            theta_0 = 0.0
            theta_1 = 0.0
            membrane_strain_0 = 0.0
            membrane_strain_1 = 0.0
            membrane_strain_2 = 0.0
            curvature_0 = 0.0
            curvature_1 = 0.0
            curvature_2 = 0.0
            drilling_rotation = 0.0

            for dof_index in range(n_dof):
                displacement = u_local[dof_index]
                theta_0 += Gw[0, dof_index] * displacement
                theta_1 += Gw[1, dof_index] * displacement
                membrane_strain_0 += B_m[0, dof_index] * displacement
                membrane_strain_1 += B_m[1, dof_index] * displacement
                membrane_strain_2 += B_m[2, dof_index] * displacement
                curvature_0 += B_b[0, dof_index] * displacement
                curvature_1 += B_b[1, dof_index] * displacement
                curvature_2 += B_b[2, dof_index] * displacement
                drilling_rotation += B_d[0, dof_index] * displacement

            membrane_strain_0 += 0.5 * theta_0 * theta_0
            membrane_strain_1 += 0.5 * theta_1 * theta_1
            membrane_strain_2 += theta_0 * theta_1

            N_0 = (
                membrane_matrix[0, 0] * membrane_strain_0
                + membrane_matrix[0, 1] * membrane_strain_1
                + membrane_matrix[0, 2] * membrane_strain_2
            )
            N_1 = (
                membrane_matrix[1, 0] * membrane_strain_0
                + membrane_matrix[1, 1] * membrane_strain_1
                + membrane_matrix[1, 2] * membrane_strain_2
            )
            N_2 = (
                membrane_matrix[2, 0] * membrane_strain_0
                + membrane_matrix[2, 1] * membrane_strain_1
                + membrane_matrix[2, 2] * membrane_strain_2
            )
            M_0 = (
                bending_matrix[0, 0] * curvature_0
                + bending_matrix[0, 1] * curvature_1
                + bending_matrix[0, 2] * curvature_2
            )
            M_1 = (
                bending_matrix[1, 0] * curvature_0
                + bending_matrix[1, 1] * curvature_1
                + bending_matrix[1, 2] * curvature_2
            )
            M_2 = (
                bending_matrix[2, 0] * curvature_0
                + bending_matrix[2, 1] * curvature_1
                + bending_matrix[2, 2] * curvature_2
            )

            for dof_index in range(n_dof):
                B_eff[0, dof_index] = B_m[0, dof_index] + theta_0 * Gw[0, dof_index]
                B_eff[1, dof_index] = B_m[1, dof_index] + theta_1 * Gw[1, dof_index]
                B_eff[2, dof_index] = (
                    B_m[2, dof_index]
                    + theta_0 * Gw[1, dof_index]
                    + theta_1 * Gw[0, dof_index]
                )

                force_values[force_pos[dof_index]] += (
                    B_eff[0, dof_index] * N_0
                    + B_eff[1, dof_index] * N_1
                    + B_eff[2, dof_index] * N_2
                    + B_b[0, dof_index] * M_0
                    + B_b[1, dof_index] * M_1
                    + B_b[2, dof_index] * M_2
                    + B_d[0, dof_index] * drilling_stiffness * drilling_rotation
                ) * detw

            if not tangent:
                continue

            for row in range(3):
                for dof_index in range(n_dof):
                    membrane_value = 0.0
                    bending_value = 0.0
                    for constitutive_index in range(3):
                        membrane_value += (
                            membrane_matrix[row, constitutive_index]
                            * B_eff[constitutive_index, dof_index]
                        )
                        bending_value += (
                            bending_matrix[row, constitutive_index]
                            * B_b[constitutive_index, dof_index]
                        )
                    membrane_times_B[row, dof_index] = membrane_value
                    bending_times_B[row, dof_index] = bending_value

            for local_row in range(n_dof):
                gw0_row = Gw[0, local_row]
                gw1_row = Gw[1, local_row]
                bd_row = B_d[0, local_row]
                for local_col in range(n_dof):
                    material_value = 0.0
                    bending_value = 0.0
                    for constitutive_index in range(3):
                        material_value += (
                            B_eff[constitutive_index, local_row]
                            * membrane_times_B[constitutive_index, local_col]
                        )
                        bending_value += (
                            B_b[constitutive_index, local_row]
                            * bending_times_B[constitutive_index, local_col]
                        )
                    geometric_value = (
                        gw0_row * (N_0 * Gw[0, local_col] + N_2 * Gw[1, local_col])
                        + gw1_row * (N_2 * Gw[0, local_col] + N_1 * Gw[1, local_col])
                    )
                    drilling_value = bd_row * drilling_stiffness * B_d[0, local_col]
                    tangent_values[
                        tangent_pos[local_row * n_dof + local_col]
                    ] += (
                        material_value
                        + bending_value
                        + geometric_value
                        + drilling_value
                    ) * detw

        for shear_index in range(n_shear):
            B_s = B_s_batch[element_index, shear_index]
            detw_shear = detw_shear_batch[element_index, shear_index]
            gamma_0 = 0.0
            gamma_1 = 0.0
            for dof_index in range(n_dof):
                gamma_0 += B_s[0, dof_index] * u_local[dof_index]
                gamma_1 += B_s[1, dof_index] * u_local[dof_index]

            shear_resultant_0 = (
                shear_matrix[0, 0] * gamma_0 + shear_matrix[0, 1] * gamma_1
            )
            shear_resultant_1 = (
                shear_matrix[1, 0] * gamma_0 + shear_matrix[1, 1] * gamma_1
            )
            for dof_index in range(n_dof):
                force_values[force_pos[dof_index]] += (
                    B_s[0, dof_index] * shear_resultant_0
                    + B_s[1, dof_index] * shear_resultant_1
                ) * detw_shear

            if tangent:
                for local_row in range(n_dof):
                    for local_col in range(n_dof):
                        shear_value = (
                            B_s[0, local_row]
                            * (
                                shear_matrix[0, 0] * B_s[0, local_col]
                                + shear_matrix[0, 1] * B_s[1, local_col]
                            )
                            + B_s[1, local_row]
                            * (
                                shear_matrix[1, 0] * B_s[0, local_col]
                                + shear_matrix[1, 1] * B_s[1, local_col]
                            )
                        )
                        tangent_values[
                            tangent_pos[local_row * n_dof + local_col]
                        ] += shear_value * detw_shear

        # Transform force blocks from local to global in place.  T0 maps global
        # displacements to local displacements, hence forces use T0.T.
        for block_start in range(0, n_dof, 3):
            local_0 = force_values[force_pos[block_start]]
            local_1 = force_values[force_pos[block_start + 1]]
            local_2 = force_values[force_pos[block_start + 2]]
            force_values[force_pos[block_start]] = (
                T0[block_start, block_start] * local_0
                + T0[block_start + 1, block_start] * local_1
                + T0[block_start + 2, block_start] * local_2
            )
            force_values[force_pos[block_start + 1]] = (
                T0[block_start, block_start + 1] * local_0
                + T0[block_start + 1, block_start + 1] * local_1
                + T0[block_start + 2, block_start + 1] * local_2
            )
            force_values[force_pos[block_start + 2]] = (
                T0[block_start, block_start + 2] * local_0
                + T0[block_start + 1, block_start + 2] * local_1
                + T0[block_start + 2, block_start + 2] * local_2
            )

        if not tangent:
            continue

        # T0 is block diagonal with 3x3 translation/rotation blocks.  Each
        # K_global block depends only on the corresponding K_local block, so it
        # can be transformed and overwritten safely without a second matrix.
        local_block = np.empty((3, 3), dtype=np.float64)
        intermediate = np.empty((3, 3), dtype=np.float64)
        global_block = np.empty((3, 3), dtype=np.float64)
        for row_block in range(0, n_dof, 3):
            for col_block in range(0, n_dof, 3):
                for row in range(3):
                    for col in range(3):
                        local_block[row, col] = tangent_values[
                            tangent_pos[
                                (row_block + row) * n_dof + col_block + col
                            ]
                        ]

                for row in range(3):
                    for col in range(3):
                        value = 0.0
                        for inner in range(3):
                            value += (
                                local_block[row, inner]
                                * T0[col_block + inner, col_block + col]
                            )
                        intermediate[row, col] = value

                for row in range(3):
                    for col in range(3):
                        value = 0.0
                        for inner in range(3):
                            value += (
                                T0[row_block + inner, row_block + row]
                                * intermediate[inner, col]
                            )
                        global_block[row, col] = value

                for row in range(3):
                    for col in range(3):
                        tangent_values[
                            tangent_pos[
                                (row_block + row) * n_dof + col_block + col
                            ]
                        ] = global_block[row, col]


_ORIGINAL_BATCH_BUILD = _performance._ShellBatchPlan.build.__func__
_ORIGINAL_PLAN_ASSEMBLE = _performance.NonlinearAssemblyPlan.assemble
_ORIGINAL_PLAN_DIAGNOSTICS = _performance.NonlinearAssemblyPlan.diagnostics
_INSTALLED = False


def _batch_b_shell_build(
    cls,
    model,
    key,
    items,
    num_layers,
):
    batch = _ORIGINAL_BATCH_BUILD(cls, model, key, items, num_layers)
    if batch.has_plasticity:
        batch._batch_b_elastic = False
        return batch

    elastic_matrix = plane_stress_elastic_matrix(
        float(batch.material.elastic_modulus),
        float(batch.material.poisson_ratio),
    )
    batch._batch_b_elastic = True
    batch._batch_b_membrane_matrix = batch.thickness * elastic_matrix
    batch._batch_b_bending_matrix = batch.thickness**3 / 12.0 * elastic_matrix
    batch._batch_b_shear_matrix = (
        float(batch.material.shear_modulus)
        * (5.0 / 6.0)
        * batch.thickness
        * np.eye(2, dtype=float)
    )
    batch._batch_b_drilling_stiffness = (
        float(batch.material.shear_modulus)
        * batch.thickness
        * batch.drilling_stabilization
    )

    # Elastic groups do not require mutable constitutive history.  Release the
    # per-element plastic work arrays allocated by the compatibility builder.
    batch.plastic_work = np.empty((0, 0, 3), dtype=float)
    batch.alpha_work = np.empty((0, 0), dtype=float)

    points_per_element = int(batch.n_gp * batch.num_layers)
    shared_plastic = np.zeros((points_per_element, 3), dtype=float)
    shared_alpha = np.zeros(points_per_element, dtype=float)
    shared_layer = np.zeros((points_per_element, 3), dtype=float)
    shared_plastic.setflags(write=False)
    shared_alpha.setflags(write=False)
    shared_layer.setflags(write=False)
    batch.elastic_states = tuple(
        {
            "plastic_strain": shared_plastic,
            "alpha": shared_alpha,
            "layer_strain": shared_layer,
        }
        for _ in batch.elements
    )
    batch._batch_b_elastic_state_mapping = {
        int(element_id): state
        for element_id, state in zip(batch.element_ids, batch.elastic_states)
    }
    return batch


def _batch_b_plan_assemble(
    self,
    displacements: np.ndarray,
    committed_states: Mapping[int, Any],
    tangent: bool = True,
) -> Tuple[np.ndarray, Optional[sparse.csr_matrix], Dict[int, Any]]:
    """Assemble with direct-buffer elastic batches and legacy plastic batches."""
    with self._lock:
        start_total = time.perf_counter()
        self.timings.calls += 1
        if tangent:
            self.timings.tangent_calls += 1
        else:
            self.timings.residual_only_calls += 1

        self.force_values.fill(0.0)
        if tangent:
            self.tangent_values.fill(0.0)
        trial_states: Dict[int, Any] = {}

        for batch in self.shell_batches:
            if getattr(batch, "_batch_b_elastic", False):
                kernel_start = time.perf_counter()
                _elastic_shell_batch_into_buffers(
                    np.asarray(displacements, dtype=float),
                    batch.dof_mappings,
                    batch.T0,
                    batch.B_m,
                    batch.B_b,
                    batch.B_d,
                    batch.Gw,
                    batch.detw,
                    batch.B_s,
                    batch.detw_shear,
                    batch._batch_b_membrane_matrix,
                    batch._batch_b_bending_matrix,
                    batch._batch_b_shear_matrix,
                    float(batch._batch_b_drilling_stiffness),
                    batch.force_positions,
                    batch.tangent_positions,
                    self.force_values,
                    self.tangent_values,
                    batch.u_work,
                    bool(tangent),
                )
                self.timings.shell_kernel_seconds += time.perf_counter() - kernel_start
                elastic_states = batch._batch_b_elastic_state_mapping
                # Preserve explicitly supplied/previous elastic states without
                # allocating a new mapping in the normal case.
                use_cached_mapping = True
                for element_id in batch.element_ids:
                    existing = committed_states.get(int(element_id))
                    if isinstance(existing, dict) and existing is not elastic_states[int(element_id)]:
                        use_cached_mapping = False
                        break
                if use_cached_mapping:
                    trial_states.update(elastic_states)
                else:
                    for element_id in batch.element_ids:
                        element_key = int(element_id)
                        existing = committed_states.get(element_key)
                        trial_states[element_key] = (
                            existing if isinstance(existing, dict) else elastic_states[element_key]
                        )
            else:
                F_batch, K_batch, batch_states, kernel_seconds = batch.evaluate(
                    displacements,
                    committed_states,
                    tangent,
                )
                self.timings.shell_kernel_seconds += kernel_seconds
                self.force_values[batch.force_positions.reshape(-1)] = np.asarray(
                    F_batch, dtype=float
                ).reshape(-1)
                if tangent and K_batch is not None:
                    self.tangent_values[batch.tangent_positions.reshape(-1)] = np.asarray(
                        K_batch, dtype=float
                    ).reshape(-1)
                trial_states.update(batch_states)

        non_shell_start = time.perf_counter()
        model = self.model
        mesh = model.mesh
        for record in self.non_shell_elements:
            material = model.get_material(record.element.material_name)
            u_element = np.asarray(displacements, dtype=float)[record.dof_mapping]
            f_element, k_element, trial_state = record.element.compute_nonlinear_response(
                mesh,
                material,
                u_element,
                committed_states.get(record.element_id),
                self.num_layers,
                tangent,
            )
            self.force_values[record.force_positions] = np.asarray(
                f_element, dtype=float
            ).reshape(-1)
            if tangent and k_element is not None:
                self.tangent_values[record.tangent_positions] = np.asarray(
                    k_element, dtype=float
                ).reshape(-1)
            if trial_state is not None:
                trial_states[record.element_id] = trial_state
        self.timings.non_shell_seconds += time.perf_counter() - non_shell_start

        scatter_start = time.perf_counter()
        force = _performance._scatter_sum(
            self.force_values,
            self.force_dofs_flat,
            self.total_dofs,
        )
        self.timings.force_scatter_seconds += time.perf_counter() - scatter_start

        tangent_matrix: Optional[sparse.csr_matrix]
        if tangent:
            scatter_start = time.perf_counter()
            csr_data = _performance._scatter_sum(
                self.tangent_values,
                self.tangent_scatter,
                self.nnz,
            )
            tangent_matrix = sparse.csr_matrix(
                (csr_data, self.csr_indices, self.csr_indptr),
                shape=(self.total_dofs, self.total_dofs),
                copy=False,
            )
            self.timings.tangent_scatter_seconds += time.perf_counter() - scatter_start
        else:
            tangent_matrix = None

        self.timings.total_seconds += time.perf_counter() - start_total
        return force, tangent_matrix, trial_states


def _batch_b_plan_diagnostics(self) -> Dict[str, Any]:
    diagnostics = _ORIGINAL_PLAN_DIAGNOSTICS(self)
    elastic_batches = [
        batch
        for batch in self.shell_batches
        if getattr(batch, "_batch_b_elastic", False)
    ]
    diagnostics.update(
        {
            "batch_b_installed": True,
            "elastic_fast_path_batch_count": len(elastic_batches),
            "elastic_fast_path_element_count": int(
                sum(batch.element_ids.size for batch in elastic_batches)
            ),
            "plastic_batch_count": int(len(self.shell_batches) - len(elastic_batches)),
            "elastic_constitutive_state_bytes": int(
                sum(
                    batch.elastic_states[0]["plastic_strain"].nbytes
                    + batch.elastic_states[0]["alpha"].nbytes
                    + batch.elastic_states[0]["layer_strain"].nbytes
                    for batch in elastic_batches
                    if batch.elastic_states
                )
            ),
        }
    )
    return diagnostics


def install_batch_b_optimizations() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    _performance._ShellBatchPlan.build = classmethod(_batch_b_shell_build)
    _performance.NonlinearAssemblyPlan.assemble = _batch_b_plan_assemble
    _performance.NonlinearAssemblyPlan.diagnostics = _batch_b_plan_diagnostics
    _performance.clear_nonlinear_assembly_cache()
    _INSTALLED = True
    return True


def uninstall_batch_b_optimizations() -> None:
    global _INSTALLED
    if not _INSTALLED:
        return
    _performance._ShellBatchPlan.build = classmethod(_ORIGINAL_BATCH_BUILD)
    _performance.NonlinearAssemblyPlan.assemble = _ORIGINAL_PLAN_ASSEMBLE
    _performance.NonlinearAssemblyPlan.diagnostics = _ORIGINAL_PLAN_DIAGNOSTICS
    _performance.clear_nonlinear_assembly_cache()
    _INSTALLED = False


def batch_b_status() -> Dict[str, Any]:
    return {
        "installed": bool(_INSTALLED),
        "description": "in-place elastic shell assembly into persistent buffers",
        "parallel_kernel": True,
    }
