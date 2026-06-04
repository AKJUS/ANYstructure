ANYstructure API
****************

The public API is exposed from ``anystruct.api``:

.. code-block:: python

   from anystruct.api import CylStru, FlatStru

The API covers flat plate buckling, cylinder/shell buckling, and project file
load/save helpers. External spreadsheet-driven DNV PULS execution is not part
of the public API in the current release. Use ``SemiAnalytical S3/U3`` or
``ML-Numeric (PULS based)`` for the built-in PULS-style flat plate workflows.

See :doc:`api_examples` for complete runnable examples covering flat plates,
cylinders, buckling methods, and project file load/save helpers. See
:doc:`api_manual_report` for a compact manual/report version.

Units and sign conventions
==========================

* Geometry setter dimensions are in millimetres unless a method docstring says
  otherwise.
* Material yield and elastic modulus are in MPa.
* Flat plate pressure and stresses are in MPa, and compression is positive.
* Cylinder pressure and stresses are in MPa, and compression is negative.
* Cylinder force input uses ``Nsd``/``Qsd`` in kN and ``Msd``/``Tsd`` in kNm.

Flat Plate API
==============

Supported domains:

* ``Flat plate, unstiffened``
* ``Flat plate, stiffened``
* ``Flat plate, stiffened with girder``

Supported buckling methods:

* ``DNV-RP-C201 - prescriptive``
* ``SemiAnalytical S3/U3``
* ``ML-Numeric (PULS based)``

``DNV-RP-C201 - prescriptive`` returns the legacy nested result dictionary with
``Plate``, ``Stiffener``, ``Girder``, and ``Local buckling`` entries where
applicable. ``SemiAnalytical S3/U3`` and ``ML-Numeric (PULS based)`` return a
method result dictionary with ``buckling UF``, ``ultimate UF``, raw UF values,
validity, and ``selected UF`` fields.

Current flat plate setters and result helpers:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Method
     - Purpose
   * - ``set_material``
     - Set yield, elastic modulus, material factor, and Poisson ratio.
   * - ``set_plate_geometry``
     - Set plate spacing, thickness, and span.
   * - ``set_stresses``
     - Set pressure, longitudinal/transverse stresses, and shear stress.
   * - ``set_stiffener``
     - Set stiffener web, flange, type, and spacing for stiffened domains.
   * - ``set_girder``
     - Set girder web, flange, type, and spacing for girder domains.
   * - ``set_fixation_parameters``
     - Set fixation and bending/shear factors used by special provisions.
   * - ``set_puls_parameters``
     - Set SP/UP panel type, PULS boundary, stiffener end support, and UP boundary code.
   * - ``set_buckling_parameters``
     - Select buckling method and set buckling, pressure, support, load-factor, fabrication, and ML options.
   * - ``set_ml_buckling_model``
     - Attach the sklearn-like model/scaler bundle used by ``ML-Numeric (PULS based)``.
   * - ``get_available_buckling_methods``
     - Return the supported flat plate buckling method names.
   * - ``get_buckling_results``
     - Run the selected flat plate buckling method.
   * - ``get_special_provisions_results``
     - Return minimum/actual plate thickness, stiffener section modulus, and stiffener shear area.

Cylinder API
============

Supported domains:

* ``Unstiffened shell``
* ``Unstiffened panel``
* ``Longitudinal Stiffened shell``
* ``Longitudinal Stiffened panel``
* ``Ring Stiffened shell``
* ``Ring Stiffened panel``
* ``Orthogonally Stiffened shell``
* ``Orthogonally Stiffened panel``

Shell domains use force input mode internally; panel domains use stress input
mode internally. The API exposes both ``set_stresses`` and ``set_forces`` for
cylinders. Configure shell geometry and the relevant stiffener/ring objects
before using force input, because forces are converted to stress from the
current geometry.

Current cylinder setters and result helpers:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Method
     - Purpose
   * - ``set_material``
     - Set yield, elastic modulus, material factor, and Poisson ratio.
   * - ``set_shell_geometry``
     - Set radius, shell thickness, ring spacing, and total cylinder length.
   * - ``set_shell_buckling_parmeters``
     - Set the effective buckling length factor used by column buckling.
   * - ``set_stresses``
     - Set axial, bending, torsional, shear, pressure, and hoop stresses.
   * - ``set_forces``
     - Set axial force, bending moment, torsional moment, shear force, and pressure.
   * - ``set_imperfection``
     - Set initial out-of-roundness.
   * - ``set_fabrication_method``
     - Set stiffener and girder fabrication method.
   * - ``set_end_cap_pressure_included_in_stress``
     - Control whether end-cap pressure is already included in stress input.
   * - ``set_uls_or_als``
     - Select ``ULS`` or ``ALS``.
   * - ``set_exclude_ring_stiffener``
     - Exclude ring stiffeners where the selected domain supports them.
   * - ``set_exclude_ring_frame``
     - Exclude ring frames/girders where the selected domain supports them.
   * - ``set_length_between_girder``
     - Set longitudinal distance between girders.
   * - ``set_panel_spacing``
     - Set curved panel width.
   * - ``set_longitudinal_stiffener``
     - Set longitudinal stiffener web, flange, type, and spacing.
   * - ``set_ring_stiffener``
     - Set ring stiffener web, flange, type, and spacing.
   * - ``set_ring_girder``
     - Set ring girder/frame web, flange, type, and spacing.
   * - ``get_buckling_results``
     - Return cylinder utilization factors and stiffener checks.

Project File API
================

The project facade allows callers to load, save, and hydrate project files
without constructing GUI objects.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Function or class
     - Purpose
   * - ``load_project_state(path)``
     - Load a project file into a ``ProjectState``.
   * - ``save_project_state(project_state, path)``
     - Save a ``ProjectState`` using the supported project-file codec.
   * - ``open_project(path, hydration_defaults=None)``
     - Load and hydrate a project through the application service facade.
   * - ``save_project(path, save_input)``
     - Create and save project state from ``ProjectSaveInput``.
   * - ``ProjectState``
     - JSON-safe project snapshot.
   * - ``ProjectSaveInput``
     - Plain save-boundary data used to create a project snapshot.
   * - ``ProjectHydrationDefaults``
     - Defaults used while hydrating legacy project data.

Public API Reference
====================

Functions
---------

.. autofunction:: anystruct.api.load_project_state
.. autofunction:: anystruct.api.save_project_state
.. autofunction:: anystruct.api.open_project
.. autofunction:: anystruct.api.save_project

.. autoclass:: anystruct.api.FlatStru
    :members:
    :autosummary:

.. autoclass:: anystruct.api.CylStru
    :members:
    :autosummary:

.. autoclass:: anystruct.api.ProjectState
    :members:

.. autoclass:: anystruct.api.ProjectSaveInput
    :members:

.. autoclass:: anystruct.api.ProjectHydrationDefaults
    :members:
