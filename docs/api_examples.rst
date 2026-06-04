API Examples
************

The public API lives in ``anystruct.api``.

For a shorter manual/report version, see :doc:`api_manual_report`.

Units and sign conventions
--------------------------

* Geometry setter dimensions are in millimetres unless the docstring says otherwise.
* Material yield and elastic modulus are in MPa.
* Flat plate pressure and stresses are in MPa, and compression is positive.
* Cylinder stresses and pressure are in MPa, and compression is negative.
* Cylinder force input uses ``Nsd``/``Qsd`` in kN and ``Msd``/``Tsd`` in kNm.

Flat plate with girder
----------------------

This example uses all flat-plate geometry setters, PULS metadata, all supported
buckling methods, special provisions, and the optional numeric ML model bundle
hook.

.. code-block:: python

   from pprint import pprint

   import numpy as np

   from anystruct.api import FlatStru


   class DemoIdentityScaler:
       """Small sklearn-like scaler used only to demonstrate the ML bundle API."""

       def transform(self, rows):
           return np.asarray(rows, dtype=float)

       def inverse_transform(self, rows):
           return np.asarray(rows, dtype=float)


   class DemoValidityModel:
       def predict(self, rows):
           return np.ones(len(rows), dtype=int)


   class DemoNumericUfModel:
       def predict(self, rows):
           # The real model should return raw [buckling UF, ultimate UF] rows.
           return np.asarray([[0.4, 0.5] for _ in rows], dtype=float)


   def demo_numeric_ml_bundle(prefix):
       return {
           f"{prefix} validity predictor": DemoValidityModel(),
           f"{prefix} validity xscaler": DemoIdentityScaler(),
           f"{prefix} UF reg predictor": DemoNumericUfModel(),
           f"{prefix} UF reg xscaler": DemoIdentityScaler(),
           f"{prefix} UF reg yscaler": DemoIdentityScaler(),
       }


   def show(title, result):
       print(f"\n{title}")
       pprint(result)


   flat = FlatStru("Flat plate, stiffened with girder")
   flat.set_material(mat_yield=355, emodule=210000, material_factor=1.15, poisson=0.3)
   flat.set_plate_geometry(spacing=700, thickness=18, span=4000)
   flat.set_stresses(
       pressure=0.2,
       sigma_x1=50,
       sigma_x2=50,
       sigma_y1=100,
       sigma_y2=100,
       tau_xy=5,
   )
   flat.set_stiffener(hw=360, tw=12, bf=150, tf=20, stf_type="T", spacing=700)
   flat.set_girder(hw=600, tw=15, bf=220, tf=25, stf_type="T", spacing=2800)
   flat.set_fixation_parameters(kpp=1, kps=1, km1=12, km2=24, km3=12)
   flat.set_puls_parameters(
       sp_or_up="SP",
       puls_boundary="Int",
       stiffener_end="Continuous",
       up_boundary="SSSS",
   )

   print("Available methods:", flat.get_available_buckling_methods())

   flat.set_buckling_parameters(
       calculation_method="DNV-RP-C201 - prescriptive",
       buckling_acceptance="ultimate",
       stiffened_plate_effective_aginst_sigy=True,
       min_lat_press_adj_span=0,
       buckling_length_factor_stf=1,
       buckling_length_factor_girder=1,
       stf_dist_between_lateral_supp=4000,
       girder_dist_between_lateral_supp=4000,
       panel_length_Lp=4000,
       stiffener_support="Continuous",
       girder_support="Continuous",
       pressure_side="both sides",
       load_factor_stresses=1.0,
       load_factor_pressure=1.0,
       fabrication_method_stiffener="Fabricated",
       fabrication_method_girder="Cold formed",
   )
   show("DNV-RP-C201 buckling", flat.get_buckling_results())
   show("Special provisions", flat.get_special_provisions_results())

   flat.set_buckling_parameters(
       calculation_method="SemiAnalytical S3/U3",
       buckling_acceptance="buckling",
   )
   show("SemiAnalytical S3/U3 buckling", flat.get_buckling_results())

   # Numeric ML requires the five sklearn-like objects for the selected prefix.
   # SP + integrated boundary uses the "num SP int" prefix.
   flat.set_ml_buckling_model(demo_numeric_ml_bundle("num SP int"))
   flat.set_buckling_parameters(
       calculation_method="ML-Numeric (PULS based)",
       buckling_acceptance="ultimate",
   )
   show("ML-Numeric buckling", flat.get_buckling_results())


Unstiffened flat plate
----------------------

Use the same material, geometry, stress, PULS, and buckling calls. Skip the
stiffener and girder setters.

.. code-block:: python

   from pprint import pprint

   from anystruct.api import FlatStru


   unstiffened = FlatStru("Flat plate, unstiffened")
   unstiffened.set_material(mat_yield=355, emodule=210000, material_factor=1.15, poisson=0.3)
   unstiffened.set_plate_geometry(spacing=700, thickness=18, span=4000)
   unstiffened.set_stresses(
       pressure=0.2,
       sigma_x1=50,
       sigma_x2=50,
       sigma_y1=100,
       sigma_y2=100,
       tau_xy=5,
   )
   unstiffened.set_puls_parameters(sp_or_up="UP", puls_boundary="GL", up_boundary="SCSC")
   unstiffened.set_buckling_parameters(
       calculation_method="SemiAnalytical S3/U3",
       buckling_acceptance="buckling",
   )
   pprint(unstiffened.get_buckling_results())


Cylinder, stress input
----------------------

This orthogonally stiffened shell example uses shell geometry, longitudinal
stiffeners, ring girders, fabrication and limit-state controls, exclusion
switches, and stress input.

.. code-block:: python

   from pprint import pprint

   from anystruct.api import CylStru


   cyl = CylStru(calculation_domain="Orthogonally Stiffened shell")
   cyl.set_material(mat_yield=355, emodule=210000, material_factor=1.15, poisson=0.3)
   cyl.set_imperfection(delta_0=0.005)
   cyl.set_fabrication_method(stiffener="Fabricated", girder="Fabricated")
   cyl.set_end_cap_pressure_included_in_stress(is_included=True)
   cyl.set_uls_or_als(kind="ULS")
   cyl.set_shell_geometry(
       radius=6500,
       thickness=24,
       tot_length_of_shell=20000,
       distance_between_rings=3300,
   )
   cyl.set_shell_buckling_parmeters(eff_buckling_length_factor=1.0)
   cyl.set_length_between_girder(val=3300)
   cyl.set_panel_spacing(val=680)
   cyl.set_longitudinal_stiffener(
       hw=260,
       tw=23,
       bf=49,
       tf=28,
       stf_type="L-bulb",
       spacing=680,
   )
   cyl.set_ring_girder(hw=500, tw=15, bf=200, tf=25, stf_type="T", spacing=3300)
   cyl.set_exclude_ring_stiffener(is_excluded=True)
   cyl.set_exclude_ring_frame(is_excluded=False)
   cyl.set_stresses(sasd=-200, smsd=0, tTsd=0, tQsd=5, psd=0, shsd=-60)

   pprint(cyl.get_buckling_results())


Cylinder, force input
---------------------

For shell domains, ``set_forces`` can be used instead of ``set_stresses``.
Call ``set_shell_buckling_parmeters`` before requesting buckling results so
the column buckling check has an effective length factor.

.. code-block:: python

   from pprint import pprint

   from anystruct.api import CylStru


   force_cyl = CylStru(calculation_domain="Longitudinal Stiffened shell")
   force_cyl.set_material()
   force_cyl.set_imperfection()
   force_cyl.set_shell_geometry(
       radius=6500,
       thickness=20,
       distance_between_rings=3000,
       tot_length_of_shell=12000,
   )
   force_cyl.set_shell_buckling_parmeters(eff_buckling_length_factor=1.0)
   force_cyl.set_panel_spacing(700)
   force_cyl.set_longitudinal_stiffener(
       hw=260,
       tw=12,
       bf=80,
       tf=20,
       stf_type="T",
       spacing=700,
   )
   force_cyl.set_forces(Nsd=-1000, Msd=0, Tsd=0, Qsd=0, psd=0.1)

   pprint(force_cyl.get_buckling_results())


Cylinder, ring stiffener
------------------------

Use the ring stiffener setter in ring-stiffened shell or panel domains.

.. code-block:: python

   from pprint import pprint

   from anystruct.api import CylStru


   ring_cyl = CylStru(calculation_domain="Ring Stiffened shell")
   ring_cyl.set_material()
   ring_cyl.set_imperfection()
   ring_cyl.set_fabrication_method(stiffener="Fabricated", girder="Fabricated")
   ring_cyl.set_end_cap_pressure_included_in_stress(True)
   ring_cyl.set_uls_or_als("ULS")
   ring_cyl.set_shell_geometry(
       radius=6500,
       thickness=20,
       distance_between_rings=3000,
       tot_length_of_shell=12000,
   )
   ring_cyl.set_shell_buckling_parmeters(1.0)
   ring_cyl.set_panel_spacing(700)
   ring_cyl.set_ring_stiffener(
       hw=300,
       tw=12,
       bf=120,
       tf=20,
       stf_type="T",
       spacing=3000,
   )
   ring_cyl.set_forces(Nsd=-1000, Msd=0, Tsd=0, Qsd=0, psd=0.1)

   pprint(ring_cyl.get_buckling_results())


Project file facade
-------------------

Project files can be loaded and saved through the public facade without
constructing GUI objects.

.. code-block:: python

   from pathlib import Path
   from tempfile import TemporaryDirectory

   from anystruct.api import (
       ProjectHydrationDefaults,
       ProjectState,
       load_project_state,
       open_project,
       save_project_state,
   )


   with TemporaryDirectory() as tmp:
       project_path = Path(tmp) / "api_project.txt"
       state = ProjectState(
           project_information="API example",
           theme="dark",
           points={"p1": [0, 0]},
           buckling_method="SemiAnalytical S3/U3",
       )

       saved_path = save_project_state(state, project_path)
       loaded_state = load_project_state(saved_path)
       opened_project = open_project(
           saved_path,
           ProjectHydrationDefaults(structure_types={}),
       )

       print(loaded_state.project_information)
       print(opened_project.transfer.theme)


Supported domains and methods
-----------------------------

Flat plate domains:

* ``Flat plate, unstiffened``
* ``Flat plate, stiffened``
* ``Flat plate, stiffened with girder``

Cylinder domains:

* ``Unstiffened shell``
* ``Unstiffened panel``
* ``Longitudinal Stiffened shell``
* ``Longitudinal Stiffened panel``
* ``Ring Stiffened shell``
* ``Ring Stiffened panel``
* ``Orthogonally Stiffened shell``
* ``Orthogonally Stiffened panel``

Flat plate buckling methods:

* ``DNV-RP-C201 - prescriptive``
* ``SemiAnalytical S3/U3``
* ``ML-Numeric (PULS based)``
