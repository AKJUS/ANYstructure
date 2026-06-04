Compact API Report Example
**************************

This compact example is intended for manuals and reports. It avoids the longer
demonstration scaffolding from :doc:`api_examples` and prints a short summary
of the main checks.

.. code-block:: python

   from anystruct.api import CylStru, FlatStru


   def flatten_numbers(value):
       if isinstance(value, dict):
           for child in value.values():
               yield from flatten_numbers(child)
       elif isinstance(value, (list, tuple)):
           for child in value:
               yield from flatten_numbers(child)
       elif value is not None and not isinstance(value, bool):
           try:
               yield float(value)
           except (TypeError, ValueError):
               return


   def max_uf(results):
       values = list(flatten_numbers(results))
       return max(values) if values else None


   def print_flat_report(flat):
       print("FLAT PLATE")
       print("Available buckling methods:", ", ".join(flat.get_available_buckling_methods()))

       flat.set_buckling_parameters(
           calculation_method="DNV-RP-C201 - prescriptive",
           buckling_acceptance="ultimate",
           pressure_side="both sides",
           fabrication_method_stiffener="Fabricated",
           fabrication_method_girder="Cold formed",
       )
       dnv = flat.get_buckling_results()
       print(f"DNV-RP-C201 max UF: {max_uf(dnv):.3f}")

       flat.set_buckling_parameters(
           calculation_method="SemiAnalytical S3/U3",
           buckling_acceptance="buckling",
       )
       semi = flat.get_buckling_results()
       print(f"SemiAnalytical selected UF: {semi['selected UF']:.3f}")
       print(f"SemiAnalytical valid: {semi['available']} ({semi['valid label']})")

       special = flat.get_special_provisions_results()
       for name, check in special.items():
           print(f"{name}: actual={check['actual']:.1f}, minimum={check['minimum']:.1f}")

       print("ML-Numeric: call set_ml_buckling_model(ml_algo) before using the ML method.")


   def print_cylinder_report(cylinder):
       results = cylinder.get_buckling_results()
       print("\nCYLINDER")
       for key in (
           "Unstiffened shell",
           "Longitudinal stiffened shell",
           "Ring stiffened shell",
           "Heavy ring frame",
           "Column stability UF",
       ):
           value = results.get(key)
           if value is not None:
               print(f"{key}: {float(value):.3f}")
       print("Stiffener check:", results.get("Stiffener check"))


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

   cylinder = CylStru("Orthogonally Stiffened shell")
   cylinder.set_material(mat_yield=355, emodule=210000, material_factor=1.15, poisson=0.3)
   cylinder.set_imperfection(delta_0=0.005)
   cylinder.set_fabrication_method(stiffener="Fabricated", girder="Fabricated")
   cylinder.set_end_cap_pressure_included_in_stress(is_included=True)
   cylinder.set_uls_or_als(kind="ULS")
   cylinder.set_shell_geometry(
       radius=6500,
       thickness=24,
       distance_between_rings=3300,
       tot_length_of_shell=20000,
   )
   cylinder.set_shell_buckling_parmeters(eff_buckling_length_factor=1.0)
   cylinder.set_length_between_girder(val=3300)
   cylinder.set_panel_spacing(val=680)
   cylinder.set_longitudinal_stiffener(hw=260, tw=23, bf=49, tf=28, stf_type="L-bulb", spacing=680)
   cylinder.set_ring_girder(hw=500, tw=15, bf=200, tf=25, stf_type="T", spacing=3300)
   cylinder.set_exclude_ring_stiffener(is_excluded=True)
   cylinder.set_exclude_ring_frame(is_excluded=False)
   cylinder.set_stresses(sasd=-200, smsd=0, tTsd=0, tQsd=5, psd=0, shsd=-60)

   print_flat_report(flat)
   print_cylinder_report(cylinder)

Example output headings:

.. code-block:: text

   FLAT PLATE
   Available buckling methods: DNV-RP-C201 - prescriptive, SemiAnalytical S3/U3, ML-Numeric (PULS based)
   DNV-RP-C201 max UF: ...
   SemiAnalytical selected UF: ...
   SemiAnalytical valid: ...
   Plate thickness: actual=..., minimum=...

   CYLINDER
   Unstiffened shell: ...
   Longitudinal stiffened shell: ...
   Heavy ring frame: ...
   Column stability UF: ...
   Stiffener check: ...
