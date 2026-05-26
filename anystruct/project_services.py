"""Application services for project edits that do not depend on Tkinter."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping

from . import api_helpers, helper as hlp, line_structure
from .helper import one_load_combination
from .project_application import (
    LoadCombinationRecord,
    OpenLoadCombinationRecord,
    ProjectHydrationDefaults,
    ProjectHydrationResult,
    ProjectHydrationService,
    ProjectFileDialogService,
    ProjectFileTarget,
    ProjectFileCodec,
    ProjectOpenResult,
    ProjectOpenService,
    ProjectOpenTransfer,
    ProjectOpenTransferService,
    ProjectPersistenceError,
    ProjectPersistenceService,
    ProjectSaveInput,
    ProjectSaveResult,
    ProjectSaveService,
    ProjectSnapshotService,
)


POINT_PREFIX = "point"
LINE_PREFIX = "line"


def _numbered_name(prefix: str, number: int) -> str:
    return f"{prefix}{number}"


def _name_number(name: str, prefix: str) -> int:
    if not name.startswith(prefix):
        raise ValueError(f"{name!r} is not a {prefix} name")
    return int(name[len(prefix):])


def _next_available_name(records: MutableMapping[str, Any], prefix: str) -> str:
    number = 1
    while _numbered_name(prefix, number) in records:
        number += 1
    return _numbered_name(prefix, number)


@dataclass(frozen=True)
class PointRecord:
    """Named point used at the project editing boundary."""

    name: str
    x: float
    y: float

    @classmethod
    def from_legacy(cls, name: str, coordinates: list[float]) -> "PointRecord":
        return cls(name=name, x=coordinates[0], y=coordinates[1])

    def to_legacy(self) -> list[float]:
        return [self.x, self.y]


@dataclass(frozen=True)
class LineRecord:
    """Named line whose endpoints use project point names."""

    name: str
    first_point: str
    second_point: str

    @classmethod
    def from_legacy(cls, name: str, point_numbers: list[int]) -> "LineRecord":
        return cls(
            name=name,
            first_point=_numbered_name(POINT_PREFIX, point_numbers[0]),
            second_point=_numbered_name(POINT_PREFIX, point_numbers[1]),
        )

    @property
    def point_numbers(self) -> list[int]:
        return [
            _name_number(self.first_point, POINT_PREFIX),
            _name_number(self.second_point, POINT_PREFIX),
        ]

    @property
    def endpoint_keys(self) -> tuple[str, str]:
        first, second = self.point_numbers
        return f"p{first}p{second}", f"p{second}p{first}"

    def to_legacy(self) -> list[int]:
        return self.point_numbers


class ProjectEditService:
    """Mutate legacy geometry dictionaries through typed project records."""

    def __init__(self, points: MutableMapping[str, list[float]], lines: MutableMapping[str, list[int]]):
        self._points = points
        self._lines = lines

    def next_point_name(self) -> str:
        return _next_available_name(self._points, POINT_PREFIX)

    def next_line_name(self) -> str:
        return _next_available_name(self._lines, LINE_PREFIX)

    def add_point(self, name: str, coordinates: tuple[float, float]) -> PointRecord | None:
        if list(coordinates) in self._points.values():
            return None

        point = PointRecord(name=name, x=coordinates[0], y=coordinates[1])
        self._points[point.name] = point.to_legacy()
        return point

    def move_point(self, name: str, coordinates: tuple[float, float]) -> PointRecord | None:
        if name not in self._points or list(coordinates) in self._points.values():
            return None

        point = PointRecord(name=name, x=coordinates[0], y=coordinates[1])
        self._points[name] = point.to_legacy()
        return point

    def add_line(self, first_point: str, second_point: str) -> LineRecord | None:
        if first_point == second_point or first_point not in self._points or second_point not in self._points:
            return None
        if self._line_exists(first_point, second_point):
            return None

        line = LineRecord(name=self.next_line_name(), first_point=first_point, second_point=second_point)
        self._lines[line.name] = line.to_legacy()
        return line

    def line(self, name: str) -> LineRecord:
        return LineRecord.from_legacy(name, self._lines[name])

    def connected_line_names(self, point: str) -> list[str]:
        point_number = _name_number(point, POINT_PREFIX)
        return [name for name, endpoints in self._lines.items() if point_number in endpoints]

    def remove_line(self, name: str) -> LineRecord:
        line = self.line(name)
        self._lines.pop(name)
        return line

    def remove_point(self, name: str) -> PointRecord:
        point = PointRecord.from_legacy(name, self._points[name])
        self._points.pop(name)
        return point

    def _line_exists(self, first_point: str, second_point: str) -> bool:
        requested = {_name_number(first_point, POINT_PREFIX), _name_number(second_point, POINT_PREFIX)}
        return any(set(endpoints) == requested for endpoints in self._lines.values())


class LineStructureService:
    """Assign and update the legacy line structure bundles behind typed access."""

    def __init__(self, line_bundles: MutableMapping[str, list[Any]]):
        self._line_bundles = line_bundles

    def has_assignment(self, line_name: str) -> bool:
        return line_name in self._line_bundles

    def assign_structure(self, line_name: str, structure: Any, cylinder: Any = None):
        bundle = line_structure.LineStructureBundle(
            line_structure=structure,
            loads=[None],
            cylinder=cylinder,
        )
        self._line_bundles[line_name] = bundle.to_legacy_bundle()
        return bundle

    def structure(self, line_name: str):
        return self.bundle(line_name).line_structure

    def fatigue(self, line_name: str):
        return self.bundle(line_name).fatigue

    def cylinder(self, line_name: str):
        return self.bundle(line_name).cylinder

    def replace_structure(self, line_name: str, structure: Any):
        bundle = self.bundle(line_name)
        bundle.line_structure = structure
        self._write_bundle(line_name, bundle)
        return structure

    def update_structure_properties(self, line_name: str, properties: dict[str, Any]):
        structure = self.structure(line_name)
        structure.set_main_properties(properties)
        structure.need_recalc = True
        return structure

    def set_cylinder(self, line_name: str, cylinder: Any):
        bundle = self.bundle(line_name)
        bundle.cylinder = cylinder
        self._write_bundle(line_name, bundle)
        return cylinder

    def sync_fatigue_after_structure_update(self, line_name: str, properties: dict[str, Any]):
        bundle = self.bundle(line_name)
        if bundle.fatigue is None:
            return None

        if bundle.line_structure.calculation_domain == "Flat plate, unstiffened":
            bundle.fatigue = None
            self._write_bundle(line_name, bundle)
            return None

        bundle.fatigue.set_main_properties(properties["Stiffener"])
        return bundle.fatigue

    def replace_loads(self, line_name: str, loads: list[Any]):
        bundle = self.bundle(line_name)
        bundle.loads = loads
        self._write_bundle(line_name, bundle)
        return bundle.loads

    def append_load(self, line_name: str, load: Any):
        bundle = self.bundle(line_name)
        bundle.loads.append(load)
        return bundle.loads

    def bundle(self, line_name: str):
        return line_structure.LineStructureBundle.from_legacy_bundle(self._line_bundles[line_name])

    def _write_bundle(self, line_name: str, bundle: line_structure.LineStructureBundle):
        self._line_bundles[line_name] = bundle.to_legacy_bundle()


@dataclass(frozen=True)
class FlatStructurePropertyRequest:
    """Plain values needed to build a flat-plate structure property bundle."""

    calculation_domain: str
    base_values: dict[str, Any]
    girder_values: dict[str, Any]
    buckling_values: dict[str, Any]
    structure_types: dict[str, Any]


class FlatStructurePropertyService:
    """Build legacy flat-plate property bundles from plain boundary values."""

    @classmethod
    def build(cls, request: FlatStructurePropertyRequest):
        values = request.base_values
        obj_dict = {
            'mat_yield': [api_helpers.mpa_to_pa(values['material']), 'Pa'],
            'mat_factor': [values['material_factor'], ''],
            'span': [api_helpers.mm_to_m(values['span']), 'm'],
            'spacing': [api_helpers.mm_to_m(values['spacing']), 'm'],
            'plate_thk': [api_helpers.mm_to_m(values['plate_thk']), 'm'],
            'stf_web_height': [api_helpers.mm_to_m(values['stf_web_h']), 'm'],
            'stf_web_thk': [api_helpers.mm_to_m(values['stf_web_t']), 'm'],
            'stf_flange_width': [api_helpers.mm_to_m(values['stf_fl_w']), 'm'],
            'stf_flange_thk': [api_helpers.mm_to_m(values['stf_fl_t']), 'm'],
            'structure_type': [values['structure_type'], ''],
            'stf_type': [values['stf_type'], ''],
            'sigma_y1': [values['sigma_y1'], 'MPa'],
            'sigma_y2': [values['sigma_y2'], 'MPa'],
            'sigma_x1': [values['sigma_x1'], 'MPa'],
            'sigma_x2': [values['sigma_x2'], 'MPa'],
            'tau_xy': [values['tau_xy'], 'MPa'],
            'plate_kpp': [values['plate_kpp'], ''],
            'stf_kps': [values['stf_kps'], ''],
            'stf_km1': [values['stf_km1'], ''],
            'stf_km2': [values['stf_km2'], ''],
            'stf_km3': [values['stf_km3'], ''],
            'press_side': [values['pressure_side'], ''],
            'structure_types': [request.structure_types, ''],
            'zstar_optimization': [values['zstar_optimization'], ''],
            'puls buckling method': [values['puls_method'], ''],
            'puls boundary': [values['puls_boundary'], ''],
            'puls stiffener end': [values['puls_stiffener_end'], ''],
            'puls sp or up': [values['puls_sp_or_up'], ''],
            'puls up boundary': [values['puls_up_boundary'], ''],
            'panel or shell': [values['panel_or_shell'], ''],
            'girder_lg': [api_helpers.mm_to_m(values['girder_lg']), ''],
        }

        obj_dict_pl = dict(obj_dict)
        obj_dict_stf = dict(obj_dict)
        obj_dict_girder = dict(obj_dict)
        girder_values = request.girder_values
        obj_dict_girder['stf_web_height'] = [api_helpers.mm_to_m(girder_values['web_h']), 'm']
        obj_dict_girder['stf_web_thk'] = [api_helpers.mm_to_m(girder_values['web_t']), 'm']
        obj_dict_girder['stf_flange_width'] = [api_helpers.mm_to_m(girder_values['fl_w']), 'm']
        obj_dict_girder['stf_flange_thk'] = [api_helpers.mm_to_m(girder_values['fl_t']), 'm']
        obj_dict_girder['stf_type'] = [girder_values['type'], '']

        buckling_values = request.buckling_values
        main_dict = {
            'minimum pressure in adjacent spans': [buckling_values['min_pressure_adjacent_spans'], ''],
            'material yield': [api_helpers.mpa_to_pa(values['material']), 'Pa'],
            'load factor on stresses': [buckling_values['load_factor_stresses'], ''],
            'load factor on pressure': [1, ''],
            'buckling method': [values['puls_method'], ''],
            'stiffener end support': [buckling_values['stiffener_end_support'], ''],
            'girder end support': [buckling_values['girder_end_support'], ''],
            'tension field': [buckling_values['tension_field'], ''],
            'plate effective agains sigy': [buckling_values['plate_effective_against_sigy'], ''],
            'buckling length factor stf': [buckling_values['buckling_length_factor_stf'], ''],
            'buckling length factor girder': [buckling_values['buckling_length_factor_girder'], ''],
            'km3': [buckling_values['km3'], ''],
            'km2': [buckling_values['km2'], ''],
            'girder distance between lateral support': [buckling_values['girder_dist_lateral_support'], ''],
            'stiffener distance between lateral support': [buckling_values['stiffener_dist_lateral_support'], ''],
            'panel length, Lp': [buckling_values['panel_length'], ''],
            'pressure side': [values['pressure_side'], ''],
            'fabrication method stiffener': [buckling_values['fabrication_method_stiffener'], ''],
            'fabrication method girder': [buckling_values['fabrication_method_girder'], ''],
            'calculation domain': [request.calculation_domain, ''],
        }

        prop_dict = {
            'main dict': main_dict,
            'Plate': obj_dict_pl,
            'Stiffener': None if request.calculation_domain == 'Flat plate, unstiffened' else obj_dict_stf,
            'Girder': None if request.calculation_domain in ['Flat plate, unstiffened', 'Flat plate, stiffened']
            else obj_dict_girder,
        }
        return prop_dict, obj_dict_stf


@dataclass(frozen=True)
class CylinderStructurePropertyRequest:
    """Plain values needed to build cylinder property dictionaries."""

    calculation_domain: str
    dummy_values: dict[str, Any]
    shell_values: dict[str, Any]
    longitudinal_values: dict[str, Any]
    ring_stiffener_values: dict[str, Any]
    ring_frame_values: dict[str, Any]
    load_input: dict[str, Any]
    main_values: dict[str, Any]
    structure_types: dict[str, Any]


@dataclass(frozen=True)
class CylinderStructurePropertyResult:
    """Cylinder property dictionaries plus derived stress/force values."""

    main_dict: dict[str, Any]
    shell_dict: dict[str, Any]
    longitudinal_dict: dict[str, Any]
    ring_stiffener_dict: dict[str, Any]
    ring_frame_dict: dict[str, Any]
    geometry: int
    derived_stresses: tuple[Any, Any, Any, Any, Any]
    derived_forces: tuple[Any, Any, Any, Any]


class CylinderStructurePropertyService:
    """Build legacy cylinder property dictionaries from plain boundary values."""

    @classmethod
    def build(cls, request: CylinderStructurePropertyRequest) -> CylinderStructurePropertyResult:
        from .calc_structure import CylinderAndCurvedPlate

        geometry = api_helpers.geometry_id_for_domain(request.calculation_domain)
        dummy_data = cls._dummy_data(request)
        shell_dict = cls._shell_dict(request)
        long_dict = cls._longitudinal_dict(request)
        ring_stf_dict = cls._ring_stiffener_dict(request)
        ring_frame_dict = cls._ring_frame_dict(request)
        derived_stresses, derived_forces = cls._convert_load_input(
            request,
            geometry,
            cylinder_class=CylinderAndCurvedPlate,
        )
        sasd, smsd, tTsd, tQsd, shsd = derived_stresses
        main_values = request.main_values
        main_dict_cyl = {
            'sasd': [api_helpers.mpa_to_pa(sasd), 'Pa'],
            'smsd': [api_helpers.mpa_to_pa(smsd), 'Pa'],
            'tTsd': [api_helpers.mpa_to_pa(tTsd), 'Pa'],
            'tQsd': [api_helpers.mpa_to_pa(tQsd), 'Pa'],
            'psd': [api_helpers.mpa_to_pa(request.load_input['psd']), 'Pa'],
            'shsd': [api_helpers.mpa_to_pa(shsd), 'Pa'],
            'geometry': [geometry, ''],
            'material factor': [main_values['material_factor'], ''],
            'delta0': [0.005, ''],
            'fab method ring stf': [main_values['fab_method_ring_stiffener'], ''],
            'fab method ring girder': [main_values['fab_method_ring_frame'], ''],
            'E-module': [main_values['e_module'], 'Pa'],
            'poisson': [main_values['poisson'], ''],
            'mat_yield': [api_helpers.mpa_to_pa(main_values['yield']), 'Pa'],
            'length between girders': [api_helpers.mm_to_m(main_values['length_between_girders']), 'm'],
            'panel spacing, s': [api_helpers.mm_to_m(main_values['panel_spacing']), 'm'],
            'ring stf excluded': [main_values['ring_stiffener_excluded'], ''],
            'ring frame excluded': [main_values['ring_frame_excluded'], ''],
            'ULS or ALS': [main_values['uls_or_als'], ''],
            'end cap pressure': [main_values['end_cap_pressure'], ''],
        }

        for key, value in dummy_data.items():
            long_dict.setdefault(key, value)
            ring_stf_dict.setdefault(key, value)
            ring_frame_dict.setdefault(key, value)

        return CylinderStructurePropertyResult(
            main_dict=main_dict_cyl,
            shell_dict=shell_dict,
            longitudinal_dict=long_dict,
            ring_stiffener_dict=ring_stf_dict,
            ring_frame_dict=ring_frame_dict,
            geometry=geometry,
            derived_stresses=derived_stresses,
            derived_forces=derived_forces,
        )

    @staticmethod
    def _dummy_data(request):
        values = request.dummy_values
        return {
            'span': [api_helpers.mm_to_m(values['span']), 'm'],
            'plate_thk': [api_helpers.mm_to_m(values['plate_thk']), 'm'],
            'structure_type': [values['structure_type'], ''],
            'sigma_y1': [values['sigma_y1'], 'MPa'],
            'sigma_y2': [values['sigma_y2'], 'MPa'],
            'sigma_x1': [values['sigma_x1'], 'MPa'],
            'sigma_x2': [values['sigma_x2'], 'MPa'],
            'tau_xy': [values['tau_xy'], 'MPa'],
            'plate_kpp': [values['plate_kpp'], ''],
            'stf_kps': [values['stf_kps'], ''],
            'stf_km1': [values['stf_km1'], ''],
            'stf_km2': [values['stf_km2'], ''],
            'stf_km3': [values['stf_km3'], ''],
            'press_side': [values['pressure_side'], ''],
            'structure_types': [request.structure_types, ''],
            'zstar_optimization': [values['zstar_optimization'], ''],
            'puls buckling method': [values['puls_method'], ''],
            'puls boundary': [values['puls_boundary'], ''],
            'puls stiffener end': [values['puls_stiffener_end'], ''],
            'puls sp or up': [values['puls_sp_or_up'], ''],
            'puls up boundary': [values['puls_up_boundary'], ''],
            'panel or shell': [values['panel_or_shell'], ''],
            'mat_factor': [values['material_factor'], ''],
            'spacing': [api_helpers.mm_to_m(values['spacing']), 'm'],
        }

    @staticmethod
    def _shell_dict(request):
        values = request.shell_values
        return {
            'plate_thk': [api_helpers.mm_to_m(values['thickness']), 'm'],
            'radius': [api_helpers.mm_to_m(values['radius']), 'm'],
            'distance between rings, l': [api_helpers.mm_to_m(values['distance_between_rings']), 'm'],
            'length of shell, L': [api_helpers.mm_to_m(values['length']), 'm'],
            'tot cyl length, Lc': [api_helpers.mm_to_m(values['total_length']), 'm'],
            'eff. buckling lenght factor': [values['k_factor'], ''],
            'mat_yield': [api_helpers.mpa_to_pa(request.main_values['yield']), 'Pa'],
        }

    @staticmethod
    def _longitudinal_dict(request):
        values = request.longitudinal_values
        return {
            'spacing': [api_helpers.mm_to_m(values['spacing']), 'm'],
            'stf_web_height': [api_helpers.mm_to_m(values['web_h']), 'm'],
            'stf_web_thk': [api_helpers.mm_to_m(values['web_t']), 'm'],
            'stf_flange_width': [api_helpers.mm_to_m(values['fl_w']), 'm'],
            'stf_flange_thk': [api_helpers.mm_to_m(values['fl_t']), 'm'],
            'stf_type': [values['type'], ''],
            'span': [api_helpers.mm_to_m(request.dummy_values['span']), 'm'],
            'mat_yield': [api_helpers.mpa_to_pa(request.main_values['yield']), 'Pa'],
            'panel or shell': ['shell', ''],
        }

    @staticmethod
    def _ring_stiffener_dict(request):
        values = request.ring_stiffener_values
        return {
            'stf_web_height': [api_helpers.mm_to_m(values['web_h']), 'm'],
            'stf_web_thk': [api_helpers.mm_to_m(values['web_t']), 'm'],
            'stf_flange_width': [api_helpers.mm_to_m(values['fl_w']), 'm'],
            'stf_flange_thk': [api_helpers.mm_to_m(values['fl_t']), 'm'],
            'stf_type': [values['type'], ''],
            'mat_yield': [api_helpers.mpa_to_pa(request.main_values['yield']), 'Pa'],
            'panel or shell': ['shell', ''],
        }

    @staticmethod
    def _ring_frame_dict(request):
        values = request.ring_frame_values
        return {
            'stf_web_height': [api_helpers.mm_to_m(values['web_h']), 'm'],
            'stf_web_thk': [api_helpers.mm_to_m(values['web_t']), 'm'],
            'stf_flange_width': [api_helpers.mm_to_m(values['fl_w']), 'm'],
            'stf_flange_thk': [api_helpers.mm_to_m(values['fl_t']), 'm'],
            'stf_type': [values['type'], ''],
            'span': [api_helpers.mm_to_m(request.dummy_values['span']), 'm'],
            'mat_yield': [api_helpers.mpa_to_pa(request.main_values['yield']), 'Pa'],
            'panel or shell': ['shell', ''],
        }

    @staticmethod
    def _convert_load_input(request, geometry, cylinder_class):
        load_input = request.load_input
        shell = request.shell_values
        longitudinal = request.longitudinal_values
        converter_kwargs = {
            'geometry': geometry,
            'shell_t': shell['thickness'],
            'shell_radius': shell['radius'],
            'shell_spacing': longitudinal['spacing'],
            'hw': longitudinal['web_h'],
            'tw': longitudinal['web_t'],
            'b': longitudinal['fl_w'],
            'tf': longitudinal['fl_t'],
            'CylinderAndCurvedPlate': cylinder_class,
        }
        if load_input['mode'] == 1:
            forces = (
                load_input['Nsd'],
                load_input['Msd'],
                load_input['Tsd'],
                load_input['Qsd'],
            )
            stresses = hlp.helper_cylinder_stress_to_force_to_stress(
                stresses=None,
                forces=forces,
                **converter_kwargs,
            )
            return stresses, forces

        stresses = (
            load_input['sasd'],
            load_input['smsd'],
            abs(load_input['tTsd']),
            load_input['tQsd'],
            load_input['shsd'],
        )
        forces_with_shsd = hlp.helper_cylinder_stress_to_force_to_stress(
            stresses=stresses,
            **converter_kwargs,
        )
        return stresses, tuple(forces_with_shsd[:4])


@dataclass(frozen=True)
class LineLoadSyncResult:
    """Outcome of rebuilding per-line load assignments."""

    invalidated_lines: tuple[str, ...]
    changed_lines: tuple[str, ...]


class LineLoadService:
    """Rebuild per-line load references independently of the load window."""

    def __init__(self, line_bundles: MutableMapping[str, list[Any]]):
        self._line_bundles = line_bundles
        self._structures = LineStructureService(line_bundles)

    def rebuild_line_loads(
        self,
        line_names,
        load_assignments: MutableMapping[str, list[Any]],
        previous_load_assignments: MutableMapping[str, list[Any]] | None,
    ) -> LineLoadSyncResult:
        for line_name in self._line_bundles:
            self._structures.replace_loads(line_name, [])

        invalidated = tuple(mark_lines_for_recalculation(self._line_bundles))
        changed = []
        previous_load_assignments = previous_load_assignments or {}

        for line_name in line_names:
            if not self._structures.has_assignment(line_name):
                continue

            for load, load_lines in load_assignments.values():
                if self._line_load_changed(line_name, load, load_lines, previous_load_assignments):
                    if line_name not in changed:
                        changed.append(line_name)
                if line_name in load_lines:
                    self._structures.append_load(line_name, load)

        return LineLoadSyncResult(invalidated_lines=invalidated, changed_lines=tuple(changed))

    @staticmethod
    def _line_load_changed(line_name, load, load_lines, previous_load_assignments):
        previous = previous_load_assignments.get(load.get_name())
        if previous is None:
            return line_name in load_lines

        previous_load, previous_lines = previous
        load_changed_on_line = str(load) != str(previous_load) and line_name in load_lines + previous_lines
        line_assignment_changed = line_name in set(previous_lines).symmetric_difference(load_lines)
        return load_changed_on_line or line_assignment_changed


def _read_boundary_value(value):
    return value.get() if hasattr(value, "get") else value


@dataclass(frozen=True)
class LoadFactorRecord:
    """Plain load-factor values captured from the load-combination boundary."""

    name: tuple[Any, ...]
    static_factor: Any
    dynamic_factor: Any
    include: Any

    @classmethod
    def from_boundary_values(cls, name, values):
        return cls(
            name=tuple(name),
            static_factor=_read_boundary_value(values[0]),
            dynamic_factor=_read_boundary_value(values[1]),
            include=_read_boundary_value(values[2]),
        )


@dataclass(frozen=True)
class _LegacyLoadFactorValue:
    """Small get()-adapter for the legacy helper pressure calculation API."""

    value: Any

    def get(self):
        return self.value


def load_factor_records(load_factors) -> dict[tuple[Any, ...], LoadFactorRecord]:
    """Capture load factors from either Tk variables or plain boundary values."""
    return {
        tuple(name): LoadFactorRecord.from_boundary_values(name, values)
        for name, values in load_factors.items()
    }


@dataclass(frozen=True)
class LinePressureInput:
    """Plain application input for one line pressure/load-combination run."""

    line_name: str
    line_bundle: list[Any]
    coordinate: tuple[Any, Any]
    defined_tanks: tuple[tuple[str, Any], ...]
    accelerations: MutableMapping[str, Any]
    load_factors: MutableMapping[tuple[Any, ...], LoadFactorRecord]


class LinePressureService:
    """Calculate line load combinations without depending on Tk variables."""

    ZERO_PRESSURE_STRUCTURE_TYPES = {"", "FRAME", "GENERAL_INTERNAL_NONWT"}

    @classmethod
    def calculate_combinations(cls, line_input: LinePressureInput, limit_state="ULS", get_load_info=False):
        if limit_state == "FLS":
            return None

        results = {}
        load_info = []
        for combination_name in ("dnva", "dnvb"):
            results[combination_name] = []
            for load_condition in ("loaded", "ballast"):
                returned = cls.calculate_one(line_input, combination_name, load_condition)
                if returned is not None:
                    results[combination_name].append(returned[0])
                    load_info.extend(returned[1])

        for combination_name, load_condition in (
            ("tanktest", "tanktest"),
            ("manual", "manual"),
            ("slamming", "slamming"),
        ):
            returned = cls.calculate_one(line_input, combination_name, load_condition)
            results[combination_name] = [returned[0]]
            load_info.extend(returned[1])

        return load_info if get_load_info else results

    @classmethod
    def calculate_one(cls, line_input: LinePressureInput, combination_name, load_condition):
        bundle = line_structure.LineStructureBundle.from_legacy_bundle(line_input.line_bundle)
        defined_loads = [
            load
            for load in bundle.loads
            if load is not None and load.get_limit_state() != "FLS"
        ]

        acceleration_key = f"dyn_{load_condition}"
        acceleration = (
            line_input.accelerations["static"],
            0 if load_condition in {"tanktest", "manual", "slamming"}
            else line_input.accelerations[acceleration_key],
        )
        structure = bundle.line_structure
        current_line_object = [line_input.line_name, structure.Plate]
        if structure.Plate.get_structure_type() in cls.ZERO_PRESSURE_STRUCTURE_TYPES:
            return [0, ""]

        return one_load_combination(
            current_line_object,
            line_input.coordinate,
            defined_loads,
            load_condition,
            list(line_input.defined_tanks),
            combination_name,
            acceleration,
            cls._legacy_load_factor_values(line_input.load_factors),
        )

    @classmethod
    def highest_pressure(cls, line_input: LinePressureInput, limit_state="ULS"):
        if limit_state != "ULS":
            return {"normal": 0, "slamming": 0}

        results = cls.calculate_combinations(line_input)
        normal_pressures = []
        slamming_pressure = 0
        slamming_plate_factor = 1
        slamming_stiffener_factor = 1

        for combination_name, values in results.items():
            if combination_name != "slamming":
                normal_pressures.append(max(values))
                continue
            if values is not None:
                for load in line_structure.loads(line_input.line_bundle):
                    if load is not None and load.get_load_condition() == "slamming":
                        slamming_plate_factor = load.get_slamming_reduction_plate()
                        slamming_stiffener_factor = load.get_slamming_reduction_stf()
                slamming_pressure = max(values)

        return {
            "normal": max(normal_pressures),
            "slamming": slamming_pressure,
            "slamming plate reduction factor": slamming_plate_factor,
            "slamming stf reduction factor": slamming_stiffener_factor,
        }

    @staticmethod
    def _legacy_load_factor_values(load_factors):
        return {
            name: (
                _LegacyLoadFactorValue(record.static_factor),
                _LegacyLoadFactorValue(record.dynamic_factor),
                _LegacyLoadFactorValue(record.include),
            )
            for name, record in load_factors.items()
        }


@dataclass(frozen=True)
class SesamExportRequest:
    """Plain project payload for the supported SESAM JavaScript export."""

    points: MutableMapping[str, Any]
    lines: MutableMapping[str, Any]
    sections: Any
    line_bundles: MutableMapping[str, list[Any]]


class SesamExportService:
    """Build the supported SESAM JavaScript export outside GUI callbacks."""

    @staticmethod
    def build_js_lines(request: SesamExportRequest, export_factory=None) -> tuple[str, ...]:
        if export_factory is None:
            from . import sesam_interface

            export_factory = sesam_interface.JSfile

        export = export_factory(
            request.points,
            request.lines,
            request.sections,
            line_to_struc=request.line_bundles,
        )
        export.write_points()
        export.write_lines()
        export.write_sections()
        export.write_beams()
        return tuple(export.output_lines)

    @classmethod
    def write_js_path(cls, request: SesamExportRequest, path, export_factory=None) -> Path:
        export_path = Path(path)
        export_path.write_text("".join(cls.build_js_lines(request, export_factory)), encoding="utf-8")
        return export_path


@dataclass(frozen=True)
class ExcelProjectImportData:
    """Supported workbook rows read before the Tk import callback applies them."""

    flat_plate_rows: tuple[Any, ...]
    cylinder_rows: tuple[Any, ...]
    flat_plate_records: tuple["FlatPlateExcelImportRecord", ...]
    cylinder_records: tuple["CylinderExcelImportRecord", ...]


@dataclass(frozen=True)
class FlatPlateExcelImportRecord:
    """Flat-plate line values decoded from the supported import workbook."""

    calculation_domain: Any
    first_point: tuple[Any, Any]
    second_point: tuple[Any, Any]
    plate_values: tuple[Any, ...]
    stress_values: tuple[Any, ...]
    manual_pressure: Any
    girder_values: tuple[Any, ...]
    buckling_values: tuple[Any, ...]

    @classmethod
    def from_row(cls, row: list[Any] | tuple[Any, ...]):
        return cls(
            calculation_domain=row[0],
            first_point=(row[1], row[2]),
            second_point=(row[3], row[4]),
            plate_values=tuple(row[5:11]),
            stress_values=tuple(row[12:17]),
            manual_pressure=row[16],
            girder_values=tuple(row[18:24]),
            buckling_values=tuple(row[24:36]),
        )


@dataclass(frozen=True)
class CylinderExcelImportRecord:
    """Cylinder line values decoded from the supported import workbook."""

    calculation_domain: Any
    first_point: tuple[Any, Any]
    second_point: tuple[Any, Any]
    shell_values: tuple[Any, ...]
    longitudinal_values: tuple[Any, ...]
    ring_stiffener_values: tuple[Any, ...]
    ring_frame_values: tuple[Any, ...]
    stress_values: tuple[Any, ...]
    force_values: tuple[Any, ...]
    end_values: tuple[Any, ...]

    @classmethod
    def from_row(cls, row: list[Any] | tuple[Any, ...]):
        return cls(
            calculation_domain=row[0],
            first_point=(row[1], row[2]),
            second_point=(row[3], row[4]),
            shell_values=tuple(row[5:12]),
            longitudinal_values=tuple(row[12:18]),
            ring_stiffener_values=tuple(row[18:23]),
            ring_frame_values=tuple(row[23:29]),
            stress_values=tuple(row[29:35]),
            force_values=tuple(row[35:39]),
            end_values=tuple(row[39:43]),
        )


@dataclass(frozen=True)
class CylinderExcelImportDefaults:
    """Current project defaults needed when a workbook omits cylinder UI-only values."""

    plate_thk: Any
    structure_type: Any
    sigma_y1: Any
    sigma_y2: Any
    sigma_x1: Any
    sigma_x2: Any
    tau_xy: Any
    plate_kpp: Any
    stf_kps: Any
    stf_km1: Any
    stf_km2: Any
    stf_km3: Any
    pressure_side: Any
    zstar_optimization: Any
    puls_method: Any
    puls_boundary: Any
    puls_stiffener_end: Any
    puls_sp_or_up: Any
    puls_up_boundary: Any
    panel_or_shell: Any
    material_factor: Any
    design_pressure: Any
    shear_stress: Any
    e_module: Any
    poisson: Any
    length_between_girders: Any
    fab_method_ring_stiffener: Any
    fab_method_ring_frame: Any
    end_cap_pressure: Any
    structure_types: dict[str, Any]
    ring_stiffener_type: Any = "T"
    ring_frame_type: Any = "T"


class CylinderExcelImportPropertyService:
    """Convert supported cylinder workbook records into plain property requests."""

    @classmethod
    def build_request(
        cls,
        record: CylinderExcelImportRecord,
        defaults: CylinderExcelImportDefaults,
    ) -> CylinderStructurePropertyRequest:
        span = hlp.dist(record.first_point, record.second_point)
        shell_thk, shell_radius, dist_rings, shell_length, shell_total_length, shell_k, shell_mat = \
            record.shell_values
        long_web_h, long_web_t, long_fl_w, long_fl_t, panel_spacing, long_type = \
            record.longitudinal_values
        ring_stf_values, ring_stf_excluded = cls._component_values(
            record.ring_stiffener_values,
            default_type=defaults.ring_stiffener_type,
            has_length=False,
        )
        ring_frame_values, ring_frame_excluded = cls._component_values(
            record.ring_frame_values,
            default_type=defaults.ring_frame_type,
            has_length=True,
        )
        load_input = cls._load_input(record, defaults)
        uls_or_als, shell_yield, _ring_stf_fab, _ring_frame_fab = record.end_values

        return CylinderStructurePropertyRequest(
            calculation_domain=record.calculation_domain,
            dummy_values={
                "span": span,
                "plate_thk": defaults.plate_thk,
                "structure_type": defaults.structure_type,
                "sigma_y1": defaults.sigma_y1,
                "sigma_y2": defaults.sigma_y2,
                "sigma_x1": defaults.sigma_x1,
                "sigma_x2": defaults.sigma_x2,
                "tau_xy": defaults.tau_xy,
                "plate_kpp": defaults.plate_kpp,
                "stf_kps": defaults.stf_kps,
                "stf_km1": defaults.stf_km1,
                "stf_km2": defaults.stf_km2,
                "stf_km3": defaults.stf_km3,
                "pressure_side": defaults.pressure_side,
                "zstar_optimization": defaults.zstar_optimization,
                "puls_method": defaults.puls_method,
                "puls_boundary": defaults.puls_boundary,
                "puls_stiffener_end": defaults.puls_stiffener_end,
                "puls_sp_or_up": defaults.puls_sp_or_up,
                "puls_up_boundary": defaults.puls_up_boundary,
                "panel_or_shell": defaults.panel_or_shell,
                "material_factor": defaults.material_factor,
                "spacing": panel_spacing,
            },
            shell_values={
                "thickness": shell_thk,
                "radius": shell_radius,
                "distance_between_rings": dist_rings,
                "length": shell_length,
                "total_length": shell_total_length,
                "k_factor": shell_k,
            },
            longitudinal_values={
                "spacing": panel_spacing,
                "web_h": long_web_h,
                "web_t": long_web_t,
                "fl_w": long_fl_w,
                "fl_t": long_fl_t,
                "type": long_type,
            },
            ring_stiffener_values={
                "web_h": ring_stf_values[0],
                "web_t": ring_stf_values[1],
                "fl_w": ring_stf_values[2],
                "fl_t": ring_stf_values[3],
                "type": ring_stf_values[4],
            },
            ring_frame_values={
                "web_h": ring_frame_values[0],
                "web_t": ring_frame_values[1],
                "fl_w": ring_frame_values[2],
                "fl_t": ring_frame_values[3],
                "type": ring_frame_values[5],
            },
            load_input=load_input,
            main_values={
                "material_factor": shell_mat,
                "fab_method_ring_stiffener": defaults.fab_method_ring_stiffener,
                "fab_method_ring_frame": defaults.fab_method_ring_frame,
                "e_module": defaults.e_module,
                "poisson": defaults.poisson,
                "yield": shell_yield,
                "length_between_girders": (
                    defaults.length_between_girders if ring_frame_excluded else ring_frame_values[4]
                ),
                "panel_spacing": panel_spacing,
                "ring_stiffener_excluded": ring_stf_excluded,
                "ring_frame_excluded": ring_frame_excluded,
                "uls_or_als": uls_or_als,
                "end_cap_pressure": defaults.end_cap_pressure,
            },
            structure_types=defaults.structure_types,
        )

    @staticmethod
    def _component_values(values, *, default_type, has_length):
        if not values or values[0] is None:
            if has_length:
                return (0, 0, 0, 0, 0, default_type), True
            return (0, 0, 0, 0, default_type), True
        return values, False

    @staticmethod
    def _load_input(record, defaults):
        stresses = record.stress_values or ()
        forces = record.force_values or ()
        has_stresses = bool(stresses) and stresses[0] is not None
        has_forces = bool(forces) and forces[0] is not None

        sasd, smsd, tTsd, tQsd, psd, shsd = (
            stresses if has_stresses else (0, 0, 0, 0, defaults.design_pressure, defaults.shear_stress)
        )
        psd = defaults.design_pressure if psd is None else psd
        shsd = defaults.shear_stress if shsd is None else shsd
        nsd, msd, tsd, qsd = forces if has_forces else (0, 0, 0, 0)
        return {
            "mode": 1 if has_forces else 2,
            "Nsd": nsd,
            "Msd": msd,
            "Tsd": tsd,
            "Qsd": qsd,
            "sasd": sasd,
            "smsd": smsd,
            "tTsd": tTsd,
            "tQsd": tQsd,
            "psd": psd,
            "shsd": shsd,
        }


@dataclass(frozen=True)
class ExcelImportedLine:
    """One workbook record matched to the project line it created."""

    record: Any
    line: LineRecord


@dataclass(frozen=True)
class ExcelGeometryImportResult:
    """Geometry changes made from workbook records before structures are applied."""

    created_points: tuple[PointRecord, ...]
    imported_lines: tuple[ExcelImportedLine, ...]


class ExcelProjectGeometryImportService:
    """Apply supported import geometry to the legacy project geometry stores."""

    @classmethod
    def add_records(cls, points, lines, records) -> ExcelGeometryImportResult:
        editor = ProjectEditService(points, lines)
        created_points = []
        imported_lines = []

        for record in records:
            for workbook_point in (record.first_point, record.second_point):
                point = editor.add_point(editor.next_point_name(), cls._point_in_meters(workbook_point))
                if point is not None:
                    created_points.append(point)

        point_names_by_coordinate = {tuple(coordinates): name for name, coordinates in points.items()}
        for record in records:
            first_point = point_names_by_coordinate.get(cls._point_in_meters(record.first_point))
            second_point = point_names_by_coordinate.get(cls._point_in_meters(record.second_point))
            if first_point is None or second_point is None:
                continue

            line = editor.add_line(first_point, second_point)
            if line is not None:
                imported_lines.append(ExcelImportedLine(record=record, line=line))

        return ExcelGeometryImportResult(
            created_points=tuple(created_points),
            imported_lines=tuple(imported_lines),
        )

    @staticmethod
    def _point_in_meters(workbook_point):
        return workbook_point[0] / 1000, workbook_point[1] / 1000


class ExcelProjectImportService:
    """Read the supported Excel project import workbook through an adapter."""

    @classmethod
    def read_path(cls, path: str | Path, workbook_factory: Callable[..., Any] | None = None):
        workbook = cls._open_workbook(
            path,
            visible=False,
            read_only=True,
            workbook_factory=workbook_factory,
        )
        try:
            flat_plate_rows = tuple(workbook.get_sheet_data("flat_plate") or ())
            cylinder_rows = tuple(workbook.get_sheet_data("cylinder") or ())
            return ExcelProjectImportData(
                flat_plate_rows=flat_plate_rows,
                cylinder_rows=cylinder_rows,
                flat_plate_records=tuple(
                    FlatPlateExcelImportRecord.from_row(row) for row in flat_plate_rows[1:]
                ),
                cylinder_records=tuple(
                    CylinderExcelImportRecord.from_row(row) for row in cylinder_rows[1:]
                ),
            )
        finally:
            workbook.close_book()

    @classmethod
    def open_example_path(cls, path: str | Path, workbook_factory: Callable[..., Any] | None = None):
        """Open the bundled input workbook for inspection in Excel."""
        return cls._open_workbook(
            path,
            visible=True,
            read_only=True,
            workbook_factory=workbook_factory,
        )

    @staticmethod
    def _open_workbook(path, *, visible, read_only, workbook_factory=None):
        if workbook_factory is None:
            from .excel_inteface import ExcelInterface

            workbook_factory = ExcelInterface
        return workbook_factory(str(path), visible=visible, read_only=read_only)


@dataclass(frozen=True)
class ReportRequest:
    """Report command values captured before the reporter adapter is called."""

    filename: str
    title: str
    seconds: int
    source_data: Any


@dataclass(frozen=True)
class ReportDataSnapshot:
    """Plain report input captured before the Tk application calls a renderer."""

    project_information: str
    buckling_method: str
    points: MutableMapping[str, Any]
    lines: MutableMapping[str, Any]
    line_bundles: MutableMapping[str, list[Any]]
    tanks: MutableMapping[str, Any]
    loads: MutableMapping[str, Any]
    result_state: MutableMapping[str, Any]
    highest_pressures: MutableMapping[str, Any]
    ml_classes: Any


class ReportRequestService:
    """Orchestrate supported report requests through an injected adapter."""

    @staticmethod
    def create_pdf(request: ReportRequest, report_factory: Callable[..., Any] | None = None):
        if report_factory is None:
            from .report_generator import LetterMaker

            report_factory = LetterMaker

        report = report_factory(request.filename, request.title, request.seconds, request.source_data)
        report.createDocument()
        report.savePDF()
        return report

    @staticmethod
    def create_table(
        request: ReportRequest,
        report_factory: Callable[..., Any] | None = None,
        document_factory: Callable[[str], Any] | None = None,
    ):
        if report_factory is None:
            from .report_generator import LetterMaker

            report_factory = LetterMaker
        if document_factory is None:
            from reportlab.lib.pagesizes import landscape, letter
            from reportlab.platypus import SimpleDocTemplate

            document_factory = lambda filename: SimpleDocTemplate(filename, pagesize=landscape(letter))

        report = report_factory(request.filename, request.title, request.seconds, request.source_data)
        document = document_factory(request.filename)
        document.build(report.createTable())
        return document


def mark_line_for_recalculation(line_bundles: MutableMapping[str, list[Any]], line_name: str) -> bool:
    """Mark one assigned line dirty and report whether a structure was available."""
    structure = line_structure.structure(line_bundles[line_name]) if line_name in line_bundles else None
    if structure is None:
        return False

    structure.need_recalc = True
    return True


def mark_lines_for_recalculation(line_bundles: MutableMapping[str, list[Any]]) -> list[str]:
    """Mark legacy line structure bundles dirty through the named bundle adapter."""
    invalidated = []
    for line_name in line_bundles:
        if mark_line_for_recalculation(line_bundles, line_name):
            invalidated.append(line_name)
    return invalidated
