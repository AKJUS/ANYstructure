"""Project snapshot, hydration, open-transfer, and persistence services."""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, MutableMapping

from . import example_data, line_structure, project_io
from .calc_loads import Loads
from .calc_structure import (
    AllStructure,
    CalcFatigue,
    CalcScantlings,
    CylinderAndCurvedPlate,
    Shell,
    Structure,
)
from .project_state import ProjectState


ProjectFileCodec = project_io.ProjectFileCodec


@dataclass(frozen=True)
class LoadCombinationRecord:
    """Plain load-combination value captured at the UI boundary."""

    name: Any
    static_factor: Any
    dynamic_factor: Any
    include: Any

    def to_legacy(self) -> list[Any]:
        return [self.name, self.static_factor, self.dynamic_factor, self.include]


class ProjectPersistenceError(RuntimeError):
    """Failure raised while a project file is loaded or saved."""

    def __init__(self, action: str, location: Any, error: Exception):
        self.action = action
        self.location = str(location)
        self.error = error
        super().__init__(f"Could not {action} project file {self.location!r}: {error}")


class ProjectPersistenceService:
    """Load and save project state without depending on Tk file objects."""

    BACKUP_FILE_NAME = "backup.txt"
    codec = ProjectFileCodec

    @classmethod
    def save_state_to_path(cls, state: ProjectState, path: str | Path) -> Path:
        project_path = Path(path)
        try:
            with project_path.open("w", encoding="utf-8") as project_file:
                cls.codec.dump(state, project_file)
        except (OSError, TypeError, ValueError) as error:
            raise ProjectPersistenceError("save", project_path, error) from error
        return project_path

    @classmethod
    def load_state_from_path(cls, path: str | Path) -> ProjectState:
        project_path = Path(path)
        try:
            with project_path.open(encoding="utf-8") as project_file:
                return cls.codec.load(project_file)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ProjectPersistenceError("load", project_path, error) from error

    @classmethod
    def backup_path(cls, root_dir: str | Path) -> Path:
        return Path(root_dir).parent / cls.BACKUP_FILE_NAME

    @classmethod
    def backup_exists(cls, root_dir: str | Path) -> bool:
        return cls.backup_path(root_dir).is_file()


@dataclass(frozen=True)
class ProjectFileTarget:
    """Resolved project file path plus whether it should become the remembered save path."""

    path: Path
    remember_as_last_save: bool = False


class ProjectFileDialogService:
    """Resolve project file targets without depending on Tk dialogs."""

    @staticmethod
    def selected_save_target(filename: str | Path, *, backup: bool = False) -> ProjectFileTarget:
        return ProjectFileTarget(Path(filename), remember_as_last_save=not backup)

    @staticmethod
    def selected_output_target(filename: str | Path | None) -> ProjectFileTarget | None:
        if filename in (None, ""):
            return None
        return ProjectFileTarget(Path(filename), remember_as_last_save=False)

    @staticmethod
    def selected_open_target(filename: str | Path | None) -> ProjectFileTarget | None:
        if filename in (None, ""):
            return None
        return ProjectFileTarget(Path(filename), remember_as_last_save=False)

    @staticmethod
    def backup_save_target(root_dir: str | Path) -> ProjectFileTarget:
        return ProjectFileTarget(ProjectPersistenceService.backup_path(root_dir), remember_as_last_save=False)

    @staticmethod
    def remembered_save_target(last_save_file: str | Path | None) -> ProjectFileTarget | None:
        if last_save_file is None:
            return None
        return ProjectFileTarget(Path(last_save_file), remember_as_last_save=False)

    @staticmethod
    def restore_target(root_dir: str | Path) -> ProjectFileTarget | None:
        backup_path = ProjectPersistenceService.backup_path(root_dir)
        if not backup_path.is_file():
            return None
        return ProjectFileTarget(backup_path, remember_as_last_save=False)

    @staticmethod
    def example_open_target(file_name: str | Path, root_dir: str | Path) -> ProjectFileTarget:
        project_path = Path(file_name)
        if not project_path.is_file():
            project_path = Path(root_dir) / project_path
        return ProjectFileTarget(project_path, remember_as_last_save=False)


@dataclass(frozen=True)
class ProjectSaveInput:
    """Plain project data captured at the save boundary."""

    project_information: str
    theme: str
    points: MutableMapping[str, Any]
    lines: MutableMapping[str, Any]
    line_bundles: MutableMapping[str, list[Any]]
    load_assignments: MutableMapping[str, list[Any]]
    accelerations: MutableMapping[str, Any]
    load_combinations: tuple[LoadCombinationRecord, ...]
    tanks: MutableMapping[str, Any]
    tank_grid: Any
    tank_search_data: Any
    buckling_method: str
    shifting: MutableMapping[str, Any]
    weight_and_cog: MutableMapping[str, Any]


@dataclass(frozen=True)
class ProjectSaveResult:
    """Persisted save state and its resolved project path."""

    path: Path
    state: ProjectState


class ProjectSnapshotService:
    """Create persistence snapshots from domain objects and plain UI values."""

    @classmethod
    def create_state(
        cls,
        *,
        project_information: str,
        theme: str,
        points: MutableMapping[str, Any],
        lines: MutableMapping[str, Any],
        line_bundles: MutableMapping[str, list[Any]],
        load_assignments: MutableMapping[str, list[Any]],
        accelerations: MutableMapping[str, Any],
        load_combinations: list[LoadCombinationRecord],
        tanks: MutableMapping[str, Any],
        tank_grid: Any,
        tank_search_data: Any,
        buckling_method: str,
        shifting: MutableMapping[str, Any],
        weight_and_cog: MutableMapping[str, Any],
    ) -> ProjectState:
        structures, shell_structures, fatigue = cls._serialize_line_bundles(line_bundles)
        return ProjectState(
            project_information=project_information,
            theme=theme,
            points=points,
            lines=lines,
            structures=structures,
            shell_structures=shell_structures,
            loads=cls._serialize_loads(load_assignments),
            accelerations=accelerations,
            load_combinations=cls._serialize_load_combinations(load_combinations),
            tanks=cls._serialize_tanks(tanks, tank_grid, tank_search_data),
            fatigue=fatigue,
            buckling_method=buckling_method,
            shifting=shifting,
            weight_and_cog=weight_and_cog,
        )

    @staticmethod
    def _serialize_line_bundles(line_bundles):
        structures = {}
        shell_structures = {}
        fatigue = {}

        for line_name, legacy_bundle in line_bundles.items():
            bundle = line_structure.LineStructureBundle.from_legacy_bundle(legacy_bundle)
            structures[line_name] = bundle.line_structure.get_main_properties()
            shell_structures[line_name] = (
                None if bundle.cylinder is None else bundle.cylinder.get_all_properties()
            )
            if bundle.fatigue is None:
                fatigue[line_name] = None
                continue

            try:
                fatigue[line_name] = bundle.fatigue.get_fatigue_properties()
            except AttributeError:
                fatigue[line_name] = None

        return structures, shell_structures, fatigue

    @staticmethod
    def _serialize_loads(load_assignments):
        return {
            load_name: [load_data[0].get_load_parmeters(), load_data[1]]
            for load_name, load_data in load_assignments.items()
        }

    @staticmethod
    def _serialize_load_combinations(load_combinations):
        return {
            combination_number: load_combination.to_legacy()
            for combination_number, load_combination in enumerate(load_combinations)
        }

    @staticmethod
    def _serialize_tanks(tanks, tank_grid, tank_search_data):
        tank_properties = {"grid": tank_grid, "search_data": tank_search_data}
        for tank_name, tank in tanks.items():
            tank_properties[tank_name] = tank.get_parameters()
        return tank_properties


class ProjectSaveService:
    """Create and persist project snapshots from plain save input."""

    @staticmethod
    def create_state(save_input: ProjectSaveInput) -> ProjectState:
        return ProjectSnapshotService.create_state(
            project_information=save_input.project_information,
            theme=save_input.theme,
            points=save_input.points,
            lines=save_input.lines,
            line_bundles=save_input.line_bundles,
            load_assignments=save_input.load_assignments,
            accelerations=save_input.accelerations,
            load_combinations=list(save_input.load_combinations),
            tanks=save_input.tanks,
            tank_grid=save_input.tank_grid,
            tank_search_data=save_input.tank_search_data,
            buckling_method=save_input.buckling_method,
            shifting=save_input.shifting,
            weight_and_cog=save_input.weight_and_cog,
        )

    @classmethod
    def save_path(cls, path: str | Path, save_input: ProjectSaveInput) -> ProjectSaveResult:
        state = cls.create_state(save_input)
        return ProjectSaveResult(
            path=ProjectPersistenceService.save_state_to_path(state, path),
            state=state,
        )


@dataclass(frozen=True)
class ProjectHydrationDefaults:
    """Legacy structure defaults that the UI used to inject while opening files."""

    structure_types: dict[str, Any]
    zstar_optimization: Any = True
    puls_buckling_method: Any = "ultimate"
    puls_boundary: Any = "Int"
    puls_stiffener_end: Any = "Continuous"
    puls_sp_or_up: Any = "SP"
    puls_up_boundary: Any = "SSSS"
    material_factor: Any = 1.15


@dataclass(frozen=True)
class ProjectHydrationResult:
    """Domain objects rebuilt from a saved project before Tk widgets are touched."""

    line_bundles: dict[str, list[Any]]
    load_assignments: dict[str, list[Any]]
    section_properties: tuple[dict[str, Any], ...]


class ProjectHydrationService:
    """Rebuild supported domain objects from saved project state."""

    LOAD_VARIABLES = (
        "poly_third",
        "poly_second",
        "poly_first",
        "poly_const",
        "load_condition",
        "structure_type",
        "man_press",
        "static_draft",
        "name_of_load",
        "limit_state",
        "slamming mult pl",
        "slamming mult stf",
    )

    @classmethod
    def hydrate_objects(cls, state: ProjectState, defaults: ProjectHydrationDefaults):
        line_bundles, section_properties = cls._hydrate_line_bundles(state, defaults)
        load_assignments = cls._hydrate_loads(state.loads, state.lines, line_bundles)
        return ProjectHydrationResult(
            line_bundles=line_bundles,
            load_assignments=load_assignments,
            section_properties=tuple(section_properties),
        )

    @classmethod
    def _hydrate_line_bundles(cls, state: ProjectState, defaults: ProjectHydrationDefaults):
        line_bundles = {}
        section_properties = []
        old_save_file = False

        for line_name, structure_properties in copy.deepcopy(state.structures).items():
            if len(structure_properties) > 10:
                old_save_file = True

            line_bundle = line_structure.LineStructureBundle(
                loads=[],
                load_combinations={},
            )

            if old_save_file:
                structure, fatigue, section_properties_for_line = cls._hydrate_legacy_flat_structure(
                    structure_properties,
                    state.fatigue.get(line_name),
                    defaults,
                )
            else:
                structure, fatigue, section_properties_for_line = cls._hydrate_structure(
                    structure_properties,
                    state.fatigue.get(line_name),
                )

            line_bundle.line_structure = structure
            line_bundle.fatigue = fatigue
            line_bundle.cylinder = cls._hydrate_cylinder(state.shell_structures.get(line_name))
            line_bundles[line_name] = line_bundle.to_legacy_bundle()
            if section_properties_for_line is not None:
                section_properties.append(section_properties_for_line)

        return line_bundles, section_properties

    @classmethod
    def _hydrate_legacy_flat_structure(cls, structure_properties, fatigue_properties, defaults):
        cls._add_legacy_structure_defaults(structure_properties, defaults)
        cls._split_legacy_sigma_x(structure_properties)

        main_dict = copy.deepcopy(example_data.prescriptive_main_dict)
        end_support_map = {"C": "Continuous", "S": "Sniped"}
        stiffener_end = structure_properties["puls stiffener end"][0]
        structure_properties["puls stiffener end"] = [
            end_support_map.get(stiffener_end, stiffener_end),
            structure_properties["puls stiffener end"][1],
        ]
        main_dict["material yield"] = [355e6, "Pa"]
        main_dict["load factor on stresses"] = [1, ""]
        main_dict["load factor on pressure"] = [1, ""]
        main_dict["buckling method"] = [structure_properties["puls buckling method"], ""]
        main_dict["stiffener end support"] = structure_properties["puls stiffener end"]
        main_dict["girder end support"] = ["Continuous", ""]

        domain = (
            "Flat plate, stiffened"
            if structure_properties["puls sp or up"][0] == "SP"
            else "Flat plate, unstiffened"
        )
        main_dict["calculation domain"] = [domain, ""]

        pressure_side_map = {"p": "plate side", "s": "stiffener side"}
        if "press_side" in structure_properties:
            pressure_side = structure_properties["press_side"][0]
            structure_properties["press_side"] = [pressure_side_map[pressure_side], ""]
        else:
            structure_properties["press_side"] = "both sides"
        structure_properties["panel or shell"] = "panel"

        structure = AllStructure(
            Plate=CalcScantlings(structure_properties),
            Stiffener=None if domain == "Flat plate, unstiffened" else CalcScantlings(structure_properties),
            Girder=None,
            main_dict=main_dict,
        )
        fatigue = (
            None
            if fatigue_properties is None
            else CalcFatigue(structure_properties, fatigue_properties)
        )
        return structure, fatigue, structure_properties

    @staticmethod
    def _hydrate_structure(structure_properties, fatigue_properties):
        structure = AllStructure(
            Plate=(
                None
                if structure_properties["Plate"] is None
                else CalcScantlings(structure_properties["Plate"])
            ),
            Stiffener=(
                None
                if structure_properties["Stiffener"] is None
                else CalcScantlings(structure_properties["Stiffener"])
            ),
            Girder=(
                None
                if structure_properties["Girder"] is None
                else CalcScantlings(structure_properties["Girder"])
            ),
            main_dict=structure_properties["main dict"],
        )
        fatigue = (
            None
            if fatigue_properties is None
            else CalcFatigue(structure_properties["Stiffener"], fatigue_properties)
        )
        return structure, fatigue, structure_properties["Stiffener"]

    @classmethod
    def _hydrate_cylinder(cls, cylinder_properties):
        if cylinder_properties is None:
            return None

        cylinder_properties = copy.deepcopy(cylinder_properties)
        for structure_type in ("Long. stf.", "Ring stf.", "Ring frame"):
            if cylinder_properties[structure_type] is not None:
                cls._split_legacy_sigma_x(cylinder_properties[structure_type])

        return CylinderAndCurvedPlate(
            cylinder_properties["Main class"],
            shell=(
                None
                if cylinder_properties["Shell"] is None
                else Shell(cylinder_properties["Shell"])
            ),
            long_stf=(
                None
                if cylinder_properties["Long. stf."] is None
                else Structure(cylinder_properties["Long. stf."])
            ),
            ring_stf=(
                None
                if cylinder_properties["Ring stf."] is None
                else Structure(cylinder_properties["Ring stf."])
            ),
            ring_frame=(
                None
                if cylinder_properties["Ring frame"] is None
                else Structure(cylinder_properties["Ring frame"])
            ),
        )

    @classmethod
    def _hydrate_loads(cls, load_properties, lines, line_bundles):
        load_assignments = {}
        for load_name, data in load_properties.items():
            values = list(data[0])
            if len(values) != len(cls.LOAD_VARIABLES):
                values.extend([1, 1])
            properties = dict(zip(cls.LOAD_VARIABLES, values))
            load_assignments[load_name] = [Loads(properties), data[1]]

            for line_name in lines:
                if line_name in data[1] and line_name in line_bundles:
                    line_structure.LineStructureBundle.from_legacy_bundle(
                        line_bundles[line_name]
                    ).loads.append(load_assignments[load_name][0])

        return load_assignments

    @staticmethod
    def _add_legacy_structure_defaults(structure_properties, defaults):
        default_values = {
            "structure_types": defaults.structure_types,
            "zstar_optimization": defaults.zstar_optimization,
            "puls buckling method": defaults.puls_buckling_method,
            "puls boundary": defaults.puls_boundary,
            "puls stiffener end": defaults.puls_stiffener_end,
            "puls sp or up": defaults.puls_sp_or_up,
            "puls up boundary": defaults.puls_up_boundary,
            "mat_factor": defaults.material_factor,
        }
        for property_name, property_value in default_values.items():
            if property_name not in structure_properties:
                unit = " " if property_name == "structure_types" else ""
                structure_properties[property_name] = [property_value, unit]

    @staticmethod
    def _split_legacy_sigma_x(properties):
        if "sigma_x" in properties:
            properties["sigma_x1"] = properties["sigma_x"]
            properties["sigma_x2"] = properties["sigma_x"]
            properties.pop("sigma_x")


@dataclass(frozen=True)
class ProjectOpenResult:
    """Assembled project payload ready for the UI open boundary."""

    state: ProjectState
    transfer: "ProjectOpenTransfer"
    hydration: ProjectHydrationResult


class ProjectOpenService:
    """Load and assemble project data before the UI applies it."""

    @classmethod
    def open_path(cls, path: str | Path, defaults: ProjectHydrationDefaults) -> ProjectOpenResult:
        return cls.assemble(
            ProjectPersistenceService.load_state_from_path(path),
            defaults,
        )

    @staticmethod
    def assemble(state: ProjectState, defaults: ProjectHydrationDefaults) -> ProjectOpenResult:
        return ProjectOpenResult(
            state=state,
            transfer=ProjectOpenTransferService.create_transfer(state),
            hydration=ProjectHydrationService.hydrate_objects(state, defaults),
        )


@dataclass(frozen=True)
class OpenLoadCombinationRecord:
    """Saved load-combination values ready for Tk variable creation."""

    name: tuple[Any, ...]
    static_factor: Any
    dynamic_factor: Any
    include: Any = None

    @property
    def has_include(self) -> bool:
        return self.include is not None

    @classmethod
    def from_legacy(cls, data: list[Any]) -> "OpenLoadCombinationRecord":
        return cls(
            name=tuple(data[0]),
            static_factor=data[1],
            dynamic_factor=data[2],
            include=None if len(data) < 4 else data[3],
        )


@dataclass(frozen=True)
class ProjectOpenTransfer:
    """Plain project values that `openfile` can apply at the Tk boundary."""

    project_information: str
    theme: str
    points: dict[str, Any]
    lines: dict[str, Any]
    shifting: dict[str, Any]
    accelerations: dict[str, Any]
    load_combinations: tuple[OpenLoadCombinationRecord, ...]
    tank_grid: Any
    tank_search_data: dict[int, Any] | None
    tank_properties: dict[str, Any]
    buckling_method: str
    weight_and_cog: dict[str, Any]


class ProjectOpenTransferService:
    """Normalize saved project values before Tk view state is populated."""

    DEFAULT_ACCELERATIONS = {"static": 9.81, "dyn_loaded": 0, "dyn_ballast": 0}

    @classmethod
    def create_transfer(cls, state: ProjectState) -> ProjectOpenTransfer:
        return ProjectOpenTransfer(
            project_information=state.project_information,
            theme=state.theme,
            points=state.points,
            lines=state.lines,
            shifting=state.shifting,
            accelerations={**cls.DEFAULT_ACCELERATIONS, **state.accelerations},
            load_combinations=tuple(
                OpenLoadCombinationRecord.from_legacy(data)
                for data in state.load_combinations.values()
            ),
            tank_grid=state.tanks.get("grid"),
            tank_search_data=cls._normalize_tank_search_data(state.tanks.get("search_data")),
            tank_properties={
                key: value
                for key, value in state.tanks.items()
                if key not in {"grid", "search_data"}
            },
            buckling_method=state.buckling_method,
            weight_and_cog=state.weight_and_cog,
        )

    @staticmethod
    def _normalize_tank_search_data(search_data):
        try:
            return {int(key): value for key, value in search_data.items()}
        except (AttributeError, TypeError, ValueError):
            return None
