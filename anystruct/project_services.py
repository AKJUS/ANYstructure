"""Application services for project edits that do not depend on Tkinter."""

from dataclasses import dataclass
from typing import Any, Callable, MutableMapping

from . import line_structure
from .helper import one_load_combination
from .project_application import (
    LoadCombinationRecord,
    OpenLoadCombinationRecord,
    ProjectHydrationDefaults,
    ProjectHydrationResult,
    ProjectHydrationService,
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
    def read_path(cls, path: str, workbook_factory: Callable[..., Any] | None = None):
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
    def open_example_path(cls, path: str, workbook_factory: Callable[..., Any] | None = None):
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
        return workbook_factory(path, visible=visible, read_only=read_only)


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
