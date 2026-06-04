import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from anystruct import solid_export


def test_mesh_dict_to_triangular_mesh_welds_vertices_and_triangulates_quads():
    mesh = {
        "name": "solid test",
        "vertices": [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
        ],
        "faces": [[1, 2, 3, 4], [5, 2, 3]],
    }

    tri_mesh = solid_export.mesh_dict_to_triangular_mesh(mesh)

    assert tri_mesh.name == "solid_test"
    assert tri_mesh.points.shape == (4, 3)
    assert tri_mesh.triangles.shape == (3, 3)
    assert tri_mesh.triangles.max() == 3
    assert tri_mesh.triangles[-1].tolist() == [0, 1, 2]


def test_connected_component_count_detects_separate_solid_parts():
    mesh = {
        "vertices": [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (5.0, 0.0, 0.0),
            (6.0, 0.0, 0.0),
            (5.0, 1.0, 0.0),
        ],
        "faces": [[1, 2, 3], [4, 5, 6]],
    }

    assert solid_export.connected_component_count(mesh) == 2


def test_write_meshio_writes_surface_mesh(tmp_path):
    filename = tmp_path / "solid_mesh.vtk"
    mesh = solid_export.SolidTriangleMesh(
        points=np.asarray([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int64),
        name="meshio_test",
    )

    solid_export.write_meshio(filename, mesh)

    assert filename.exists()
    assert filename.stat().st_size > 0


def test_write_numpy_stl_uses_numpy_stl_backend(monkeypatch, tmp_path):
    saved = {}

    class FakeMode:
        BINARY = "binary"
        ASCII = "ascii"

    class FakeMesh:
        dtype = np.dtype([("vectors", float, (3, 3))])
        Mode = FakeMode

        def __init__(self, data):
            self.data = data
            self.vectors = np.zeros((len(data), 3, 3), dtype=float)

        def save(self, filename, mode=None):
            saved["filename"] = filename
            saved["mode"] = mode
            saved["vectors"] = self.vectors.copy()

    fake_stl = ModuleType("stl")
    fake_stl.Mode = FakeMode
    fake_stl.mesh = SimpleNamespace(Mesh=FakeMesh, Mode=FakeMode)
    monkeypatch.setitem(sys.modules, "stl", fake_stl)

    mesh = solid_export.SolidTriangleMesh(
        points=np.asarray([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int64),
        name="stl_test",
    )
    filename = tmp_path / "solid.stl"

    solid_export.write_numpy_stl(filename, mesh, binary=False)

    assert saved["filename"] == str(filename)
    assert saved["mode"] == "ascii"
    assert saved["vectors"].shape == (1, 3, 3)


def test_write_solid_export_rejects_unknown_backend(tmp_path):
    mesh = {
        "vertices": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        "faces": [[1, 2, 3]],
    }

    try:
        solid_export.write_solid_export(tmp_path / "out.fake", mesh, backend="unknown")
    except solid_export.SolidExportError as exc:
        assert "Unsupported solid export backend" in str(exc)
    else:
        raise AssertionError("Expected SolidExportError")
