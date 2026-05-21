import anystruct.example_data as ex
import anystruct.line_structure as line_structure


def test_line_structure_accessors_read_existing_bundle_slots():
    line_bundle = ex.get_line_to_struc()["line1"]

    assert line_structure.structure(line_bundle) is line_bundle[0]
    assert line_structure.plate(line_bundle) is line_bundle[0].Plate
    assert line_structure.stiffener(line_bundle) is line_bundle[0].Stiffener
    assert line_structure.girder(line_bundle) is line_bundle[0].Girder
    assert line_structure.fatigue(line_bundle) is line_bundle[2]
    assert line_structure.loads(line_bundle) is line_bundle[3]
    assert line_structure.load_combinations(line_bundle) is line_bundle[4]
    assert line_structure.cylinder(line_bundle) is None
    assert line_structure.has_stiffener(line_bundle)
    assert not line_structure.has_cylinder(line_bundle)


def test_line_structure_copy_bundle_preserves_shape_and_copies_objects():
    line_bundle = ex.get_line_to_struc()["line1"]

    copied_bundle = line_structure.copy_bundle(line_bundle)

    assert len(copied_bundle) == len(line_bundle)
    assert copied_bundle is not line_bundle
    assert line_structure.structure(copied_bundle) is not line_structure.structure(line_bundle)


def test_typed_line_bundle_adapter_preserves_legacy_shape():
    line_bundle = ex.get_line_to_struc()["line1"]

    typed = line_structure.LineStructureBundle.from_legacy_bundle(line_bundle)

    assert typed.line_structure is line_bundle[0]
    assert typed.loads is line_bundle[3]
    assert typed.load_combinations is line_bundle[4]
    assert typed.to_legacy_bundle() == line_bundle
