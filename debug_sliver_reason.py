import anystruct.fe_plate_fields as fp
res = fp.read_fea_shell_model('ref_Cases/barge.FEM')
patches = [p for p in fp.detect_surface_patches(res, decimals=3) if p.area > 1e-3 and len(p.element_ids) > 0]
all_fields = []
for i, p in enumerate(sorted(patches, key=lambda item: (-item.area, item.patch_id))):
    all_fields.extend(fp._split_sesam_patch_by_beams(res, p, i))
merged1 = fp._merge_sesam_curved_side_fields(res, all_fields)

field_by_id = {field_item.field_id: field_item for field_item in merged1}
def is_sliver(field_item) -> bool:
    if not (field_item.spacing_m < 0.30 or field_item.span_m < 0.45): return False
    if len(field_item.element_ids) > 4 and field_item.spacing_m >= 0.25: return False
    return True

changed = True
while changed:
    changed = False
    current_fields = list(field_by_id.values())
    edge_to_fields = fp.defaultdict(set)
    node_to_fields = fp.defaultdict(set)
    edges_by_field = {}
    nodes_by_field = {}
    for field_item in current_fields:
        edges = tuple(fp._field_shell_edges(res, field_item))
        nodes = fp._field_shell_nodes(res, field_item)
        edges_by_field[field_item.field_id] = edges
        nodes_by_field[field_item.field_id] = nodes
        for edge in edges: edge_to_fields[edge].add(field_item.field_id)
        for node_id in nodes: node_to_fields[node_id].add(field_item.field_id)

    ordered = sorted(current_fields, key=lambda item: (len(item.element_ids), item.spacing_m, item.span_m, item.field_id))
    for field_item in ordered:
        if field_item.field_id not in field_by_id or not is_sliver(field_item): continue
        
        neighbour_scores = fp.defaultdict(int)
        for edge in edges_by_field.get(field_item.field_id, ()):
            for neighbour_id in edge_to_fields.get(edge, ()):
                if neighbour_id != field_item.field_id and neighbour_id in field_by_id:
                    neighbour_scores[neighbour_id] += 10
        if not neighbour_scores: continue
        
        field_normal = fp._field_representative_normal(res, field_item)
        field_thickness = field_item.shell_section_thickness_m or 0.0

        def neighbour_score(neighbour_id: str):
            neighbour = field_by_id[neighbour_id]
            neighbour_normal = fp._field_representative_normal(res, neighbour)
            return (abs(fp._dot(field_normal, neighbour_normal)) >= 0.70, abs((neighbour.shell_section_thickness_m or 0.0) - field_thickness) <= 0.002, not is_sliver(neighbour), neighbour_scores[neighbour_id], len(neighbour.element_ids), -abs((neighbour.shell_section_thickness_m or 0.0) - field_thickness))
        
        target_id = max(neighbour_scores, key=neighbour_score)
        
        target = field_by_id[target_id]
        if len(target.element_ids) > 50:
            print(f"Target before merge: {len(target.element_ids)} elements, span: {target.span_m:.3f}, spacing: {target.spacing_m:.3f}")
            new_target = fp._merge_sesam_general_fields(res, target, field_item, "test")
            print(f"Target AFTER merge: {len(new_target.element_ids)} elements, span: {new_target.span_m:.3f}, spacing: {new_target.spacing_m:.3f}")
            if is_sliver(new_target):
                print(f"  -> BECAME A SLIVER! (spacing: {new_target.spacing_m:.3f})")
        
        field_by_id[target_id] = fp._merge_sesam_general_fields(res, field_by_id[target_id], field_item, "test")
        del field_by_id[field_item.field_id]
        changed = True
        break
