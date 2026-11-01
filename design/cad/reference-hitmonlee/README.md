# Hitmonlee reference model

This is an appearance reference imported into FreeCAD. It is not a robot CAD part, mechanical design, or source of dimensional requirements.

## Source and attribution

- Original: [Hitmonlee model by nguyenlouis32 on Sketchfab](https://sketchfab.com/3d-models/hitmonlee-b7534fa6f08e4760839df0f80f00e072)
- Distribution mirror: [Pokemon-3D-api/assets](https://github.com/Pokemon-3D-api/assets)
- Pinned repository commit: `429de1288cea0d43f5b4f56305d2276e94239d65`
- Upstream path: `models/opt/regular/106.glb`
- GLB metadata license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Original GLB SHA-256: `A11A7365AF8BA1D899359C3D964618C98F6037AF85C7A713C380A9016EFA86CE`

The upstream repository notes that Pokémon character assets remain subject to their underlying rights. The user confirmed applicable licenses are available for this project. Keep this provenance with the asset and confirm rights for any external distribution.

## Files and pipeline

- `assets/reference_models/pokemon-3d-api/hitmonlee_106.glb` — pinned, textured source (two meshes; no animation clips).
- `assets/reference_models/pokemon-3d-api/hitmonlee_106_decoded.glb` — decoded geometry; created because the source GLB uses required Draco compression that FreeCAD did not import directly.
- `assets/reference_models/pokemon-3d-api/hitmonlee_106.obj` — geometry-only bridge format; 6,147 vertices and 9,492 triangles. Materials and textures remain in the source GLB.
- `design/cad/reference-hitmonlee/hitmonlee_appearance_reference.FCStd` — native FreeCAD mesh reference with separately selectable body and eye meshes.
- `design/cad/reference-hitmonlee/hitmonlee_appearance_reference.png` — FreeCAD shaded preview.

Recreate the conversion with Node.js/npm:

```powershell
npx --yes @gltf-transform/cli@4.5.1 copy assets/reference_models/pokemon-3d-api/hitmonlee_106.glb assets/reference_models/pokemon-3d-api/hitmonlee_106_decoded.glb
python tools/cad/convert_glb_to_obj.py assets/reference_models/pokemon-3d-api/hitmonlee_106_decoded.glb assets/reference_models/pokemon-3d-api/hitmonlee_106.obj
```

Then open `design/cad/reference-hitmonlee/import_reference.FCMacro` in FreeCAD and run it to regenerate the `.FCStd` reference and preview. The OBJ is imported as a triangulated mesh; it is useful for silhouette and interference context, not feature-based editing or manufacturing. The upstream scene's source units and mechanical scale are not established here, so do not use its raw coordinates as build dimensions.
