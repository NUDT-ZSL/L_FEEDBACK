# Validation report

- Scene: `/home/liang/feedback/uid78/sample_scene.json`
- Live image: `/home/liang/feedback/uid78/artifacts/validation_run/live.png`
- Snapshot restore difference: 0/255
- Add light: v1 -> v2, 65 point lights, old hash preserved
- Remove light: v2 -> v3, back to 64 point lights, image difference 0/255
- Corruption codes: ERR_SNAPSHOT_CORRUPT, ERR_SNAPSHOT_CORRUPT, ERR_SNAPSHOT_VERSION, ERR_SNAPSHOT_NOT_FOUND
- Performance: CPU ratio 0.0081, GPU ratio 0.0758 (72 lights)
- AddressSanitizer: run, no application/sanitizer errors

Machine-readable details: `/home/liang/feedback/uid78/validation/report.json`
