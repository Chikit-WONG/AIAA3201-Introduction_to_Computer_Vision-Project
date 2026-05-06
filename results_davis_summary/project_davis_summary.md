# Full DAVIS Summary

This table follows the rerun protocol that keeps only DAVIS `JM` and `JR`.

| Part | Method | Variant | #Seq | Avg JM | Avg JR | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| part1 | spatial_only |  | 91 | 0.2435 | 0.2355 |  |
| part1 | temporal_aligned |  | 91 | 0.2435 | 0.2355 |  |
| part1 | temporal_no_align |  | 91 | 0.2435 | 0.2355 |  |
| part2 | sam2_propainter |  | 91 | **0.9167** | **0.9846** |  |
| part2 | vggt4d_propainter |  | 2 | 0.0000 | 0.0000 |  |
| part3 | sam3_diffueraser_object | sam3 | 90 | 0.7599 | 0.7943 | DAVIS rows reuse the same SAM 3 object masks; inpainting backend is not part of JM/JR. |
| part3 | sam3_diffueraser_side_effect | sam3 | 90 | 0.7599 | 0.7943 | DAVIS rows reuse the same SAM 3 object masks; inpainting backend is not part of JM/JR. |
| part3 | sam3_rose_object | sam3 | 90 | 0.7599 | 0.7943 | DAVIS rows reuse the same SAM 3 object masks; inpainting backend is not part of JM/JR. |
| part3 | sam3_rose_side_effect | sam3 | 90 | 0.7599 | 0.7943 | DAVIS rows reuse the same SAM 3 object masks; inpainting backend is not part of JM/JR. |

## Notes

- Best `JM`: part2 / sam2_propainter
- Best `JR`: part2 / sam2_propainter
