# v1.0 release evidence manifest

| Evidence | Frozen reference | Outcome |
|---|---|---|
| Original trained weights | [Run 37922112172](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37922112172) | Three-seed one-hot and topology-aware v0.3 experiment |
| Original artifact ZIP SHA-256 | `02a82e4d53544203376749e7054778b9b93a2a5d6af3c83ab7d2820fd24f6666` | Digest-pinned in benchmark workflow |
| Real ANDES runtime smoke | [Run 37924049053](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37924049053) | Passed |
| Trained-model integration | [Run 37924679301](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37924679301) | Passed, 13 surrogate / 92 reference, zero observed false accepts |
| New-duration preregistration | [Protocol](andes_v1_fresh_duration_protocol.md) | Frozen before experiment |
| Fresh-duration release gate | [Run 37925430499](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37925430499) | Passed: 76/84 interpolation accepted, 0 false accepts, 21/21 new duration-OOD fallback |
| Frozen results and caveats | [Results](results_andes_v1_fresh_duration.md) | Documented |
| Routing runtime | [Source](../src/gridguardpinn/andes_runtime.py) | Model residual, deterministic route, ANDES fallback |
| Reproducibility | [Runtime guide](v1_0_runtime.md) | Checkpoint digests and CLI |

Release scope is a **research demonstrator on the frozen IEEE-14 case**. Artifacts in GitHub Actions are not permanent until mirrored as release assets with verified hashes. This manifest records evidence, not a production safety certification.
