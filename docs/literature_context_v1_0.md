# External literature context (verified references, 9 October 2026)

This note distinguishes **independent published background** from **GridGuardPINN's original numerical evidence**. It is a targeted starting bibliography, **not a systematic review** and not a claim that the project's methods are unprecedented.

1. **Raissi, M.; Perdikaris, P.; Karniadakis, G. E. (2019).** “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.” *Journal of Computational Physics* **378**, 686–707. DOI: [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045). Establishes the broader combined data-and-differential-equation loss approach underpinning the term PINN. **Not** evidence for GridGuardPINN's IEEE-14 performance.

2. **Cui, H.; Li, F.; Tomsovic, K. (2021).** “Hybrid Symbolic-Numeric Framework for Power System Modeling and Analysis.” *IEEE Transactions on Power Systems* **36**(2), 1373–1384. DOI: [10.1109/TPWRS.2020.3017019](https://doi.org/10.1109/TPWRS.2020.3017019). Foundational reference for the [ANDES simulator](https://github.com/CURENT/andes) used in GridGuardPINN. Cite this work when describing the reference solver. ANDES is an independent external project.

3. **Xu, X.; Qiang, K.; Wu, J.; Xuan, L.; Lu, Y.; Ma, W.; Zhang, J. (2026).** “Physics-informed neural networks for power systems: A systematic review.” *Sustainable Energy, Grids and Networks*. DOI: [10.1016/j.segan.2026.102261](https://doi.org/10.1016/j.segan.2026.102261). Reviews PINN applications in device- and system-level power systems, including dynamic simulation and stability. It is **not** an empirical comparison with GridGuardPINN.

## How this relates to the contribution

The paper should describe GridGuardPINN's **specific experimental contribution** as a bounded comparison of trained surrogate accuracy, domain-aware refusal, risk from residual-only routing and honest end-to-end cost in a frozen IEEE-14 setup. This is narrower than proposing PINNs for the first time and narrower than establishing guaranteed transient-stability screening.

## Gaps before external academic submission

Expand the literature review to focused neural transient-stability surrogates, selective prediction/abstention, simulation-based model validation, and calibrated out-of-distribution rejection. Compare protocols and metrics directly rather than asserting priority or state-of-the-art from these three background sources.
