# Model, protocol, and claim boundaries

This is a six-mass implementation of the original FBM **transient electrical
circuits and shaft**, with an **approximate balanced fault**. It is not an
exact reproduction of the paper's phase-by-phase Case 1-T switching waveforms.
It is synthetic benchmark work, with no real plant or PMU data validation.
See [source ledger](SOURCES.md) for every parameter and independent reference.

## Equations and conventions

Let `theta` be six electrical angles relative to a frame rotating at 60 Hz,
`w` six speed deviations in pu, `Omega=2*pi*60`, `M=diag(2H)` and `K` the
free-free stiffness matrix formed from the five torque/radian stiffnesses.
The transient mechanics are

```
theta_dot = Omega*w
M*w_dot = torque_shares*P0 - K*theta - e_GEN*Te
```

Mechanical damping is zero as specified for transient studies. There is no
governor, AVR, protection, saturation, shaft yielding, or mechanical limit.
The exciter has inertia but zero applied torque. Electrical torque acts on GEN.

In each rotor d/q axis, let `i` be stator current positive **out** of the
generator and `r=[i_f,i_k]` positive rotor winding currents. With mutual
reactance `Xa` and leakage reactances from Table I,

```
L = diag([XL, Xf, Xk]) + Xa*ones(3,3)
[psi_s, psi_f, psi_k] = L @ [-i, i_f, i_k]
psi_f_dot = Omega*(E_f - Rf*i_f)
psi_k_dot = -Omega*Rk*i_k
```

`E_f` is constant on the d axis and zero on the q axis. The q-axis two rotor
circuits are retained; their `f` label does not imply an energized q field.
Define `J[d,q]=[-q,d]`, actual GEN speed `omega=1+w_GEN`, terminal vector `v`,
series capacitor vector `vc`, and rotor-frame infinite-bus vector `v_inf`:

```
v = -Ra*i + psi_s_dot/Omega + omega*J*psi_s
v = v_inf + Rline*i + Xexternal*(i_dot/Omega + omega*J*i) + vc
vc_dot = Omega*(Xc*i - omega*J*vc)
Te = psi_d*i_q - psi_q*i_d
```

Rotor fluxes are states, rotor currents are obtained by solving the two
2-by-2 inductance systems, and the two voltage equations eliminate terminal
voltage to give explicit current derivatives. The circuit retains stator,
rotor, inductor and capacitor dynamics. Total **20 states**: six angles,
six speeds, two stator currents, four rotor fluxes, two capacitor voltages.

Initialization fixes generator terminal voltage at 1 pu, P=0.9 pu and power
factor=0.9 lagging. The infinite-bus voltage magnitude/angle and constant field
voltage are solved consistently; the infinite bus is not arbitrarily forced
to 1 pu. Shaft twists balance the published turbine torque shares.

## Fault approximation and numerical integration

A simultaneous balanced fault is applied at 0.5 s and cleared at 0.575 s.
At location B, the positive-sequence Thevenin approximation has
`V_inf_fault=V_inf*Xf/(Xs+Xf)` and `Xs_fault=Xs*Xf/(Xs+Xf)`.
The external series reactance is transformer + line + this system reactance.
Currents, fluxes, shaft states and capacitor voltages remain continuous when
the coefficients switch.

**This is a balanced Thevenin fault approximation.** It does not represent
the separate fault-inductor circulating current, phase-current-zero clearing,
the specified point-on-wave inception, unbalanced clearing intervals, or
switching energy in removed branches. It must not be presented as a verified
EMT replication of the original Fig. 5 waveforms. The intact-network ODE's
energy balance is tested; that test does not validate the switching surrogate.
The actual Case 1-T timing and omitted details are recorded in SOURCES.md.

`solve_ivp(DOP853)` integrates separate segments at each switching time,
relative tolerance 1e-8, absolute tolerance 1e-10, maximum step 1/2400 s.
Output is first evaluated at 2400 samples/s. Polyphase FIR low-pass resampling
produces **240 reports/s** (120 also supported); complex synchronous-frame
terminal voltage is filtered before magnitude and angle differentiation.
Reported frequency is `60 + derivative(unwrapped voltage angle)/(2*pi)`;
values below 0.05 pu voltage are marked unavailable. This is a synthetic
space-vector/envelope measurement, **not** an IEEE C37.118-compliant PMU
estimator, calibrated sensor, or field measurement. Filtering is offline and
noncausal; fault edges and record edges are excluded from estimation.

60 reports/s cannot represent the 32.28 Hz mode (or computed fifth mode
around 47.46 Hz). The default analysis band is 5–55 Hz, and the estimators
reject a sampling rate that violates its Nyquist bound. The higher stated
reporting rate is deliberate; no claim that ordinary 30/60 fps PMUs resolve
all these modes is made.

Default fault duration of the **record** is 3 seconds. An exploratory 8-second
run drove the ideal undamped exciter beyond zero speed (range approximately
-2.95 to 4.90 pu); such an extrapolation is not a physical turbine prediction.
Even the shorter record has large excursions; its maximum speed deviation is
reported. The code does not convert that trajectory into a plant risk rating.

## Estimator and comparison protocol

1. Independent mechanical references: Table III determines the full shaft
   eigenproblem. Fig. 4's separately printed frequencies supply the comparison.
   A GEN speed impulse of 0.001 pu excites a 12-second shaft-only ringdown.
   One experiment has zero damping; a second applies the four published
   no-load modal decrements. Rigid and fifth-mode damping are set to zero;
   the latter is an **unverified assumption**, not a published no-load value.
2. Matrix pencil: real signal, mean removed, Hankel rows=min(160,N//3), truncated
   SVD, 12 poles for the shaft experiment and 20 for fault signals. Relative
   singular cutoff 1e-10. Discrete poles are mapped with `log(z)*fs`; a complex
   exponential least-squares fit over the full window supplies amplitudes.
   No published frequencies are passed to either estimator.
3. Positive reported decay means damping; negative means growth. Pencil
   amplitude is the peak sinusoidal envelope at the **analysis-window start**.
   Damping ratio is decay divided by the complex eigenvalue magnitude.
4. Independent Welch baseline: Hann window, segment length N//2, half overlap,
   1% of maximum in-band PSD prominence threshold, log-parabolic peak
   interpolation. Amplitude is sqrt(2) times local-band spectral RMS and
   therefore differs from pencil's initial amplitude for a ringdown.
   Half-power widths are saved as diagnostics. They are **not** relabeled
   as damping: finite-window and nonstationarity effects prevent an independent
   decay estimate here. Welch decay/ratio fields explicitly contain null.
5. Compare all four published modes, using one-to-one nearest matching within
   0.5 Hz. Preserve missed modes, duplicate/ambiguous candidates and unmatched
   estimates. Frequency errors are absolute and percent. Zero-reference decay
   percentage error is undefined and stored as null. Frequency tolerances in
   CI are 0.005 Hz for the source-rounded shaft values, not tuned to fault data.
6. Fault analysis window: 1.0–2.75 s. Analyse generator speed, electrical torque,
   and terminal voltage magnitude; export all six speeds and bus frequency too.
   Comparisons with bare-shaft frequencies are **proximity only**, not coupled
   ground-truth validation. Electrical coupling shifts frequency and damping.
   The original paper's separate Case 1-S decrement cannot validate these fits.

No compensation sweep, uncertainty bounds, protection design or field-risk
classification is claimed. A useful next research step is exact phase-domain
fault clearing and a second independently implemented EMT simulator comparison.

## Reproduction

```
python -m pip install -e ".[dev]"
python -m pytest tests/test_ssr_estimator.py
python scripts/run_ssr_benchmark.py
```

The last command writes `artifacts/ssr/results.json` and `fault_signals.csv`.
`--shaft-only` skips the fault circuit; `--output PATH` selects another folder.
No copyrighted source download, ANDES, PyTorch, external data, or network
connection is needed after package installation. Floating-point results may
vary slightly by BLAS/runtime; tests use physical tolerances.

## Verification matrix

| Claim | Evidence | Limit |
|---|---|---|
| Original parameters transcribed | Rendered original Tables I–III/VI, Fig. 1; source ledger | Owner review pending |
| Shaft frequencies agree with publication | Four eigenfrequencies within 0.005 Hz | Fifth reference unverified |
| Pencil estimates frequency, decay, amplitude | Analytic mixtures, growing/zero/decaying modes, seeded noise | No field-noise robustness claim |
| Published no-load decrements recovered | Separate shaft ringdown | Decrements are configured, not independently predicted |
| Independent spectral frequency cross-check | Welch peaks, explicit misses and amplitudes | No independently identifiable Welch damping |
| Circuit equilibrium | Published operating point, small residual | Initialization alone is not dynamic validation |
| Intact coupled equations conserve energy | Off-equilibrium input/loss balance test | Does not certify fault switching |
| Numerical fault solution converges | Smaller step and tighter tolerance, speed/torque comparison | Self-consistency, not external EMT validation |
| Hosted portability | SSR job on Ubuntu/Windows/macOS with uploaded results | See PR's actual Actions checks |
| Plant/PMU applicability | None | No field validation or PMU compliance |

## AI disclosure and review ownership

Architecture/problem framing is supplied by Abhijith's build specification;
implementation, source transcription, tests and numerical checks are
AI-assisted. The project owner remains responsible for verification and
interpretation, consistent with the repository disclosure. **Independent
numerical review by Abhijith is pending.** This document does not assert that
he has already checked the results or approved their use in an application.
