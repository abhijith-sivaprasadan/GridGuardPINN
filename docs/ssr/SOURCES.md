# Source ledger: IEEE FBM

IEEE Subsynchronous Resonance Task Force of the Dynamic System Performance
Working Group, Power System Engineering Committee, **“First Benchmark Model for
Computer Simulation of Subsynchronous Resonance,”** *IEEE Transactions on Power
Apparatus and Systems*, PAS-96(5), September/October 1977, pp. 1565–1572.
DOI: [10.1109/T-PAS.1977.32485](https://doi.org/10.1109/T-PAS.1977.32485).
[Publisher record](https://ieeexplore.ieee.org/document/1602087).

The original eight-page scan was accessed through the
[public scan viewer](https://studylib.net/doc/18154805/1977-first-benchmark-model-for-c)
on 2026-10-11. Pages 1565–1567 were rendered and visually inspected, including
the drawings and tables; this transcription does not rely on the viewer's OCR.
The downloaded research copy has SHA-256
`e26b07a28f020f34078e29b89c0450667c9aa9c7819c210ef996f90a577821cb`.
The copyrighted scan is not redistributed in this repository.

## Parameter transcription

All values below are on the machine base, **892.4 MVA, 60 Hz** (p. 1565).
The implementation uses `2*pi*60` for angular frequency; the printed `377 R`
resistance column is divided by **377**, as printed, not by a fitted divisor.

| Quantity | Values | Exact locator |
|---|---|---|
| Mass order | HP, IP, LPA, LPB, GEN, EXC | Fig. 2, p. 1565 |
| H, seconds | 0.092897, 0.155589, 0.858670, 0.884215, 0.868495, 0.0342165 | Table III, p. 1566 |
| Successive shaft stiffness, pu torque/rad | 19.303, 34.929, 52.038, 70.858, 2.822 | Table III, rightmost column |
| Alternative printed stiffness, base angle 377 rad | 7277, 13168, 19618, 26713, 1064 | Table III, middle column; not mixed with rightmost column |
| Turbine torque shares, HP/IP/LPA/LPB | 30%, 26%, 22%, 22%; exciter zero | Paragraph below Table III discussion, p. 1566 |
| Transient mechanical damping | Zero | p. 1566, immediately below shaft torque equation |
| D-axis: 377 Rf, Xf, 377 Rk, Xk, Xa, XL | 0.53, 0.062, 1.54, 0.0055, 1.66, 0.13 | Table I, p. 1566 |
| Q-axis: 377 Rf, Xf, 377 Rk, Xk, Xa, XL | 5.3, 0.326, 3.1, 0.095, 1.58, 0.13 | Table I |
| Armature resistance R1 | 0.0 | Table II, p. 1566 |
| Positive-sequence transformer / line / system X | 0.14 / 0.50 / 0.06 | Fig. 1, p. 1565 |
| Line positive-sequence R | 0.02 | Fig. 1 |
| Generator terminal voltage | Unity | Fig. 1 label below generator |
| P0, power factor | 0.9 pu, 0.9 lagging | Table VI, p. 1567 |
| Xc, fault Xf, location | 0.371, 0.04, B | Table VI, Case 1-T |
| First phase clearing delay | 0.075 s; remaining phases at subsequent current zero | Table VI |
| Capacitor bypass / reinsertion gaps | Not used | Table VI |

The transient model uses the **Table I rotor circuits**, not an approximate
fit to the Table II reactances/time constants. For cross-reading, Table II
prints Xd=1.79, Xd'=0.169, Xd''=0.135, Xq=1.71, Xq'=0.228,
Xq''=0.200; Td0'=4.3 s, Td0''=0.032 s, Tq0'=0.85 s, Tq0''=0.05 s.

## Independent reference values and their limits

| Mode | Published frequency, Hz | Published no-load decrement, 1/s | Locator |
|---|---:|---:|---|
| 1 | 15.71 | 0.05 | Fig. 4, p. 1567; frequencies also Table IV |
| 2 | 20.21 | 0.11 | Same |
| 3 | 25.55 | 0.028 | Same |
| 4 | 32.28 | 0.028 | Same |
| 5 | **Unverified** | **Unverified** | No fifth reference value in these tables |

Fig. 4 defines decrement `sigma = Dn/(4 Hn)` and undamped frequency
`2*pi*fn = sqrt(Kn/(2 Hn))`. Decrement is not damping ratio.
The corresponding ratio is `sigma / sqrt(sigma^2 + (2*pi*f_damped)^2)`.
Published frequencies are rounded to 0.01 Hz. Published no-load decrement is
load dependent; the paper says full-load values can be up to twenty times
larger. It does not provide unique individual shaft/turbine damping constants.

The separate no-load validation therefore constructs a modal damping matrix
from the four published decrements. Estimating those decrements tests the
estimator against **configured literature inputs**, not an independent
prediction of damping by the electrical model. The transient circuit instead
uses the explicitly specified zero mechanical damping. Its coupled electrical
damping is not expected to equal either reference decrement set.

The paper states Case 1-T is tuned to **approximately 40 Hz** electrical
resonance during and after the fault (p. 1567). This is the independently
confirmed electrical reference; no exact compensation-to-frequency curve is
claimed. `60*sqrt(Xc/Xtotal)` is an elementary ideal RLC approximation, not
an extracted FBM curve: the generator impedance is frequency dependent.

No tabulated modal amplitude reference was found. Amplitudes are checked
against separate analytic signals; they are **not validated against published
plant amplitudes**. Case 1-S uses a different frequency-fitted rotor circuit
(Tables IV/V), so its published self-excitation decrements must not be used
as ground truth for this Table I transient model.

## ANDES capability audit

Inspected ANDES **v2.0.0**, commit
`eda5163c9ee8d19945a1dd5d1771fec5da608c27`:

- [SHAFT5 source](https://github.com/CURENT/andes/blob/eda5163c9ee8d19945a1dd5d1771fec5da608c27/andes/models/shaft/shaft5.py)
  implements HP–IP–LP–Rotor–EX: five masses, one LP, using the generator's
  existing rotor states. It cannot represent this six-mass chain unchanged.
- [Line source](https://github.com/CURENT/andes/blob/eda5163c9ee8d19945a1dd5d1771fec5da608c27/andes/models/line/line.py)
  implements an algebraic AC pi branch/transformer using bus voltage and angle
  variables. It has no dynamic series-capacitor voltage or inductor-current
  states. An algebraic negative reactance alone is insufficient for this task.
- Official [shaft documentation](https://docs.andes.app/en/v2.0.0/reference/models/DynShaft.html)
  and [ACLine documentation](https://docs.andes.app/en/v2.0.0/reference/models/ACLine.html).

The new simulator is consequently a **self-contained custom SciPy ODE**.
It reuses GridGuardPINN packaging and CI, but does not call ANDES or claim
an ANDES-native FBM implementation. Existing ANDES functionality is unchanged.
