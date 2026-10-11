"""Transcription of the original IEEE FBM; see docs/ssr/SOURCES.md."""

import numpy as np

BASE_HZ = 60.0
OMEGA_BASE = 2 * np.pi * BASE_HZ
BASE_MVA = 892.4
MASSES = ("HP", "IP", "LPA", "LPB", "GEN", "EXC")
# Table III, p. 1566: H [seconds] and stiffness [pu torque / rad].
H = np.array([0.092897, 0.155589, 0.858670, 0.884215, 0.868495, 0.0342165])
STIFFNESS = np.array([19.303, 34.929, 52.038, 70.858, 2.822])
TORQUE_SHARES = np.array([0.30, 0.26, 0.22, 0.22, 0.0, 0.0])
# Table I: resistance column is 377 R, NOT R. Keep the printed divisor.
XD_LEAKAGES = np.array([0.13, 0.062, 0.0055])  # stator, field, damper
XQ_LEAKAGES = np.array([0.13, 0.326, 0.095])
XD_MUTUAL, XQ_MUTUAL = 1.66, 1.58
RD = np.array([0.53, 1.54]) / 377.0
RQ = np.array([5.3, 3.1]) / 377.0
ARMATURE_R = 0.0  # Table II
# Fig. 1 positive-sequence network and Table VI Case 1-T.
TRANSFORMER_X, LINE_X, SYSTEM_X, LINE_R = 0.14, 0.50, 0.06, 0.02
CAPACITOR_X, FAULT_X = 0.371, 0.04
POWER, POWER_FACTOR, FAULT_DURATION = 0.9, 0.9, 0.075
# Fig. 4 / Table IV, p. 1567. Only four modes are independently tabulated.
PUBLISHED_HZ = np.array([15.71, 20.21, 25.55, 32.28])
NO_LOAD_DECREMENT = np.array([0.05, 0.11, 0.028, 0.028])  # 1/s, not zeta


def stiffness_matrix():
    """Free-free, six-mass chain, including the exciter."""
    incidence = np.diff(np.eye(6), axis=0)
    return incidence.T @ np.diag(STIFFNESS) @ incidence
