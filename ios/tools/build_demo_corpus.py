#!/usr/bin/env python3
"""
Build a small SYNTHETIC demo corpus for smoke-testing the AW139 Diagnostics
iPad app with NO model files.

It embeds a handful of realistic AW139 maintenance snippets using the SAME
deterministic hashing algorithm as HashingEmbedder.swift (FNV-1a 32-bit →
bucket, L2-normalized). Turn on "Demo mode" in the app's Setup screen, transfer
the output onto the iPad as corpus.json, and the full retrieval + UI pipeline
works offline with zero models.

This is NOT semantic — it matches on shared keywords only. For production use
build the real corpus with build_offline_corpus.py + an MLX embedding model.

Usage:
    python build_demo_corpus.py --output demo-corpus.json --dim 384
"""

import argparse
import json

DIM_DEFAULT = 384


def fnv1a32(s: str) -> int:
    """FNV-1a 32-bit. Must match HashingEmbedder.fnv1a32 in Swift."""
    h = 2166136261
    for b in s.encode("utf-8"):
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def tokens(text: str):
    """Lowercase, split on non-alphanumerics, keep tokens length >= 2."""
    out, cur = [], []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if len(cur) >= 2:
                out.append("".join(cur))
            cur = []
    if len(cur) >= 2:
        out.append("".join(cur))
    return out


def embed(text: str, dim: int):
    v = [0.0] * dim
    for tok in tokens(text):
        v[fnv1a32(tok) % dim] += 1.0
    norm = sum(x * x for x in v) ** 0.5
    if norm > 0:
        v = [x / norm for x in v]
    return [round(x, 6) for x in v]


# (doc_path, ata, text) — DMC-style paths so DMC.extractCode works in the app.
DOCS = [
    ("dmc-39-A-24-32-00-00A-320A-A.xml", "24",
     "Generator voltage fluctuating fault isolation procedure. The DC generator "
     "control unit GCU monitors output voltage. If the generator voltage is "
     "unstable, check the voltage regulator setpoint, inspect the generator "
     "field winding continuity, and verify the GCU connector pins. Replace the "
     "voltage regulator A76 if adjustment does not stabilize output."),

    ("dmc-39-A-31-41-00-00A-340A-A.xml", "31",
     "Altimeter zero altitude adjustment. If the altimeter does not show zero "
     "feet on ground, perform the calibration. Locate the ZERO ALTITUDE "
     "ADJUSTMENT screw on the rear of the instrument and adjust until the "
     "indicator reads zero. Allow a five minute warmup before adjustment."),

    ("dmc-39-A-30-42-00-00A-320A-A.xml", "30",
     "HEATER FAIL CAS message fault isolation. The pitot heater control unit "
     "reports HEATER FAIL when it detects an open or shorted heating element. "
     "Identify the affected probe, test the heater element resistance, inspect "
     "the connector and wiring, and replace the pitot probe if the element is "
     "open circuit."),

    ("dmc-39-A-32-10-00-00A-320A-A.xml", "32",
     "Landing gear retraction troubleshooting. If the landing gear does not "
     "retract, check the hydraulic system pressure, verify the gear selector "
     "valve operation, and inspect the downlock microswitches. Test continuity "
     "of the squat switch circuit."),

    ("dmc-39-A-28-21-00-00A-320A-A.xml", "28",
     "Fuel low pressure warning fault isolation. The FUEL LOW PRESS caution "
     "indicates the boost pump output is below threshold. Check the boost pump "
     "operation, inspect the fuel filter for blockage, and test the pressure "
     "switch. Replace the boost pump if output pressure is insufficient."),

    ("dmc-39-A-29-31-00-00A-320A-A.xml", "29",
     "Hydraulic system low pressure procedure. If hydraulic pressure is low, "
     "check the reservoir fluid level, inspect the pump for leaks, and verify "
     "the pressure relief valve setting. Bleed the system after any component "
     "replacement per the standard practice."),

    ("dmc-39-A-25-60-00-00A-320A-A.xml", "25",
     "Emergency floatation system inspection. Inspect the float bags for "
     "damage, verify the inflation cylinder pressure is within limits, and "
     "test the electrical arming circuit continuity. Check the squib resistance "
     "before reconnecting."),

    ("dmc-39-A-26-21-00-00A-320A-A.xml", "26",
     "Engine fire detection BITE test. The fire detection control unit performs "
     "a built in test of the sensing elements. If the FIRE DET FAIL message "
     "appears, isolate the loop, measure the sensing element resistance, and "
     "inspect for chafing or short to ground along the routing."),

    ("dmc-39-A-34-22-00-00A-320A-A.xml", "34",
     "Radio altimeter intermittent reading. If the radio altimeter shows "
     "intermittent or erroneous height, inspect the antenna connectors, check "
     "the coaxial cable for damage, and verify the transceiver mounting. "
     "Perform the operation test after reconnection."),

    ("dmc-39-A-33-41-00-00A-320A-A.xml", "33",
     "Landing light inoperative troubleshooting. If the landing light does not "
     "illuminate, check the circuit breaker CB3, test the relay coil, verify "
     "the lamp filament continuity, and inspect the connector for corrosion. "
     "Replace the lamp assembly if the filament is open."),

    ("dmc-39-A-24-21-00-00A-320A-A.xml", "24",
     "Battery bus voltage low procedure. A low battery bus voltage may indicate "
     "a failed battery cell or charging fault. Measure the battery open circuit "
     "voltage, perform a capacity test, and inspect the charger output. Replace "
     "the battery if capacity is below the serviceable limit."),

    ("dmc-39-A-27-31-00-00A-320A-A.xml", "27",
     "Tail rotor servo control check. Inspect the tail rotor hydraulic servo "
     "for leaks, verify the input linkage rigging, and check the servo response "
     "during the flight control operation test. Confirm there is no binding "
     "through the full range of travel."),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", "-o", default="demo-corpus.json")
    ap.add_argument("--dim", "-d", type=int, default=DIM_DEFAULT,
                    help="Must match HashingEmbedder dimension in the app (default 384)")
    args = ap.parse_args()

    corpus = []
    for i, (path, ata, text) in enumerate(DOCS):
        corpus.append({
            "id": str(i),
            "doc_path": path,
            "ata": ata,
            "text": text,
            "embedding": embed(text, args.dim),
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=0)
    print(f"Wrote {len(corpus)} demo documents to {args.output} ({args.dim}-dim).")
    print("Transfer onto the iPad AS corpus.json, enable Demo mode, and run a query.")
    print('Try: ATA 24 + "generator voltage fluctuating", or "HEATER FAIL CAS message".')


if __name__ == "__main__":
    main()
