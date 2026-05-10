from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from qiskit.synthesis import gridsynth_rz


BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=str(BASE_DIR / "static"), static_url_path="/static")


ALLOWED_GATES = {"h", "s", "sdg", "t", "tdg", "x", "id"}


def gate_sequence_from_circuit(circuit):
    """Extract a Clifford+T gate sequence from a one-qubit Qiskit circuit."""
    sequence = []

    for instruction in circuit.data:
        name = instruction.operation.name.lower()

        if name in {"barrier", "delay", "id"}:
            continue

        if len(instruction.qubits) != 1:
            raise ValueError(f"Expected a one-qubit gate, found {name!r}.")

        if name == "z":
            # Z is Clifford and equals S*S.
            sequence.extend(["s", "s"])
            continue

        if name not in ALLOWED_GATES:
            raise ValueError(f"Unsupported gate returned by gridsynth: {name!r}.")

        sequence.append(name)

    return sequence


def rz_gridsynth_sequence(theta_rad, epsilon):
    """Use Qiskit gridsynth_rz to synthesize Rz(theta) into Clifford+T."""
    circuit = gridsynth_rz(theta_rad, epsilon=epsilon)
    return gate_sequence_from_circuit(circuit)


def tagged(gates, axis, source):
    """Attach metadata used by the frontend animation."""
    return [
        {
            "gate": gate,
            "axis": axis,
            "source": source,
        }
        for gate in gates
    ]


def synthesize_axis(axis, theta_rad, epsilon):
    """Synthesize Rx, Ry, or Rz using only Clifford wrappers plus gridsynth Rz."""
    axis = axis.upper()

    if abs(theta_rad) < 1e-15:
        return []

    if axis == "Z":
        rz_seq = rz_gridsynth_sequence(theta_rad, epsilon)
        return tagged(rz_seq, "Rz", "gridsynth_rz")

    if axis == "X":
        # Rx(theta) = H Rz(theta) H.
        rz_seq = rz_gridsynth_sequence(theta_rad, epsilon)
        return (
            tagged(["h"], "Rx", "Clifford wrapper")
            + tagged(rz_seq, "Rx", "gridsynth_rz")
            + tagged(["h"], "Rx", "Clifford wrapper")
        )

    if axis == "Y":
        # Ry(theta) = S H Rz(theta) H Sdg as a total unitary.
        # Circuit order is therefore Sdg, H, Rz, H, S.
        rz_seq = rz_gridsynth_sequence(theta_rad, epsilon)
        return (
            tagged(["sdg", "h"], "Ry", "Clifford wrapper")
            + tagged(rz_seq, "Ry", "gridsynth_rz")
            + tagged(["h", "s"], "Ry", "Clifford wrapper")
        )

    raise ValueError(f"Unknown axis: {axis!r}")


def build_full_sequence(angles_deg, order, epsilon):
    """Build the full Clifford+T sequence for the requested Euler-style rotations."""
    angle_map = {
        "X": np.deg2rad(float(angles_deg.get("x", 0.0))),
        "Y": np.deg2rad(float(angles_deg.get("y", 0.0))),
        "Z": np.deg2rad(float(angles_deg.get("z", 0.0))),
    }

    order = order.upper()
    if sorted(order) != ["X", "Y", "Z"]:
        raise ValueError("Order must be a permutation of XYZ, for example XYZ or ZYX.")

    gates = []
    for axis in order:
        gates.extend(synthesize_axis(axis, angle_map[axis], epsilon))

    return gates


def run_self_tests():
    """Small backend sanity checks."""
    seq_45 = rz_gridsynth_sequence(np.pi / 4, 1e-10)
    assert all(gate in {"h", "s", "sdg", "t", "tdg", "x"} for gate in seq_45)

    seq_x = synthesize_axis("X", np.pi / 4, 1e-10)
    assert seq_x[0]["gate"] == "h"
    assert seq_x[-1]["gate"] == "h"

    seq_y = synthesize_axis("Y", np.pi / 4, 1e-10)
    assert seq_y[0]["gate"] == "sdg"
    assert seq_y[1]["gate"] == "h"
    assert seq_y[-2]["gate"] == "h"
    assert seq_y[-1]["gate"] == "s"

    full = build_full_sequence({"x": 22.5, "y": 37, "z": -60}, "XYZ", 1e-10)
    assert all(item["gate"] in {"h", "s", "sdg", "t", "tdg", "x"} for item in full)

    print("Backend self-tests passed.")


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "demo.html")


@app.route("/demo.html")
def demo():
    return send_from_directory(BASE_DIR, "demo.html")


@app.route("/api/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json(force=True)

    angles_deg = data.get("angles_deg", {})
    order = data.get("order", "XYZ")
    epsilon = float(data.get("epsilon", 1e-10))

    gates = build_full_sequence(angles_deg, order, epsilon)

    flat_sequence = [item["gate"] for item in gates]
    t_count = sum(gate in {"t", "tdg"} for gate in flat_sequence)

    return jsonify(
        {
            "gates": gates,
            "flat_sequence": flat_sequence,
            "meta": {
                "mode": "qiskit.synthesis.gridsynth_rz",
                "order": order,
                "epsilon": epsilon,
                "sequence_length": len(flat_sequence),
                "t_count": t_count,
                "clifford_count": len(flat_sequence) - t_count,
            },
        }
    )


if __name__ == "__main__":
    print("Server ready.")
    print("Open: http://172.27.208.87:8018")
    app.run(host="0.0.0.0", port=8018, debug=True, use_reloader=False)