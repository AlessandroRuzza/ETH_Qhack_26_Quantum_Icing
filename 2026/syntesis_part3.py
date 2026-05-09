import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit.synthesis import gridsynth_rz

from gate_syntesis import (
    Rz,
    gate_distance,
    gate_sequence_from_circuit,
    t_count_from_sequence,
    unitary_from_gate_sequence,
)


DISPLAY_COLUMNS = [
    "n",
    "target",
    "logical_gate_count",
    "logical_T_count",
    "logical_depth_part2",
    "injected_T_count",
    "two_qubit_gate_count",
    "measurements",
    "feedforward_decisions",
    "estimated_depth_with_injection",
    "ancillas_no_reuse",
    "ancillas_with_reuse",
    "postselection_success_probability",
    "expected_trials_without_feedforward",
    "distance",
]


def estimated_injected_depth(sequence):
    depth = 0

    for gate_name in sequence:
        if gate_name in {"h", "s", "x"}:
            depth += 1
        elif gate_name == "sdg":
            depth += 3
        elif gate_name == "t":
            # H/T ancilla prep, CNOT, measurement, and worst-case feed-forward S.
            depth += 5
        elif gate_name == "tdg":
            # T-injection plus three S gates implementing S^\dagger.
            depth += 8

    return depth


def ensure_part2_results(
    n_values=range(51),
    part2_epsilon=1e-10,
    part2_circuits=None,
    part2_sequences=None,
):
    part2_circuits = dict(part2_circuits or {})
    part2_sequences = dict(part2_sequences or {})

    for n in n_values:
        if n in part2_circuits and n in part2_sequences:
            continue

        theta = np.pi / (2 ** n)
        circuit = gridsynth_rz(theta, epsilon=part2_epsilon)
        part2_circuits[n] = circuit
        part2_sequences[n] = gate_sequence_from_circuit(circuit)

    return part2_circuits, part2_sequences


def build_part3_dataframe(part2_sequences, n_values=range(51), part2_circuits=None):
    rows = []

    for n in n_values:
        sequence = part2_sequences[n]
        theta = np.pi / (2 ** n)

        logical_t_count = t_count_from_sequence(sequence)
        distance = gate_distance(Rz(theta), unitary_from_gate_sequence(sequence))
        success_probability = 2 ** (-logical_t_count)
        expected_trials = 2 ** logical_t_count
        logical_depth = (
            part2_circuits[n].depth()
            if part2_circuits is not None and n in part2_circuits
            else len(sequence)
        )

        rows.append(
            {
                "n": n,
                "target": f"Rz(pi/2^{n})",
                "logical_sequence": " ".join(sequence).upper(),
                "logical_gate_count": len(sequence),
                "logical_T_count": logical_t_count,
                "logical_depth_part2": logical_depth,
                "direct_T_on_main": 0,
                "injected_T_count": logical_t_count,
                "CNOT_from_injection": logical_t_count,
                "two_qubit_gate_count": logical_t_count,
                "measurements": logical_t_count,
                "max_feedforward_S": logical_t_count,
                "feedforward_decisions": logical_t_count,
                "estimated_depth_with_injection": estimated_injected_depth(sequence),
                "ancillas_no_reuse": logical_t_count,
                "ancillas_with_reuse": 1 if logical_t_count > 0 else 0,
                "postselection_success_probability": success_probability,
                "expected_trials_without_feedforward": expected_trials,
                "log10_expected_trials_without_feedforward": (
                    logical_t_count * np.log10(2)
                ),
                "distance": distance,
            }
        )

    return pd.DataFrame(rows)


def print_part3_summary(df_part3, display_fn=None):
    if display_fn is None:
        print(df_part3[DISPLAY_COLUMNS])
    else:
        display_fn(df_part3[DISPLAY_COLUMNS])

    print("\nDetailed logical sequences:\n")
    for _, row in df_part3.iterrows():
        print(f"n = {row['n']} | {row['target']}")
        print(f"sequence = {row['logical_sequence']}")
        print("-" * 80)


def save_part3_plots(df_part3, results_dir="results", show=True):
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(df_part3["n"], df_part3["distance"], marker="o")
    plt.xlabel("n")
    plt.ylabel("Distance d(U,V)")
    plt.title("Part 3: logical approximation distance")
    plt.xticks(df_part3["n"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "part3_logical_approximation_distance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(
        df_part3["n"],
        df_part3["injected_T_count"],
        marker="o",
        label="Injected T gadgets",
    )
    plt.plot(
        df_part3["n"],
        df_part3["two_qubit_gate_count"],
        marker="s",
        label="CNOT gates",
    )
    plt.plot(
        df_part3["n"],
        df_part3["measurements"],
        marker="^",
        label="Measurements",
    )
    plt.plot(
        df_part3["n"],
        df_part3["feedforward_decisions"],
        marker="x",
        label="Feed-forward decisions",
    )
    plt.xlabel("n")
    plt.ylabel("Count")
    plt.title("Part 3: overhead from replacing T by T-injection")
    plt.xticks(df_part3["n"])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "part3_t_injection_overhead.png"),
        dpi=300,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(
        df_part3["n"],
        df_part3["logical_depth_part2"],
        marker="o",
        label="Part 2 logical depth",
    )
    plt.plot(
        df_part3["n"],
        df_part3["estimated_depth_with_injection"],
        marker="s",
        label="Part 3 estimated injected depth",
    )
    plt.xlabel("n")
    plt.ylabel("Depth")
    plt.title("Part 3: estimated circuit depth overhead")
    plt.xticks(df_part3["n"])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "part3_estimated_depth_overhead.png"),
        dpi=300,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(7, 4))
    plt.semilogy(
        df_part3["n"],
        df_part3["expected_trials_without_feedforward"].astype(float),
        marker="o",
    )
    plt.xlabel("n")
    plt.ylabel("Expected trials")
    plt.title("Part 3: expected repeated trials without feed-forward")
    plt.xticks(df_part3["n"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "part3_expected_trials_without_feedforward.png"),
        dpi=300,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(7, 4))
    plt.semilogy(
        df_part3["n"],
        df_part3["postselection_success_probability"],
        marker="o",
    )
    plt.xlabel("n")
    plt.ylabel("Success probability")
    plt.title("Part 3: postselection success probability without feed-forward")
    plt.xticks(df_part3["n"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "part3_postselection_success_probability.png"),
        dpi=300,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    else:
        plt.close()


def run_part3_analysis(
    part2_circuits=None,
    part2_sequences=None,
    n_values=range(51),
    part2_epsilon=1e-10,
    results_dir="results",
    display_fn=None,
    show_plots=True,
):
    part2_circuits, part2_sequences = ensure_part2_results(
        n_values=n_values,
        part2_epsilon=part2_epsilon,
        part2_circuits=part2_circuits,
        part2_sequences=part2_sequences,
    )
    df_part3 = build_part3_dataframe(
        part2_sequences,
        n_values=n_values,
        part2_circuits=part2_circuits,
    )

    print_part3_summary(df_part3, display_fn=display_fn)
    save_part3_plots(df_part3, results_dir=results_dir, show=show_plots)

    return {
        "df_part3": df_part3,
        "part2_circuits": part2_circuits,
        "part2_sequences": part2_sequences,
    }
