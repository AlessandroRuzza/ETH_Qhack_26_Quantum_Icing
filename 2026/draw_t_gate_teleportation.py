import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(-1.5, 13.5)
ax.set_ylim(0, 6)
ax.set_aspect('equal')
ax.axis('off')

ORANGE_EDGE = '#D4783A'
ORANGE_FILL = '#F9DFC0'
BLUE       = '#2A4FB0'
GRAY       = '#666666'
GW, GH     = 0.70, 0.70   # gate box size

data_y = 4.2
anc_y  = 1.8

# ── Wire labels ──────────────────────────────────────────────────────────────
ax.text(-1.4, data_y, r'data $|\psi\rangle$',   ha='left', va='center', fontsize=12)
ax.text(-1.4, anc_y,  r'ancilla $|0\rangle$',   ha='left', va='center', fontsize=12)

# ── Wires ────────────────────────────────────────────────────────────────────
wire_start = 1.0
ax.plot([wire_start, 13.0], [data_y, data_y], 'k-', lw=1.6, zorder=1)
ax.plot([wire_start, 11.5], [anc_y,  anc_y],  'k-', lw=1.6, zorder=1)

# ── Output label ─────────────────────────────────────────────────────────────
ax.text(13.1, data_y, r'$T|\psi\rangle$', ha='left', va='center', fontsize=12)


def gate_box(cx, cy, label, facecolor=ORANGE_FILL):
    rect = patches.FancyBboxPatch(
        (cx - GW/2, cy - GH/2), GW, GH,
        boxstyle="round,pad=0.07",
        linewidth=1.8, edgecolor=ORANGE_EDGE, facecolor=facecolor, zorder=4
    )
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=13, fontweight='bold', zorder=5)


# ── H gate ───────────────────────────────────────────────────────────────────
Hx = 2.5
gate_box(Hx, anc_y, 'H')

# ── T gate ───────────────────────────────────────────────────────────────────
Tx = 3.9
gate_box(Tx, anc_y, 'T')

ax.text((Hx + Tx) / 2, anc_y - 0.80,
        r'$|A\rangle = T|+\rangle$',
        ha='center', va='top', fontsize=9.5, color=GRAY)

# ── CNOT ─────────────────────────────────────────────────────────────────────
Cx = 6.4

# label above
ax.text(Cx, data_y + 0.58, 'CNOT', ha='center', va='bottom', fontsize=11)

# vertical wire (control → target)
ax.plot([Cx, Cx], [anc_y, data_y], color=BLUE, lw=2.0, zorder=3)

# control dot on ancilla
ax.plot(Cx, anc_y, 'o', color=BLUE, markersize=10, zorder=5)

# target ⊕ on data
R = 0.28
circ = plt.Circle((Cx, data_y), R, color='white', ec=BLUE, lw=2.0, zorder=4)
ax.add_patch(circ)
ax.plot([Cx - R, Cx + R], [data_y, data_y], color=BLUE, lw=2.0, zorder=5)
ax.plot([Cx, Cx], [data_y - R, data_y + R], color=BLUE, lw=2.0, zorder=5)

# ── Measurement ──────────────────────────────────────────────────────────────
Mx = 8.5
mw, mh = 0.78, 0.68
meas_rect = patches.FancyBboxPatch(
    (Mx - mw/2, anc_y - mh/2), mw, mh,
    boxstyle="round,pad=0.07",
    linewidth=1.8, edgecolor=ORANGE_EDGE, facecolor='white', zorder=4
)
ax.add_patch(meas_rect)

# arc
theta = np.linspace(np.pi, 0, 80)
arc_r = 0.21
arc_cy = anc_y - 0.05
ax.plot(Mx + arc_r * np.cos(theta),
        arc_cy + arc_r * 0.65 * np.sin(theta),
        'k-', lw=1.4, zorder=5)
# arrow from center to upper-right
ax.annotate('', xy=(Mx + 0.17, anc_y + 0.18),
            xytext=(Mx, arc_cy),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.4),
            zorder=6)

ax.text(Mx, anc_y - 0.80, r'measure $M_z$',
        ha='center', va='top', fontsize=9.5, color=GRAY)

# ── Classical double wire ─────────────────────────────────────────────────────
Sx   = 10.8
gap  = 0.085
cl_y = anc_y         # start height (middle of measurement box)

# horizontal: from right edge of measurement box to x of S gate
for dy in (-gap, gap):
    ax.plot([Mx + mw/2, Sx], [cl_y + dy, cl_y + dy],
            color=GRAY, lw=1.5, zorder=2)

# vertical: from ancilla wire height up to bottom of S gate
for dx in (-gap, gap):
    ax.plot([Sx + dx, Sx + dx], [cl_y - gap, data_y - GH/2],
            color=GRAY, lw=1.5, zorder=2)

# ── S gate ───────────────────────────────────────────────────────────────────
gate_box(Sx, data_y, 'S')
ax.text(Sx, data_y + GH/2 + 0.15, 'if $m = 1$',
        ha='center', va='bottom', fontsize=9.5, color=GRAY)

# ─────────────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.1)
plt.savefig('results/t_gate_teleportation_circuit.png',
            dpi=180, bbox_inches='tight', facecolor='white')
plt.show()
print("Salvato in results/t_gate_teleportation_circuit.png")
