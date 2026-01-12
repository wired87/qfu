QUARK_FLAVORS = ["up", "down", "charm", "strange", "top", "bottom"]

G_FIELDS=[
    "photon",  # A_μ
    # Schwache Wechselwirkung (SU(2)_L)
    "w_plus",  # W⁺
    "w_minus",  # W⁻
    "z_boson",  # Z⁰
    # Starke Wechselwirkung (SU(3)_C)
    #"gluon",  # G 8x4 array
    *[f"gluon_{i}" for i in range(8)]

]


H = [
    "higgs"
]

QUARKS = [f"{flav}_quark_{i}" for flav in QUARK_FLAVORS for i in range(3)]
FERMIONS=[
    # Leptonen
    "electron",  # ψₑ
    "muon",  # ψ_μ
    "tau",  # ψ_τ
    "electron_neutrino",  # νₑ
    "muon_neutrino",  # ν_μ
    "tau_neutrino",  # ν_τ
    *QUARKS
]

"""
"up_quark_1",  # ψᵤ
"up_quark_2",  # ψᵤ
"up_quark_3",  # ψᵤ
"down_quark_1",  # ψ_d
"down_quark_2",  # ψ_d
"down_quark_3",  # ψ_d
"charm_quark_1",  # ψ_c
"charm_quark_2",  # ψ_c
"charm_quark_3",  # ψ_c
"strange_quark_1",  # ψ_s
"strange_quark_2",  # ψ_s
"strange_quark_3",  # ψ_s
"top_quark_1",  # ψ_t
"top_quark_2",  # ψ_t
"top_quark_3",  # ψ_t
"bottom_quark_1",  # ψ_b
"bottom_quark_2",  # ψ_b
"bottom_quark_3",  # ψ_b
"""

G_MAP=[
    "photon",  # A_μ
    # Schwache Wechselwirkung (SU(2)_L)
    "w_plus",  # W⁺
    "w_minus",  # W⁻
    "z_boson",  # Z⁰
    # Starke Wechselwirkung (SU(3)_C)
]

GLON_MAP = [f"gluon_{i}" for i in range(8)]

ALL_SUBS_LOWER=[*FERMIONS, *G_FIELDS, *H]
ALL_SUBS=[*[f.upper() for f in FERMIONS],*[g.upper() for g in G_FIELDS],*[h.upper() for h in H]]


