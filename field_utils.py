import os

import numpy as np
from data import ENV
from qf_utils.all_subs import GLON_MAP, FERMIONS, G_FIELDS, H, QUARKS
from utils.serialize_complex import serialize_complex_dict


class FieldUtils:

    def __init__(self):
        # set programmable field interaction
        self.set_interaction()
        self.shift_dirs = {
            "x": (1, 0, 0),
            "y": (0, 1, 0),
            "z": (0, 0, 1),
            "xy_pp": (1, 1, 0),
            "xy_pm": (1, -1, 0),
            "xz_pp": (1, 0, 1),
            "xz_pm": (1, 0, -1),
            "yz_pp": (0, 1, 1),
            "yz_pm": (0, 1, -1),
            "xyz_ppp": (1, 1, 1),
            "xyz_ppm": (1, 1, -1),
            "xyz_pmp": (1, -1, 1),
            "xyz_pmm": (1, -1, -1)
        }
        
        self.dim=int(os.getenv("DIM", 4))


        self.lambda_1 = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ], dtype=complex)
        self.lambda_2 = np.array([[0, -1j, 0],
                                  [1j, 0, 0],
                                  [0, 0, 0]], dtype=complex)

        self.lambda_3 = np.array([[1, 0, 0],
                                  [0, -1, 0],
                                  [0, 0, 0]], dtype=complex)

        self.lambda_4 = np.array([[0, 0, 1],
                                  [0, 0, 0],
                                  [1, 0, 0]], dtype=complex)

        self.lambda_5 = np.array([[0, 0, -1j],
                                  [0, 0, 0],
                                  [1j, 0, 0]], dtype=complex)

        self.lambda_6 = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ], dtype=complex)

        self.lambda_7 = np.array(
            [[0, 0, 0],
                  [0, 0, -1j],
                  [0, 1j, 0]], dtype=complex)

        self.lambda_8 = (1 / np.sqrt(3)) * np.array(
            [[1, 0, 0],
             [0, 1, 0],
            [0, 0, -2]
         ])

        self.o_operators = [
            self.lambda_1,
            self.lambda_2,
            self.lambda_3,
            self.lambda_4,
            self.lambda_5,
            self.lambda_6,
            self.lambda_7,
            self.lambda_8
        ]

        self.T1 = 0.5 * np.array([
            [0, 1],
            [1, 0]
        ], dtype=complex)

        self.T2 = 0.5 * np.array([
            [0, -1j],
            [1j, 0]
        ], dtype=complex)

        self.T3 = 0.5 * np.array([
            [1, 0],
            [0, -1]
        ], dtype=complex)

        self.T = [
            self.T1,
            self.T2,
            self.T3
        ]

        ### GAMMA ###
        self.gamma_0 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)

        self.gamma_1 = np.array([
            [0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, -1, 0, 0],
                            [-1, 0, 0, 0]], dtype=complex)

        self.gamma_2 = np.array([
            [0, 0, 0, -1j],
                            [0, 0, 1j, 0],
                            [0, 1j, 0, 0],
                            [-1j, 0, 0, 0]], dtype=complex)

        self.gamma_3 = np.array([[0, 0, 1, 0],
                            [0, 0, 0, -1],
                            [-1, 0, 0, 0],
                            [0, 1, 0, 0]
                                 ], dtype=complex)

        self.gamma = [self.gamma_0, self.gamma_1, self.gamma_2, self.gamma_3]
        self.gamma0_inv = np.linalg.inv(self.gamma[0])  # (4,4) matrix

        self.mt = np.diag([1, -1, -1, -1]) #minkowski_tensor

        self.vertex_combi_map = {
            "z_boson": [],
            "photon": [],
            "w_plus": [],
            "w_minus": [],
            "gluon": [],
        }


        self.parameters = {
            "o_operators": self.o_operators,
            "gamma": self.gamma,
            "T": self.T,
            "gamma0_inv": self.gamma0_inv,
            "minkovski_tensor": self.mt,
            "i": 1j
        }

        self.triple_vertex_schema = np.zeros((4, 4, 4), dtype=complex)
        self.quad_vertex_schema = np.zeros((4, 4, 4, 4), dtype=complex)

        self.ckm = {
            "up": {"down": 0.974, "strange": 0.225, "bottom": 0.004},
            "charm": {"down": 0.225, "strange": 0.973, "bottom": 0.041},
            "top": {"down": 0.009, "strange": 0.040, "bottom": 0.999},
        }

        self.fermion_fields = [
            # Leptonen
            "electron",  # ψₑ
            "muon",  # ψ_μ
            "tau",  # ψ_τ
            "electron_neutrino",  # νₑ
            "muon_neutrino",  # ν_μ
            "tau_neutrino",  # ν_τ
        
            # Quarks
            *QUARKS
        ]

        self.g_qcd=[
            "gluon",  # G 8x4 array
        ]

        self.g_qed = [
            # Elektromagnetismus
            "photon",  # A_μ
        ]

        self.g_electroweak = [
            # Schwache Wechselwirkung
            "w_plus",  # W⁺
            "w_minus",  # W⁻
            "z_boson",  # Z⁰
        ]

        self.gauge_fields = [
            *self.g_qcd,
            *self.g_electroweak,
            *self.g_qed
        ]

        self.gauge_to_gauge_couplings = {
            "gluon": ["gluon"],

            # Elektroschwach
            "w_plus": ["w_minus", "z_boson", "photon"],
            "w_minus": ["w_plus", "z_boson", "photon"],
            "z_boson": ["w_plus", "w_minus", "photon"],
            "photon": ["w_plus", "w_minus", "z_boson", "photon"],
        }

        self.g_h_couplings = [
            "w_plus",
            "w_minus",
            "z_boson",
        ]

        self.gauge_to_fermion_couplings = {
            "photon": [
                "electron", "muon", "tau",
                *QUARKS
            ],
            "w_plus": [
                "electron", "muon", "tau",
                "electron_neutrino", "muon_neutrino", "tau_neutrino",
                *QUARKS
            ],
            "w_minus": [
                "electron", "muon", "tau",
                "electron_neutrino", "muon_neutrino", "tau_neutrino",
                *QUARKS
            ],
            "z_boson": [
                "electron", "muon", "tau",
                "electron_neutrino", "muon_neutrino", "tau_neutrino",
                *QUARKS
            ],
            "gluon": QUARKS
        }

        self.all_sub_fields = [
            *[k.upper() for k in self.gauge_fields],
            *[k.upper() for k in self.fermion_fields],
            "PHI"
        ]

        # 4 array
        self.leptons = [
            "electron",
            "muon",
            "tau",
            "electron_neutrino",  # νₑ
            "muon_neutrino",  # ν_μ
            "tau_neutrino",  # ν_τ
        ]

        # Have color charge and interat with gluons -> psi is different fro leptons
        self.quarks = [
            "up_quark",
            "down_quark",
            "charm_quark",
            "strange_quark",
            "top_quark",
            "bottom_quark"
        ]

        self.fermions = {
            'field_name': ['electron', 'muon', 'tau', 'up_quark', 'down_quark', 'charm_quark', 'strange_quark', 'bottom_quark', 'top_quark', 'electron_neutrino', 'muon_neutrino', 'tau_neutrino'],
            'y': [2.94e-06, 0.00061, 0.0101, 1.32e-05, 2.8e-05, 0.0073, 0.00055, 0.024, 1.0, 2.87e-11, 5.77e-11, 1.44e-10],
            'energy': [0.000511, 0.1057, 1.777, 0.00255, 0.0048, 1.275, 0.095, 4.18, 172.76, 0.05, 0.05, 0.05],
            'isospin': [-0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5],
            'charge': [-1, -1, -1, 0.666666666666666, -0.3333333333, 0.666666666666666, -0.3333333333, -0.3333333333, 0.6666666666666, 0, 0, 0],
            'mass': [0.000511, 0.10566, 1.777, 0.00255, 0.0048, 1.27, 0.095, 4.7, 172, 0.05, 0.05, 0.05]
        }

        self.env = self.create_env()


    def create_env(
            self,
            world_cfg:dict=dict()
    ):
        """
        Create an ENVC layer (Environment Control Layer)
        which manages global fields/states like temperature, etc.
        """
        # todo customize settings -> read from received yaml -> incl tiemsteps
        # todo dev: apply envid
        env = dict(
            pm_axes=self.get_dirs(),
            **world_cfg,
            **ENV,
            **self.parameters
        )
        return env

    def get_dirs(self):
        # get_dirs
        plus = list(self.shift_dirs.values())

        minus = []

        for dir_str, struct in self.shift_dirs.items():
            n_struct = []

            for dir in struct:
                is_negative = dir < 0

                if is_negative:
                    n_struct.append(abs(dir))
                else:
                    n_struct.append(int(dir * -1))

            minus.append(tuple(n_struct))

        axes = [plus, minus]
        return axes


    def parent_ntype(self, ntype):
        parent = None
        ntype = ntype.lower()
        if ntype in FERMIONS:
            parent = "FERMION"
        elif ntype in G_FIELDS:
            parent = "GAUGE"
        elif ntype in H:
            parent = "HIGGS"
        return parent


    def get_pauli_matrice(
            self,
            field_name: str
    ):
        """
        Wählt die korrekte SU(2)-Generator-Matrix basierend auf dem Feldnamen aus.

        Args:
            field_name: Der Name des Eichbosons ('Wp', 'Wm', 'Z').

        Returns:
            Die entsprechende JAX-Matrix (z.B. T+, T-, T3).
        """
        T_PLUS = self.T1 + 1j * self.T2  # T+ = T1 + i*T2
        T_MINUS = self.T1 - 1j * self.T2  # T- = T1 - i*T2

        generator_map: dict = {
            "w_plus": T_PLUS,  # W+ Boson (Aufstiegsoperator)
            "w_minus": T_MINUS,  # W- Boson (Abstiegsoperator)
            "z_boson": self.T3,  # Z Boson (neutrale Komponente des Isospins)
        }

        return generator_map[field_name.lower()]

    def get_interactive_neighbors(self, ntype) -> list[str]:
        try:
            return self.interactants[ntype.lower()]
        except Exception as e:
            print(f"Err get_interactive_neighbors: {e}")
        return []




    def _field_value(self, type):
        type = type.lower()
        key=None
        if type in ["gluon", "gluon_item"]:
            key= "G"
        if type == "z_boson":
            key= "Z"
        if type == "w_plus":
            key= "Wp"
        if type == "w_minus":
            key= "Wm"
        if type == "photon":
            key= "A"
        return key

    def gamma5(self, i, gamma):
        return i * (gamma[0] @ gamma[1] @ gamma[2] @ gamma[3])

    def o_operator(
            self,
            ntype,
            o_operators,
            g_V=None,
            isospin=None,
            charge=None,
            gluon_index=None
    ):
        ntype = ntype.lower()

        if ntype in ["z_boson", "w_plus", "w_minus"]:
            if g_V is None or isospin is None:
                raise ValueError("Für Z müssen g_V, isospin und gamma5 gesetzt sein.")
            return g_V * np.eye(4, dtype=complex) - isospin * self.gamma5(self.i, self.gamma)

        elif ntype == "photon":
            return charge * np.eye(4, dtype=complex)

        elif ntype == "gluon":
            if gluon_index is None:
                raise ValueError("Für Gluon muss su3_generator gesetzt sein.")
            return o_operators[gluon_index]

        else:
            raise ValueError(f"Unbekannter Feld-Typ '{ntype}'")

    def _convert_to_complex(self, com):
        """
        Wandelt Listen oder serialisierte Dicts in ein komplexes NumPy-Array um.
        """
        # Wenn es ein Dict ist, deserialisieren
        if isinstance(com, dict) and "bytes" in com:
            com = np.array(com, complex)

        # Wenn es schon ein NumPy-Array ist, direkt zurückgeben
        if isinstance(com, np.ndarray):
            return com

        # Sonst weiterverarbeiten wie bisher
        complex_rows = []
        for row_data in com:
            complex_row = [complex(re, im) for re, im in row_data]
            complex_rows.append(complex_row)

        psi = np.array(complex_rows, dtype=complex)
        return psi

    def get_sym_group(self, ntype):
        all_active_groups = []
        for group, ntypes in self.symmetry_groups.items():
            if ntype.lower() in [n.lower() for n in ntypes]:
                all_active_groups.append(group)

       #print(f"Sym Groups for {ntype} set: {all_active_groups}")
        if not len(all_active_groups):
            raise ValueError(f"No SymGroup for {ntype}")
       #print("get_sym_group", all_active_groups)

        return all_active_groups

    def _get_gauge_generator(
            self,
            ntype,
            quark_index=None,
            gluon_index=None,
            Q=1.0,
            theta_w=28.7,
    ):
        """
        Returns a uniform 4×4 gauge generator matrix for:
            photon, Z, gluon, W+, W-
        After SSB:
            - W+, W- only couple to left-handed SU(2) doublets
            - So their generator is a 4×4 with a 2×2 SU(2) block at the top-left
            - The right-handed sector (lower 2×2) stays zero
        """
        ntype=ntype.lower()
        theta = np.deg2rad(theta_w)
        cosw = np.cos(theta)
        sinw = np.sin(theta)

        # ----- Helper: Embed a 2×2 matrix into a 4×4 -----
        def embed_2x2(M2):
            M4 = np.zeros((4, 4), dtype=complex)
            M4[:2, :2] = M2
            return M4

        # ----- Photon -----
        def calculate_photon(_):
            return Q * np.zeros((4, 4), dtype=complex)

        # ----- Z boson: T = cosθ * T3 - sinθ * Y -----
        def calculate_z(_):
            T3 = embed_2x2(self.T3)
            Y = 0.5 * np.zeros((4, 4), dtype=complex)
            return cosw * T3 - sinw * Y

        # ----- Gluon: SU(3) generator -----
        def calculate_gluon(_):
            o = self.o_operators[gluon_index]
            o = o[quark_index]
            return o

        # ----- W+ -----
        # SU(2) raising operator: T⁺ = (T1 + i T2)
        def calculate_w_plus(_):
            Tplus = self.T1 + 1j * self.T2  # 2×2
            return embed_2x2(Tplus)

        # ----- W− -----
        # SU(2) lowering operator: T⁻ = (T1 - i T2)
        def calculate_w_minus(_):
            Tminus = self.T1 - 1j * self.T2  # 2×2
            return embed_2x2(Tminus)

        try:
            # ---- Dispatch table ----
            lookup = {
                "photon": calculate_photon,
                "z_boson": calculate_z,
                "gluon": calculate_gluon,
                "w_plus": calculate_w_plus,
                "w_minus": calculate_w_minus,
            }
            #print("ntype.lower()", ntype.lower())
            a = lookup[ntype.lower()](None)
            return a
        except Exception as e:
            print("Err _get_gauge_generator", e)


    def restore_selfdict(self, data):
        new_dict = {}
        #print("restore_selfdict admin_data B4:", admin_data)
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and "serialized_complex" in v:  # ("bytes" in v or "admin_data" in v)
                    v = None
                #v = convert_numeric(v)
                new_dict[k] = v
        return new_dict


    def g_V(self, isospin, charge, sin2W=0.231):
        """print("get_g_V")
        #print("charge", charge)
        #print("isospin", isospin)"""
        g_V = float(isospin) - 2 * float(charge) * sin2W
        return g_V

    def dmu(self, dim:int, shape=True):
        """if shape:
            return (
                len(list(self.shift_dirs.values())),
                dim,
            )"""

        return [
            self.field_value(dim=dim)
            for _ in range(len(list(self.shift_dirs.values())))
        ]

    def fmunu(self, dim, shape=True):

        """
        field_value = np.array([
            [1.0 + 0.0j],  # Zeitkomponente A_0
            [0.0 + 0.0j],  # A_1
            [0.0 + 0.0j],  # A_2
            [0.0 + 0.0j]  # A_3
        ], dtype=complex)
        """
        # dim X dim
        """
        if shape:
            return (
                dim,
                dim
            )
        """
        return [
            [0 for _ in range(dim)]
            for a in range(dim)
        ]


    def field_value(self, pos=None, dim=None) -> list or tuple:
        # 1, dim
        """

        """
        return [0 for _ in range(dim)]

    def gluon_fieldv(self, pos):
        return [
            self.field_value(pos)
            for _ in range(8)
        ]

    def quark_field(self, pos):
        return [
            self.field_value(pos)
            for _ in range(3)
        ]

    def dmu_fmunu(self, dim, shape=True):
        # self.shift_dirs, dim, dim
        """
        if shape is True:
            return (
                len(list(self.shift_dirs.values())),
                dim,
                dim
            )
        """
        return [
            [
                [0 for b in range(dim)]      # (3) Spalten (dim)
                for a in range(dim)          # (2) Zeilen (dim)
            ]
            for _ in range(len(self.shift_dirs)) # (1) Operatoren (num_ops)
        ]

    def get_gauge_field_symbol(self, ntype):
        ntype = ntype.lower()
        GAUGE_FIELD_SYMBOLS = {
            "photon": "γ",  # Photon
            "gluon": "g",  # Gluon
            "w_plus": "W⁺",  # W+
            "w_minus": "W⁻",  # W-
            "z_boson": "Z⁰",  # Z
            "phi": "H",  # Higgs
            "b_field": "B",  # Hypercharge
            "w_field": "W",  # Weak isospin
        }
        return GAUGE_FIELD_SYMBOLS.get(ntype)


    def _tripple_vertex_type_combi(self, ntype):
        if ntype.lower() == "z_boson":
            return ["w_plus", "w_minus", "z_boson"]
        elif ntype.lower() == "photon":
            return ["w_plus", "w_minus", "photon"]
        elif ntype.lower() == "w_plus" or ntype.lower() in "w_minus":
            return [["w_plus", "w_minus", "photon"], ["w_plus", "w_minus", "z_boson"]]
        elif ntype.lower() == "gluon":
            return ["gluon" for _ in range(3)]



    def set_interaction(self):
        ggc = {k: [*QUARKS, *[g for g in GLON_MAP if g != k]] for k in GLON_MAP}

        # qmap: each quark color has photon, w+, w-, z and all gluons
        qmap = {
            q: ["photon", "w_plus", "w_minus", "z_boson", *GLON_MAP]
            for q in QUARKS
        }

        neutrino_couplings = {
            k: ["w_plus", "w_minus", "z_boson"]
            for k in ["electron_neutrino", "muon_neutrino", "tau_neutrino"]
        }

        general_ferm_couplings = {
            k: ["photon", "w_plus", "w_minus", "z_boson"]
            for k in ["electron", "muon", "tau"]
        }

        fermion_g_couplings = {
            **general_ferm_couplings,
            **neutrino_couplings,
            **qmap,
        }

        gauge_couplings = {
            "w_plus": ["w_minus", "z_boson", "photon"],
            "w_minus": ["w_plus", "z_boson", "photon"],
            "z_boson": ["w_plus", "w_minus", "photon"],
            "photon": ["w_plus", "w_minus", "z_boson", "photon"],
            **ggc,
        }

        self.modules_fields = {
            "FERMION": FERMIONS,
            "GAUGE": G_FIELDS,
            "HIGGS": H,
        }

        self.couplings = {
            "FERMION": {
                "HIGGS": {ferm: ["phi"] for ferm in FERMIONS},
                "GAUGE": fermion_g_couplings,
            },
            "GAUGE": {
                "HIGGS": {g: ["phi"] for g in ["w_plus", "w_minus", "z_boson"]},
                "FERMION": fermion_g_couplings,
                "GAUGE": gauge_couplings,
            },
            "HIGGS": {
                "FERMION": {"phi": [
                    *QUARKS,
                    "electron", "muon", "tau", "tau_neutrino", "muon_neutrino", "electron_neutrino",
                ]},
                "GAUGE": {"phi": ["w_plus", "w_minus", "z_boson"]},
            }
        }
