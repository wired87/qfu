import pprint
from typing import Any

import numpy as np

from firegraph.qfu.data import FERM_PARAMS, GAUGE_FIELDS
from firegraph.qfu.fermion.ferm_creator import FermCreator
from firegraph.qfu.gauge.g_creator import GaugeCreator
from firegraph.qfu.higgs.higgs_creator import HiggsCreator
from firegraph.qfu.all_subs import FERMIONS, G_FIELDS, H, ALL_SUBS
from firegraph.qfu.field_utils import FieldUtils

from firegraph.file._csv import dict_2_csv
from firegraph.gutils import GUtils
from firegraph.utils.serialize_complex import is_complex
from firegraph.utils.get_shape import get_shape


class QFUtils(FieldUtils):

    def __init__(
            self,
            g=None,
            G=None,
            testing=False
    ):
        super().__init__()
        if G is not None and g is None:
            self.g = GUtils(G=G)
        else:
            self.g=g

        self.field_utils = FieldUtils()
        # self.user_id = user_id
        self.metadata_path = "metadata"
        self.testing = testing

        # Creator classes
        self.g_creator = GaugeCreator(
            g_utils=self.g,
        )

        self.ferm_creator = FermCreator(
            g=self.g,
        )

        self.higgs_creator = HiggsCreator(
            g=self.g,
        )


    def get_modules_methods(self, mid):
        print("get_modules_methods...")
        methods = {}
        method_nodes = self.g.get_neighbor_list_rel(
            trgt_rel="has_method",
            node=mid,
            as_dict=True,
        )
        methods.update(method_nodes)

        params = self.g.get_neighbor_list_rel(
            trgt_rel="has_param",
            node=mid,
            as_dict=True,
        )

        for pid, pattrs in params.items():
            if "code" in pattrs:
                # method
                methods[pid] = pattrs

        pprint.pp(methods)
        print("get_modules_methods... done", )
        return methods


    def filter_ntype(self, ntype):
        """
        REMOVE INDEXING FROM NTYPE
        """
        last_part = ntype.split("_")[-1]
        #print("last_part", last_part)
        if last_part.isdigit():
            ntype = ntype.replace(f"_{last_part}", "")
        return ntype


    def get_parent(self, ntype):
        ntype = ntype.lower()
        parent = None
        if ntype in FERMIONS:
            parent = "FERMION"
        elif ntype in G_FIELDS:
            parent = "GAUGE"
        elif ntype in H:
            parent = "HIGGS"
        return parent


    def get_nids_from_pxid(self, npxs, ntypes:list[str]):
        nnids = []
        for vn in ntypes:
            vn = vn.upper()
            for px_neighbor in npxs:
                g_neighbors = self.get_neighbor_ids(vn, px_neighbor)
                nnids.extend(g_neighbors)
        #print("nnids", nnids)
        return nnids


    def i(self):
        return self.i


    def all_px_neighbors(self, attrs):
        nid_part = attrs["id"].split("_px_")[-1].split("_")[0]

        neighbors = [
            pxid
            for pxid in (*attrs["npm"][0], *attrs["npm"][1])
            if pxid != f"px_{nid_part}"
        ]

        #print("neighbors", neighbors)
        return neighbors


    def get_neighbor_ids(self, ntype, px_id):
        nid_map = []
        if "quark" in ntype.lower():
            for item_index in range(3):
                nid_map.append(
                    f"{ntype}__px_{px_id}__{item_index}"
                )
        elif "gluon" in ntype.lower():
            if "gluon" == ntype.lower():
                for item_index in range(8):
                    nid_map.append(
                        f"{ntype}__px_{px_id}__{item_index}"
                    )
        else:
            nid_map.append(
                f"{ntype}__px_{px_id}"
            )
        return nid_map


    def build_parameter(self):
        print("build_parameter...")

        for key, value in self.create_env().items():
            _complex = is_complex(value)
            self.g.add_node(
                dict(
                    id=key,
                    type="PARAM",
                    value=value,
                    const=True,
                    axis_def=None,
                    param_type="complex" if _complex is True else type(value)
                )
            )

        # add return key as param jsut use within sm managrer workflow
        for key, value in self.g.G.nodes(data=True):
            if value.get("type") == "METHOD":
                self.g.add_node(
                    dict(
                        id=key,
                        type="PARAM",
                        value=[],
                        const=False,
                        axis_def=None,
                        param_type="Any" #todo
                    )
                )
        print("build_parameter... done")


    def build_interacion_G(self):
        """
        Create the fundament for a interaction G in
        the sm
        """

        # ADD MODULEE
        for i, (src_layer, interactant_struct) in enumerate(self.couplings.items()):
            if not self.g.G.has_node(src_layer):
                self.g.add_node(
                    dict(
                        id=src_layer.upper(),
                        type="MODULE",
                        module_index=i
                    )
                )

        # CREATE FIELDS
        for ntype in ALL_SUBS:
            ntype=ntype.upper()
            module_id = self.get_parent(ntype)
            field_index = self.modules_fields[module_id].index(ntype.lower())
            # create batch attrs
            field_value = self.batch_field_single(
                ntype,
                dim=3,
            )

            # set axis def
            axis_def = list(self.set_axis(list(field_value.keys())))

            interactant_fields = [f.upper() for f in self.set_interactions_fild(ntype)]

            self.g.add_node(
                dict(
                    id=ntype,
                    type="FIELD",
                    field_index=field_index,
                    parent=[
                        module_id,
                    ],
                    values=list(field_value.values()),
                    keys=list(field_value.keys()),
                    shape=[get_shape(value) for value in field_value.values()],
                    axis_def=axis_def,
                    interactant_fields=interactant_fields,
                    module_id=self.get_parent(ntype),
                )
            )

        for src_layer, interactant_struct in self.couplings.items():
            for i, (trgt_layer, trgt_struct) in enumerate(interactant_struct.items()):
                #print("trgt_layer, trgt_struct", trgt_layer, trgt_struct)
                for ntype, interactants in trgt_struct.items():
                    ntype = ntype.upper()
                    self.add_edges_from_map(
                        ntype=ntype,
                        src_type=src_layer,
                        interactants=interactants,
                    )
        print("build_interacion_G... done")


    def set_interactions_fild(self, field_id):
        #print("set_interactions_fild")
        all_interactants = []
        for src_layer, interactant_struct in self.couplings.items():
            for i, (trgt_layer, trgt_struct) in enumerate(interactant_struct.items()):
                #print("trgt_layer, trgt_struct", trgt_layer, trgt_struct)
                for ntype, interactants in trgt_struct.items():
                    if field_id.upper() == ntype.upper():
                        all_interactants.extend(interactants)
        #print("set_interactions_fild... done")
        return all_interactants


    def set_axis(self, data_keys:list[Any]):
        """
        Determines the vmap axis for each parameter in the admin_data bundle.
        - Use axis 0 for array-like admin_data (map over it).
        - Use None for scalar admin_data (broadcast it).
        """
        return (
            0 if not p in self.env else None
            for p in data_keys
        )

    # Helper to add edges safely and set metadata
    def add_edges_from_map(self, ntype, src_type, interactants):
        # mapping can be dict-of-lists or dict-of-dicts
        for neighbor in interactants:
            # field -> field
            self.g.add_edge(
                src=ntype.upper(),
                trgt=neighbor.upper(),
                attrs=dict(
                    rel="has_interactant",
                    src_layer="FIELD",
                    trgt_layer="FIELD",
                )
            )

        # A module owns only its declared fields; other fields are linked as interactants.
        module_fields = self.modules_fields.get(src_type.upper(), [])
        if ntype.lower() in module_fields:
            self.g.add_edge(
                src=src_type.upper(),
                trgt=ntype,
                attrs=dict(
                    rel="has_field",
                    src_layer="MODULE",
                    trgt_layer="FIELD",
                )
            )


    def get_npm_values(
            self,
            npm:list[list[int]],
            ntype,
            field_key=None,
    ) -> dict or list[list, list]:
        """
        Extract all values from neighbo_pm-struct
        """
        npm_val_struct = [[],[]]
        try:
            if field_key is None:
                field_key = self._field_value(ntype)

            """
            
            Make request values for all pms
            
            """

            # Assume field_key is defined and self.g is initialized
            for i, (p, m) in enumerate(zip(npm[0], npm[1])):

                npm_val_struct[0].append(self.get_pm_val(
                    ntype, p, field_key
                ))

                npm_val_struct[1].append(self.get_pm_val(
                    ntype, m, field_key
                ))

        except Exception as e:
            print("Err get_npm_values:", e)
        #print("npm_val_struct", npm_val_struct)
        return npm_val_struct


    def get_pm_val(self, ntype, pxid, field_key):
        for nnid in self.get_neighbor_ids(ntype, pxid):
            # print("nnid", nnid)
            node_data = self.g.get_node(id=nnid)
            # print("node_data", node_data)

            if node_data:
                field_value = node_data.get(field_key)
                # print("field_value", field_value)
                value = None#deserialize(field_value)
                return value
            else:
                print("NO NODE FOUND WITH ID", nnid)





    def npm(
            self,
            node_id:str,
            self_attrs:dict,
            all_pixel_nodes,
    ):
        try:

            """
            :return: pixel neighbors foreach pos dir (p + m)
            (x before and x after)
            Deine Beobachtung in der Simulation hat eine sehr reale Entsprechung in der Physik, nämlich die gravitative Zeitdilatation, ein Phänomen der Allgemeinen Relativitätstheorie von Albert Einstein.

            Es ist tatsächlich so: An einem Ort, an dem "mehr los ist" im
            Sinne von mehr Masse oder Energie, vergeht die Zeit langsamer.
            Das liegt daran, dass Masse und Energie die Raumzeit krümmen.
            Je stärker diese Krümmung ist (also je mehr "Payload" oder
            Masse an einem Punkt vorhanden ist), desto langsamer ticken die
            Uhren in diesem Bereich.
            """
            # print("Set neighbors pm")


            npm_struct = [[], []]

            self_pos = np.array(self_attrs["pos"])

            node_pos_dict = {node: np.array(attrs.get("pos")) for node, attrs in all_pixel_nodes}

            for direction_name, direction_matrix in self.direction_definitions.items():
                offset = np.array(direction_matrix) * self.env["d"]
                pos_plus = self_pos + offset
                pos_minus = self_pos - offset

                node_plus = next((k for k, v in node_pos_dict.items() if np.allclose(v, pos_plus)), None)
                node_minus = next((k for k, v in node_pos_dict.items() if np.allclose(v, pos_minus)), None)

                # Fallback auf self_node, falls Ziel nicht gefunden
                node_plus = node_plus if node_plus else node_id
                node_minus = node_minus if node_minus else node_id
                npm_struct[0].append(node_plus.split("_")[-1])
                npm_struct[1].append(node_minus.split("_")[-1])
            # print("npm_struct", npm_struct)
            return npm_struct
        except Exception as e:
            print(f"Error in npm: {e}")


    def classify_px_neighbors(self):
        struct = {}
        for nid, attrs in [
            (nid, attrs)
            for nid, attrs in self.g.G.nodes(data=True)
            if attrs.get("type").upper() == "PIXEL"
        ]:
            struct[nid] = []
            nids = self.g.get_neighbor_list(nid, "PIXEL",just_ids=True)
            for pxid in nids:
                struct[nid].append(pxid)
        return struct



    def classify_nid_list_to_px(self, nid_list) -> dict:
        """
        Takes a list[str] of nids and sorts its ntype to px
        """
        # px_id:list[nid]
        px_struct = {}
        for nid in nid_list:
            print("nid classify_nid_list_to_px", nid)
            px_id = f"px{nid.split('_px')[1]}"
            if px_id not in px_struct:
                px_struct[px_id] = set()
            #ntype = nid.split("__")[0]
            px_struct[px_id].add(nid)
        return px_struct


    def split_qf_id(self, nid):
        # SPLIT ID INTO:
        # - ntype
        # - px_id
        # - item_index
        item_index=None
        ntype=None
        px_id=None
        try:
            id_parts = nid.split("__")
            print("QFU id_parts",id_parts)
            px_id = id_parts[1]
            ntype = id_parts[0]
            if len(id_parts) == 3:
                item_index = id_parts[2]
        except Exception as e:
            print(f"Err split_qf_id: {e}")
        return ntype, px_id, item_index

    def get_qf_nid(self, ntype, px_id) -> dict:
        """
        get ntype and pxid
        for quark and gluon:
        - create nids for each item_index
        - extend to id list
        """
        nids = {}
        if "quark" in ntype.lower():
            for i in range(3):
                id = f"{ntype}__{px_id}__{i}"
                nids[nid] = []
        elif "gluon" in ntype.lower():
            for i in range(8):
                id = f"{ntype}__{px_id}__{i}"
                nids[nid] = []
        else:
            id = f"{ntype}__{px_id}"
            nids[nid] = []
        return nids




    def validate_extend_attrs(
            self, nid, light=False
    ):
        ntype, px_id, item_index = self.split_qf_id(nid)

        field_attrs:list[dict] = self.get_attrs_from_ntype(
            ntype,
            px_id,
            light=light
        )

        for fattrs in field_attrs:
            nnid = fattrs["id"]
            if self.g.G.has_node(nnid):
                print("Apply stim to", nnid)
                # check for item attrs
                current_attrs = self.g.G.nodes[nid]
                print("Merge attrs with", fattrs)
                current_attrs.update(fattrs)
                print("Write changes to G")
                self.g.update_node(current_attrs)
            else:
                self.g.add_node(
                    attrs=fattrs
                )
        print("finished validate_extend_attrs ")


    def deserailize_values(
            self,
            values:list[tuple[str, dict]]
    ):
        for nid, attrs in values:
            for k, v in attrs.items():
                attrs[k]=None#deserialize(v)
        print(f"desrialisation of {len(values)} values finsiehd")


    def get_attrs_from_ntype(self, ntype, px_id, pos=None, light=False, id=None) -> list:
        attrs=[]
        try:
            if ntype.lower() in FERMIONS:
                attrs = self.ferm_creator.create_ferm_attrs(
                    ntype,
                    px_id=px_id,
                    light=light,
                    id=nid,
                    pos=pos,
                )

            elif ntype.lower() in H:
                attrs = self.higgs_creator.higgs_attrs(
                    px_id=px_id,
                    id=nid,
                )

            elif ntype.lower() in G_FIELDS:
                attrs = self.g_creator.gfield(
                    ntype,
                )

            # add parent px -> table query
            for attr in attrs:
                attr["px"] = px_id
            #print("attrs get_attrs_from_ntype", attrs)
        except Exception as e:
            print(f"Err get_attrs_from_ntype: {e}")
        return attrs

    def batch_field(
            self,
            just_k,
            amount_nodes,
            dim,
    ) -> list:
        attrs=[]
        # datemn 100% auf cpu
        try:
            fstruct = []
            gstruct = []
            hstruct = []
            for f in FERMIONS:
                attrs = self.ferm_creator.create_f_core_batch(
                    f,
                    amount_nodes,
                    dim
                )
                fstruct.append(
                    list(attrs.values()))

            for h in H:
                attrs = self.higgs_creator.higgs_params_batch(
                    amount_nodes,
                    dim
                )
                hstruct.append(attrs)

            for gf in G_FIELDS:
                attrs = self.g_creator.gfield(
                    gf, dim, amount_nodes
                )
                gstruct.append(gf)

            if just_k:
                return list(attrs.keys())
            return [fstruct, gstruct, hstruct]
        except Exception as e:
            print(f"Err get_attrs_from_ntype: {e}")
        return attrs

    def batch_field_single(
            self,
            ntype,
            dim:int=3,
            just_k=False,
            just_v=False,
            param_struct={},
    ) -> list or dict:
        attrs=[] #todo
        field_id = ntype
        l_ntype = ntype.lower()
        try:
            if l_ntype in FERMIONS or l_ntype == "FERMION":
                attrs = self.ferm_creator.create_f_core_batch(
                    self.filter_ntype(l_ntype),
                    dim,
                )

            elif l_ntype in H or l_ntype == "HIGGS":
                attrs = self.higgs_creator.higgs_params_batch(
                    dim
                )

            elif l_ntype in G_FIELDS or l_ntype == "GAUGE":
                attrs = self.g_creator.gfield(
                    self.filter_ntype(l_ntype),
                    dim,
                )

            else:
                print("YEEEAAAHH", l_ntype)
                attrs = self.create_synthetic_default(
                    param_struct=param_struct,
                )

            # add params to node
            # create param nodes
            #self.add_params_link_fields(attrs, field_id, self.parent_ntype(l_ntype))
            if just_k:
                return list(attrs.keys())
            if just_v:
                return list(attrs.values())

        except Exception as e:
            print(f"Err batch_field_single: {e}")
        return attrs


    def check_field_id_sm(self, field_id):
        return field_id in FERMIONS or field_id in G_FIELDS or field_id in H


    def add_params_link_fields(self, keys, values, field_id, parent):
        try:
            for k, v in zip(keys, values):
                self.g.add_node(
                    dict(
                        id=k,
                        type="PARAM",
                        value=v,
                        parent=[parent]
                    )
                )

                # PARAM -> FIELD
                self.g.add_edge(
                    src=field_id,
                    trgt=k,
                    attrs=dict(
                        rel='has_param',
                        trgt_layer='PARAM',
                        src_layer='FIELD',
                    )
                )
        except Exception as e:
            print("Err add params to node", e)

    def create_synthetic_default(
            self,
            param_struct: dict,
    ):
        print("CREATE SYNTHETIC FIELD")
        field_value = self.field_value(dim=self.dim) #todo change

        return {
            key: (value if value is not None else field_value)
            for key, value in param_struct.items()
        }



    def change_state(self):
        """Changes state of ALL metadata entries"""
        upsert_data = {}
        data = self.db_manager.get_data(path=self.metadata_path)
        for mid, data in data["metadata"].items():
            current_state = data["status"]["state"]
            if current_state == "active":
                new_state = "inactive"
            else:
                new_state = "active"
            upsert_data[f"{mid}/status/state/"] = new_state

        self.db_manager.update_data(
            path=self.metadata_path,
            data=upsert_data
        )

    def extract_model_data(self):
        for nid, attrs in [
            (nid, attrs) for nid, attrs in self.g.datastore.nodes(data=True)
            if attrs.get("base_type").upper() in self.all_sub_fields]:
            # extract importantfields
            ntype = attrs.get("type")
            if ntype in FERMIONS:
                pass
                # todo

    def save_G_data_local(self, data, keys, path):
        dict_2_csv(data=data, keys=keys)

    def fetch_db_build_G(self):
        self.initial_frontend_data = {}

        initial_data = self.db_manager._fetch_g_data(
        )

        # Build a G from init admin_data and load in self.g
        self.g.build_G_from_data(initial_data)

    def get_all_node_sub_fields(self, nid, as_dict=False, edges=False):
        # get intern qf parents
        all_subs={}
        phi = self.g.get_neighbor_list(
            node=nid,
            target_type="PHI",
        )
        psis = self.g.get_neighbor_list(
            node=nid,
            target_type=[k.upper() for k, v in FERM_PARAMS.items()],
        )
        gs = self.g.get_neighbor_list(
            node=nid,
            target_type=[k.upper() for k, v in GAUGE_FIELDS.items()],
        )

        # michael.kobel@tu-dresden.de
        if as_dict is True:
            all_subs = {
                "PHI": phi,
                "FERMION": psis,
                "GAUGE": gs
            }
        else:
            all_subs = [
                phi,
                psis,
                gs
            ]

        if edges is True and as_dict is True:
            all_edges = self.edges_for_subs(
                nid,
                all_subs,
                as_dict
            )
            all_subs["EDGES"] = all_edges
        else:
            all_edges = {}
        return all_subs

    def get_ids_from_struct(self, all_subs):
        node_ids = []
        edge_ids = []
        for field_type, ntype in all_subs.items():
            if field_type.lower() == "EDGES":
                edge_ids.extend(
                    list(ntype.keys())
                )
                continue
            for ntype, nnids in ntype.items():
                node_ids.extend(
                    list(nnids.keys())
                )
        return node_ids, edge_ids



    def edges_for_subs(self, nid, all_subs: list[list[tuple]] or dict, as_dict) -> dict:
        """
        Uses the return value of get_all_node_sub_fields to get
        all edges of that connections and save it in
        dict: id: eattrs  -fromat
        """
        all_edges = {}
        if as_dict is True:
            for field_type, ntype in all_subs.items():
                #print("field_type", field_type)
                #print("type", ntype)
                for nntype, node_attrs in ntype.items():
                    #print(f"<<<<<NTYPE: {nntype}")
                    #print(f"<<<<<node_attrs: {node_attrs}")
                    node_id_list = list(node_attrs.keys())
                    for nnid in node_id_list:
                        edge_attrs = self.g.G.edges[nid, nnid]
                        #print(f"edge_attrs: {edge_attrs}")
                        eid = edge_attrs.get("id")
                        all_edges[eid] = edge_attrs
        else:
            for ntype, nnid in all_subs:
                edge_attrs = self.g.G.edges[nid, nnid]
                eid = edge_attrs.get("id")
                all_edges[eid] = edge_attrs
        #print("all_edges", all_edges)
        return all_edges



    def get_all_subs_list(self, check_key="type", datastore=False, just_attrs=False, just_id=False, sort_for_types=False, sort_index_key="entry_index", return_dict=False):
        if datastore is True:
            all_subs:list[dict or tuple] = self.get_all_field_nodes(
                self.g.datastore, check_key
            )
        else:
            all_subs:list[dict or tuple] = self.get_all_field_nodes(
                self.g.G, check_key
            )
        sorted_node_types={}

        if sort_for_types == True :
           #print("Sort nodes for types")
            for nid, attrs in all_subs:
                # save for type
                ntype = attrs.get("type")
                if ntype not in sorted_node_types:
                    sorted_node_types[ntype] = []

                # Deserialize Node
                converted_dict = self.field_utils.restore_selfdict(attrs)
                if return_dict is False:
                    sorted_node_types[ntype].append(converted_dict)
                else:
                    sorted_node_types[ntype][nid] = converted_dict

            # Sort nodes newest -> oldest
            if sort_index_key is not None:
               #print("Sort nodes for index")
                for node_type, rows in sorted_node_types.items():
                    new_rows = sorted(rows, key=lambda d: d[sort_index_key])
                    sorted_node_types[node_type] = new_rows

            #print("Return nodes")
            return sorted_node_types

        else:
            if just_attrs is True:
                return [attrs for _, attrs in all_subs]
            elif just_id is True:
                return [nid for nid, _ in all_subs]
            else:
                return all_subs

    def get_all_field_nodes(self, G, check_key):
        return [(nid, attrs) for nid, attrs in G.nodes(data=True) if
                attrs.get(check_key).upper() in self.all_sub_fields]


    def list_subs_ids(self):
        all_subs = [nid for nid, attrs in self.g.G.nodes(data=True) if attrs.get("type").upper() in [*ALL_SUBS, "PIXEL", "PX"]]
        return all_subs


    def create_connection(
            self,
            node_data: list,
            coupling_strength,
            env_id,
            con_type: str,  # "TRIPPLE" | "QUAD"
            nid
    ):
        # Does the tripple already exists?
        tripples = self.g.get_neighbor_list(nid, con_type)
        tripple_exists = True
        for tid, tattrs in tripples:
            for n in node_data:
                if n["id"] not in tid:
                    tripple_exists = False
                    break

        # No! -> create
        if tripple_exists is False:
            node_id = "_".join(n["id"] for n in node_data)
            ntype = con_type
            self.g.add_node(
                attrs={
                    "id": node_id,
                    "coupling_strength": coupling_strength,
                    "type": ntype,
                    "ids": node_data,
                }
            )

            # Connect tripple to ENV
            self.g.add_edge(
                src=env_id,
                trgt=node_id,
                attrs=dict(
                    rel=f"has_{con_type.lower()}",
                    src_layer="ENV",
                    trgt_layer=ntype
                )
            )

            # Connect nodes to tripple
            for item in node_data:
                self.g.add_edge(
                    src=node_id,
                    trgt=item["id"],
                    attrs=dict(
                        rel=ntype.lower(),
                        src_layer=ntype,
                        trgt_layer=item["type"]
                    )
                )
        else:
           print("Tripple already exists")


    def get_field_value(self, ntype):
        if ntype.lower() in FERMIONS:
            return "psi"
        elif ntype.lower() in G_FIELDS:
            return self._field_value(type=ntype)
        elif ntype.lower() == "phi":
            return "h"
