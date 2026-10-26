from datetime import datetime
from typing import List, Dict, Tuple

import networkx as nx

from firegraph.equador.mcreator import ModuleCreator
from firegraph.qfu.field_utils import FieldUtils
from firegraph.qfu.qf_utils import QFUtils
from firegraph.utils.get_shape import get_shape
from firegraph.utils.id_gen import generate_id

_SM_DEBUG = "[SMManager]"


class SMManager:
    def __init__(self):
        pass

    def check_sm_exists(self, user_id: str = "public"):
        try:
            print(f"{_SM_DEBUG} check_sm_exists: user_id={user_id}")
            user = self.qb.row_from_id(user_id, table="users")
            if not user:
                print(f"{_SM_DEBUG} check_sm_exists: no user")
                return False
            
            user = user[0]
            ok = user.get("sm_stack_status") == "created"
            return ok
        except Exception as e:
            print(f"{_SM_DEBUG} check_sm_exists: error: {e}")
            import traceback
            traceback.print_exc()
            return False


    def main(self, user_id: str = "public"):
        """
        Upsert standard nodes and edges from firegraph.qfutils to DB tables.
        """
        print("PROCESSING STANDARD MANAGER WORKFLOW")

        # Ensure tables exist (prod, test, CLI all use same canonical DB)
        #self.qb.initialize_all_tables()

        # Create module stack
        qf:QFUtils = self._initialize_graph(G=nx.Graph())

        print("FINISHED SM WORKFLOW")


    def enable_sm(self, user_id: str, session_id: str, env_id: str):
        """
        Link environment to Standard Model modules and fields.

        Return Structure:
        {
            "sessions": {
                session_id: {
                    "envs": {
                        env_id: {
                            "modules": {
                                module_id: {
                                    "fields": [field_id, ...]
                                }
                            }
                        }
                    }
                }
            }
        }
        """

        try:
            print(f"ENABLING SM FOR Env: {env_id}, Session: {session_id}")
            fu = FieldUtils()
            sm_modules = fu.modules_fields

            env_module_links = []
            module_field_links = []

            now = datetime.now()

            for module_name, fields in sm_modules.items():
                mid = module_name

                # Link Env -> Module
                env_module_links.append({
                    "id": generate_id(),
                    "env_id": env_id,
                    "module_id": mid,
                    "session_id": session_id,
                    "user_id": user_id,
                    "status": "active",
                    "created_at": now,
                    "updated_at": now
                })

                for field_name in fields:
                    fid = field_name
                    # Link Module -> Field
                    module_field_links.append({
                        "id": generate_id(),
                        "module_id": mid,
                        "field_id": fid,
                        "session_id": session_id,
                        "env_id": env_id,
                        "user_id": user_id,
                        "status": "active",
                        "created_at": now,
                        "updated_at": now,
                    })

            # Upsert
            if env_module_links:
                for row in env_module_links:
                    self.qb.set_item("envs_to_modules", row, keys={"id": row["id"]})

            #
            if module_field_links:
                for row in module_field_links:
                    self.qb.set_item("modules_to_fields", row, keys={"id": row["id"]})

            #
            formatted_modules = {}
            for mid, fids in sm_modules.items():
                formatted_modules[mid] = {"fields": fids}

                print(f"{_SM_DEBUG} enable_sm: done")
                return {
                    "sessions": {
                        session_id: {
                            "envs": {
                                env_id: {
                                    "modules": formatted_modules
                                }
                            }
                        }
                    }
                }
        except Exception as e:
            print("")

    def _initialize_graph(self, env_id, g, qf) -> QFUtils:
        """Initialize QFUtils and load standard modules."""
        try:
            print(f"{_SM_DEBUG} _initialize_graph: starting")
            env_attrs = qf.create_env()
            shape = [get_shape(v) for v in env_attrs.values()]
            qf.g.add_node(
                dict(
                    id=env_id,
                    type="ENV",
                    keys=env_attrs.keys(),
                    values=env_attrs.values(),
                    axis_def=[
                        0
                        for _ in range(len(list(env_attrs.keys())))
                    ],
                    field_index=0, # necessary in Guard,
                    shape=shape
            ))
            qf.build_interacion_G()
            qf.build_parameter()

            module_creator = ModuleCreator(g=g, qfu=qf)
            module_creator.load_sm()
            #qf.g.print_status_G()
            print(f"{_SM_DEBUG} _initialize_graph: finished")

        except Exception as e:
            print(f"{_SM_DEBUG} _initialize_graph: error: {e}")
            raise RuntimeError("Failed to initialize the standard-model graph") from e
        return qf


    def _upsert_graph_content(self, qf: QFUtils, user_id: str):
        """Graph stays in memory / exported via cfg; no cloud or DB upsert."""
        print(f"{_SM_DEBUG} _upsert_graph_content: skipped (local-only pipeline)")

    def _extract_nodes(
            self,
            qf: QFUtils,
            user_id: str
    ) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """Extract module and field nodes from the graph."""
        modules = []
        param_rows = []
        method_rows = []
        field_rows = []

        for nid, attrs in qf.g.G.nodes(data=True):
            ntype = attrs.get("type")

            if ntype == "MODULE":
                # Get METHOD neighbors
                methods = qf.g.get_neighbor_list_rel(
                    node=nid,
                    trgt_rel="has_method",
                    as_dict=True
                )

                method_ids = list(methods.keys())
                print(f"{nid} has method ids: {method_ids}")

                # Get FIELD neighbors
                fields = qf.g.get_neighbor_list_rel(
                    node=nid,
                    trgt_rel="has_field",
                    as_dict=True
                )
                field_ids = list(fields.keys())
                print(f"{nid} has_field ids: {field_ids}")

                # Upsert Module
                module_data = {
                    "id": attrs.get("id", nid),
                    "user_id": user_id,
                    "status": "active",
                    "fields": field_ids,
                    "methods": method_ids,
                    "origin": "SM",
                    **{
                        k: v
                        for k, v in attrs.items() if k not in [
                            "type",
                            "id",
                            "params",
                            "code",
                            "module_index"
                        ]
                    },
                }
                modules.append(module_data)

            # FIELDS
            elif ntype == "FIELD":
                # Upsert Field
                keys = attrs.get("keys") or attrs.get("field_keys")
                values = attrs.get("values") or attrs.get("value")
                axis_def = attrs.get("axis_def")
                interactant_fields = attrs.get("interactant_fields")

                if keys is None or values is None:
                    print(f"Skipping malformed FIELD node {attrs}. Keys/Values missing.")
                    continue

                # extend with neighbor vals
                # todo: rm values -> receive in process param attrs
                param_neighbors = qf.g.get_neighbor_list(nid, "PARAM")
                for pid, pattrs in param_neighbors.items():
                    if pid not in keys:
                        # add val stack
                        values.append(pattrs.get("value", [0]))
                        keys.append(pattrs.get(pid))
                        list(axis_def).append(0) # todo

                # Ensure list types for JSON serialization
                if keys is not None and not isinstance(keys, list):
                    keys = list(keys)
                if values is not None and not isinstance(values, list):
                    values = list(values)
                if axis_def is not None and not isinstance(axis_def, list):
                    axis_def = list(axis_def)

                field_data = {
                    "id": nid,
                    "keys": keys,
                    "values": values, # values are field specific
                    "axis_def": axis_def,
                    "module_id": attrs.get(
                        "module_id",
                        attrs["parent"][0],
                    ),
                    "origin": "SM",
                    "description": "",
                    "interactant_fields": interactant_fields,
                }
                field_rows.append(field_data)

            # PARAMS
            elif ntype == "PARAM":
                param_data = {
                    "id": nid,
                    "param_type": attrs.get("param_type"),
                    "description": "",
                    "origin": "SM",
                    "const": attrs.get("const"),
                    "axis_def": attrs.get("axis_def", 0),
                    "value": attrs.get("value"),
                    "shape": attrs.get("shape"),
                }
                param_rows.append(param_data)

            elif ntype == "METHOD":
                param_ids = qf.g.get_neighbor_list_rel(
                    node=nid,
                    trgt_rel="requires_param",
                    as_dict=True
                )
                param_ids = list(param_ids.keys())

                method_data = {
                    "id": nid,
                    'return_key': attrs.get("return_key"),
                    "params": param_ids,
                    #"jax_code": attrs.get("jax_code", attrs.get("code", None)),
                    "code": attrs.get("code", None),
                    "axis_def": attrs.get("axis_def", None),
                }
                method_rows.append(method_data)

        print("ROWS CREATED:")
        print("modules", len(modules))
        print("field_rows", len(field_rows))
        print("param_rows", len(param_rows))
        print("method_rows", len(method_rows))
        return modules, field_rows, param_rows, method_rows

    def _extract_edges(self, qf: QFUtils, user_id: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract edges from the graph."""
        mfs = []
        ffs = []

        for u, v, attrs in qf.g.G.edges(data=True):
            src_layer = attrs.get("src_layer")
            trgt_layer = attrs.get("trgt_layer")

            if src_layer == "MODULE" and trgt_layer == "FIELD":
                # Module -> Field
                data = {
                    "id": generate_id(),
                    "module_id": u,
                    "field_id": v,
                    "user_id": user_id
                }
                mfs.append(data)

            elif src_layer == "FIELD" and trgt_layer == "FIELD":
                # Field -> Field
                data = {
                    "id": generate_id(),
                    "field_id": u,
                    "interactant_field_id": v,
                    "user_id": user_id
                }
                ffs.append(data)
        
        return mfs, ffs
