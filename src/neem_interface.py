import os
from typing import List, Tuple, Optional
import time
from .rosprolog_client import Prolog, atom

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class NEEMError(Exception):
    pass


class NEEMInterface:
    def __init__(self):
        self.prolog = Prolog()

        # Load neem-interface.pl into KnowRob
        neem_interface_path = os.path.join(SCRIPT_DIR, os.pardir, "../neem-interface/neem-interface.pl")
        self.prolog.once(f"ensure_loaded({atom(neem_interface_path)})")

        self.current_episode = None

    ### NEEM Creation ###############################################################

    def start_episode(self, task_type: str, env_owl: str, env_owl_ind_name: str, env_urdf: str, env_urdf_prefix: str,
                      agent_owl: str, agent_owl_ind_name: str, agent_urdf: str, start_time: float = None):
        """
        Start an episode and return the prolog atom for the corresponding action.
        """
        q = f"mem_episode_start(Action, {atom(task_type)}, {atom(env_owl)}, {atom(env_owl_ind_name)}, {atom(env_urdf)}," \
            f"{atom(env_urdf_prefix)}, {atom(agent_owl)}, {atom(agent_owl_ind_name)}, {atom(agent_urdf)}," \
            f"{start_time if start_time is not None else time.time()})"
        res = self.prolog.once(q)
        return res["Action"]

    def stop_episode(self, neem_path: str, end_time: float = None):
        """
        End the current episode and save the NEEM to the given path
        """
        return self.prolog.once(f"mem_episode_stop({atom(neem_path)}, {end_time if end_time is not None else time.time()})")

    def add_subaction_with_task(self, parent_action, sub_action_type, task_type) -> str:
        q = f"add_subaction_with_task({atom(parent_action)},{atom(sub_action_type)},{atom(task_type)},SubAction)"
        solution = self.prolog.once(q)
        if solution is not None:
            return solution["SubAction"]
        else:
            raise NEEMError("Failed to create action")

    ### NEEM Parsing ###############################################################

    def load_neem(self, neem_path: str):
        self.prolog.once(f"remember({atom(neem_path)})")

    def get_all_actions(self) -> List[str]:
        res = self.prolog.all_solutions("is_action(Action)")
        if len(res) > 0:
            return list(set([dic["Action"] for dic in
                             res]))  # Deduplicate: is_action(A) may yield the same action more than once
        else:
            raise NEEMError("Failed to find any actions")

    def get_interval_for_action(self, action: str) -> Optional[Tuple[float, float]]:
        res = self.prolog.once(f"event_interval({atom(action)}, Begin, End)")
        if res is None:
            return res
        return res["Begin"], res["End"]

    def get_tf_trajectory(self, obj: str, start_timestamp: float, end_timestamp: float) -> List:
        res = self.prolog.once(f"tf_mng_trajectory({obj}, {start_timestamp}, {end_timestamp}, Trajectory)")
        return res["Trajectory"]

    def get_wrench_trajectory(self, obj: str, start_timestamp: float, end_timestamp: float) -> List:
        res = self.prolog.once(f"wrench_mng_trajectory({obj}, {start_timestamp}, {end_timestamp}, Trajectory)")
        return res["Trajectory"]


class Episode:
    def __init__(self, neem_interface: NEEMInterface, task_type: str, env_owl: str, env_owl_ind_name: str,
                 env_urdf: str,
                 env_urdf_prefix: str, agent_owl: str, agent_owl_ind_name: str, agent_urdf: str, neem_output_path: str,
                 start_time=None):
        self.neem_interface = neem_interface
        self.task_type = task_type
        self.env_owl = env_owl
        self.env_owl_ind_name = env_owl_ind_name
        self.env_urdf = env_urdf
        self.env_urdf_prefix = env_urdf_prefix
        self.agent_owl = agent_owl
        self.agent_owl_ind_name = agent_owl_ind_name
        self.agent_urdf = agent_urdf
        self.neem_output_path = neem_output_path

        self.top_level_action_iri = None
        self.episode_iri = None
        self.start_time = start_time if start_time is not None else time.time()

    def __enter__(self):
        self.top_level_action_iri = self.neem_interface.start_episode(self.task_type, self.env_owl,
                                                                      self.env_owl_ind_name, self.env_urdf,
                                                                      self.env_urdf_prefix, self.agent_owl,
                                                                      self.agent_owl_ind_name, self.agent_urdf,
                                                                      self.start_time)
        self.episode_iri = \
            self.neem_interface.prolog.once(f"kb_call(is_setting_for(Episode, {atom(self.top_level_action_iri)}))")[
                "Episode"]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.neem_interface.stop_episode(self.neem_output_path)
