import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional
import time

from tqdm import tqdm

from neem_interface_python.rosprolog_client import Prolog, atom
from neem_interface_python.utils.utils import Datapoint, Pose

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class NEEMError(Exception):
    pass


class NEEMInterface:
    """
    Low-level interface to KnowRob, which enables the easy creation of NEEMs in Python.
    For more ease of use, consider using the Episode object in a 'with' statement instead (see below).
    """

    def __init__(self):
        self.prolog = Prolog()
        self.pool_executor = ThreadPoolExecutor(max_workers=4)

        # Load neem-interface.pl into KnowRob
        neem_interface_path = os.path.join(SCRIPT_DIR, os.pardir, "neem-interface", "neem-interface.pl")
        self.prolog.ensure_once(f"ensure_loaded({atom(neem_interface_path)})")

    def __del__(self):
        # Wait for all currently running futures
        self.pool_executor.shutdown(wait=True)

    def clear_beliefstate(self):
        self.prolog.ensure_once("mem_clear_memory")

    ### NEEM Creation ###############################################################

    def start_episode(self, task_type: str, env_owl: str, env_owl_ind_name: str, env_urdf: str,
                      agent_owl: str, agent_owl_ind_name: str, agent_urdf: str, start_time: float = None):
        """
        Start an episode and return the prolog atom for the corresponding action.
        """
        q = f"mem_episode_start(Action, {atom(task_type)}, {atom(env_owl)}, {atom(env_owl_ind_name)}, {atom(env_urdf)}," \
            f"{atom(agent_owl)}, {atom(agent_owl_ind_name)}, {atom(agent_urdf)}," \
            f"{start_time if start_time is not None else time.time()})"
        res = self.prolog.ensure_once(q)
        return res["Action"]

    def stop_episode(self, neem_path: str, end_time: float = None):
        """
        End the current episode and save the NEEM to the given path
        """
        return self.prolog.ensure_once(
            f"mem_episode_stop({atom(neem_path)}, {end_time if end_time is not None else time.time()})")

    def add_subaction_with_task(self, parent_action, sub_action_type="dul:'Action'", task_type="dul:'Task'",
                                start_time: float = None, end_time: float = None) -> str:
        """
        Assert a subaction of a given type, and an associated task of a given type.
        """
        q = f"mem_add_subaction_with_task({atom(parent_action)},{atom(sub_action_type)},{atom(task_type)},SubAction)"
        solution = self.prolog.ensure_once(q)
        action_iri = solution["SubAction"]
        if start_time is not None and end_time is not None:
            self.prolog.ensure_once(f"kb_project(has_time_interval({atom(action_iri)}, {start_time}, {end_time}))")
        return action_iri

    def add_participant_with_role(self, action: str, participant: str, role_type="dul:'Role'") -> None:
        """
        Assert that something was a participant with a given role in an action.
        Participant must already have been inserted into the knowledge base.
        """
        q = f"mem_add_participant_with_role({atom(action)}, {atom(participant)}, {atom(role_type)})"
        self.prolog.ensure_once(q)

    def assert_tf_trajectory(self, points: List[Datapoint]):
        print(f"Inserting {len(points)} points")
        for point in tqdm(points):
            ee_pose_str = point.to_knowrob_string()
            self.prolog.ensure_once(f"""
                time_scope({point.timestamp}, {point.timestamp}, QS),
                tf_set_pose({atom(point.frame)}, {ee_pose_str}, QS).
            """)

    def get_tf_trajectory(self, frame: str, start_time: float, end_time: float) -> List[Datapoint]:
        pass

    def assert_transition(self, agent_iri: str, object_iri: str, start_time: float, end_time: float) -> Tuple[
        str, str, str]:
        res = self.prolog.ensure_once(f"""
            kb_project([
                new_iri(InitialScene, soma:'Scene'), is_individual(InitialScene), instance_of(InitialScene, soma:'Scene'),
                new_iri(InitialState, soma:'State'), is_state(InitialState),
                has_participant(InitialState, {atom(object_iri)}),
                has_participant(InitialState, {atom(agent_iri)}),
                holds(InitialScene, dul:'includesEvent', InitialState),
                has_time_interval(InitialState, {start_time}, {start_time}),

                new_iri(TerminalScene, soma:'Scene'), is_individual(TerminalScene), instance_of(TerminalScene, soma:'Scene'),
                new_iri(TerminalState, soma:'State'), is_state(TerminalState),
                has_participant(TerminalState, {atom(object_iri)}),
                has_participant(TerminalState, {atom(agent_iri)}),
                holds(TerminalScene, dul:'includesEvent', TerminalState),
                has_time_interval(TerminalState, {end_time}, {end_time}),

                new_iri(Transition, dul:'Transition'), is_individual(Transition), instance_of(Transition, soma:'StateTransition'),
                holds(Transition, soma:'hasInitialScene', InitialScene),
                holds(Transition, soma:'hasTerminalScene', TerminalScene)
            ]).
        """)
        transition_iri = res["Transition"]
        initial_state_iri = res["InitialState"]
        terminal_state_iri = res["TerminalState"]
        return transition_iri, initial_state_iri, terminal_state_iri

    def assert_agent_with_effector(self, effector_iri: str, agent_type="dul:'PhysicalAgent'",
                                   agent_iri: str = None) -> str:
        if agent_iri is None:
            agent_iri = self.prolog.ensure_once(f"""
                kb_project([
                    new_iri(Agent, dul:'Agent'), is_individual(Agent), instance_of(Agent, {atom(agent_type)})
                ]).""")["Agent"]
        self.prolog.ensure_once(f"kb_project(has_end_link({atom(agent_iri)}, {atom(effector_iri)}))")
        return agent_iri

    def assert_state(self, participant_iris: List[str], start_time: float = None, end_time: float = None,
                     state_class="soma:'State'", state_type="soma:'StateType'") -> str:
        state_iri = self.prolog.ensure_once(f"""
            kb_project([
                new_iri(State, soma:'State'), is_individual(State), instance_of(State, {atom(state_class)}),
                new_iri(StateType, soma:'StateType'), is_individual(StateType), instance_of(StateType, {atom(state_type)}), 
                holds(StateType, dul:'classifies',  State)
            ])
        """)["State"]
        if start_time is not None and end_time is not None:
            self.prolog.ensure_once(f"kb_project(has_time_interval({atom(state_iri)}, {start_time}, {end_time}))")
        for iri in participant_iris:
            self.prolog.ensure_once(f"kb_project(has_participant({atom(state_iri)}, {atom(iri)}))")
        return state_iri

    def assert_situation(self, agent_iri: str, involved_objects: List[str], situation_type="dul:'Situation'") -> str:
        situation_iri = self.prolog.ensure_once(f"""
            kb_project([
                new_iri(Situation, {atom(situation_type)}), is_individual(Situation), instance_of(Situation, {atom(situation_type)}),
                holds(Situation, dul:'includesAgent', {atom(agent_iri)})
            ])
        """)["Situation"]
        for obj_iri in involved_objects:
            self.prolog.ensure_once(f"kb_project(holds({atom(situation_iri)}, dul:'includesObject', {atom(obj_iri)}))")
        return situation_iri

    def assert_object_pose(self, obj_iri: str, obj_pose: Pose, start_time: float = None, end_time: float = None):
        print(f"Object pose of {obj_iri} at {start_time}: {obj_pose.to_knowrob_string()}")
        if start_time is not None and end_time is not None:
            qs_query = f"time_scope({start_time}, {end_time}, QS)"
        elif start_time is not None and end_time is None:
            qs_query = f"time_scope({start_time}, {time.time()}, QS)"
        else:
            qs_query = f"time_scope({time.time()}, {time.time()}, QS)"
        self.prolog.ensure_once(f"{qs_query}, tf_set_pose({atom(obj_iri)}, {obj_pose.to_knowrob_string()}, QS)")

    def assert_object_trajectory(self, obj_iri: str, obj_poses: List[Pose], start_times: List[float],
                                 end_times: List[float], insert_last_pose_synchronously=True):
        """
        :param insert_last_pose_synchronously: Ensure that the last pose of the trajectory has been inserted when this
        method returns
        """
        # Insert in reversed order, so it becomes easy to wait for the last pose of the trajectory
        obj_poses_reversed = list(reversed(obj_poses))
        start_times_reversed = list(reversed(start_times))
        end_times_reversed = list(reversed(end_times))
        obj_iris = [obj_iri] * len(obj_poses_reversed)
        generator = self.pool_executor.map(self.assert_object_pose, obj_iris, obj_poses_reversed, start_times_reversed,
                                           end_times_reversed)
        if insert_last_pose_synchronously:
            next(generator)

    ### Pouring specific functions ###
    def assert_task_and_roles(self, action_iri: str, task_type: str, source_iri: str, dest_iri: str, agent_iri: str,
                              goal_reached: bool = False) -> str:

        # One can make it more descriptive by adding transitions. Initial and Terminal states has objects as participants
        # are classified by different state types
        if task_type == "holding":
            self.prolog.ensure_once(f"""
            kb_project([
            new_iri(Task, "dul:'Task'"), has_type(Task, "soma:'Holding'"), executes_task({action_iri}, Task),
            has_participant({action_iri}, source_iri), has_participant({action_iri}, dest_iri), 
            new_iri(Role, soma:'AgentRole'), has_type(Role, soma:'AgentRole'), has_role({agent_iri},Role), 
            new_iri(Role1, "dul:'Role'"), new_iri(Role2, "dul:'Role'"), new_iri(Role3, "dul:'Role'"),
            has_type(Role1, "soma:'Container'"), has_type(Role2, "soma:'RecipientRole'"), 
            has_type(Role3, "soma:'SupportedObject'"), has_role({source_iri}, Role1), has_role({dest_iri}, Role2), 
            has_role({source_iri}, Role3)]).
            """)
            # Info on image schema can be added later with state transition by classifying the state by ContainmentState
            # or say
            # new_iri(ContTheory, "soma:'ContainmentTheory'"),
            # has_type(ContTheory, "soma:'ContainmentTheory'"), new_iri(VertTheory, "soma:'VerticalityTheory'"),
            # has_type(VertTheory, "soma:'VerticalityTheory'"), triple(ContTheory, "dul:'isClassifiedBy'", Role1),
            # triple(VertTheory, "dul:'isClassifiedBy'", Role2)
        elif task_type == "MovingTo" or task_type == "MoveLeft" or task_type == "MoveRight" or task_type == "MoveUp" \
                or task_type == "MoveDown":
            # There is SPG, LinkageTheory satisfied in this transition

            self.prolog.ensure_once(f"""
            kb_project([
            new_iri(Task, "dul:'Task'"), has_type(Task, "soma:{task_type}"), executes_task({action_iri}, Task)
            has_participant({action_iri}, source_iri), has_participant({action_iri}, dest_iri),
            new_iri(Role, soma:'AgentRole'), has_type(Role, soma:'AgentRole'), has_role({agent_iri},Role),
            new_iri(Role1, "dul:'Role'"), new_iri(Role2, "dul:'Role'"), new_iri(Role3, "dul:'Role'"),
            has_type(Role1, "soma:'Container'"), has_type(Role2, "soma:'RecipientRole'"),
            has_type(Role3, "soma:'MovedObject'"), has_role({source_iri}, Role1),
            has_role({dest_iri}, Role2), has_role({source_iri}, Role3)]).
            """)
        elif task_type == "TiltForward" or task_type == "TiltBackward":
            self.prolog.ensure_once(f"""
            kb_project([
            new_iri(Task, "dul:'Task'"), has_type(Task, "soma:{task_type}"), executes_task({action_iri}, Task)
            has_participant({action_iri}, source_iri), has_participant({action_iri}, dest_iri),
            new_iri(Role, soma:'AgentRole'), has_type(Role, soma:'AgentRole'), has_role({agent_iri},Role),
            new_iri(Role1, "dul:'Role'"), new_iri(Role2, "dul:'Role'"), new_iri(Role3, "dul:'Role'"),
            has_type(Role1, "soma:'Container'"), has_type(Role2, "soma:'RecipientRole'"),
            has_type(Role3, "soma:'NonVerticalObject'"), has_role({source_iri}, Role1),
            has_role({dest_iri}, Role2), has_role({source_iri}, Role3)]).
            """)
        elif task_type == "TiltBackward" and goal_reached is True:
            # once the goal is reached the container role is assigned to the destination object
            self.prolog.ensure_once(f"""
            kb_project([
            new_iri(Task, "dul:'Task'"), has_type(Task, "soma:{task_type}"), executes_task({action_iri}, Task)
            has_participant({action_iri}, source_iri), has_participant({action_iri}, dest_iri),
            new_iri(Role, soma:'AgentRole'), has_type(Role, soma:'AgentRole'), has_role({agent_iri},Role),
            new_iri(Role1, "dul:'Role'"), new_iri(Role2, "dul:'Role'"), new_iri(Role3, "dul:'Role'"),
            has_type(Role1, "soma:'Container'"), has_type(Role2, "soma:'RecipientRole'"),
            has_type(Role3, "soma:'NonVerticalObject'"), has_role({dest_iri}, Role1),
            has_role({dest_iri}, Role2), has_role({source_iri}, Role3)]).
            """)

    ### NEEM Parsing ###############################################################

    def load_neem(self, neem_path: str):
        """
        Load a NEEM into the KnowRob knowledge base.
        """
        self.prolog.ensure_once(f"mem_clear_memory, remember({atom(neem_path)})")

    def get_all_actions(self, action_type: str = None) -> List[str]:
        if action_type is not None:  # Filter by action type
            query = f"is_action(Action), instance_of(Action, {atom(action_type)})"
        else:
            query = "is_action(Action)"
        res = self.prolog.ensure_all_solutions(query)
        if len(res) > 0:
            return list(set([dic["Action"] for dic in
                             res]))  # Deduplicate: is_action(A) may yield the same action more than once
        else:
            raise NEEMError("Failed to find any actions")

    def get_all_states(self) -> List[str]:
        res = self.prolog.ensure_all_solutions("is_state(State)")
        if len(res) > 0:
            return list(set([dic["State"] for dic in res]))  # Deduplicate
        else:
            raise NEEMError("Failed to find any states")

    def get_interval_for_event(self, event: str) -> Optional[Tuple[float, float]]:
        res = self.prolog.ensure_once(f"event_interval({atom(event)}, Begin, End)")
        if res is None:
            return res
        return res["Begin"], res["End"]

    def get_object_pose(self, obj: str, timestamp: float = None) -> Pose:
        if timestamp is None:
            query = f"mem_tf_get({atom(obj)}, Pose)"
        else:
            query = f"mem_tf_get({atom(obj)}, Pose, {timestamp})"
        return Pose.from_prolog(self.prolog.ensure_once(query)["Pose"])

    def get_tf_trajectory(self, obj: str, start_timestamp: float, end_timestamp: float) -> List:
        res = self.prolog.ensure_once(f"tf_mng_trajectory({atom(obj)}, {start_timestamp}, {end_timestamp}, Trajectory)")
        return res["Trajectory"]

    def get_wrench_trajectory(self, obj: str, start_timestamp: float, end_timestamp: float) -> List:
        res = self.prolog.ensure_once(
            f"wrench_mng_trajectory({atom(obj)}, {start_timestamp}, {end_timestamp}, Trajectory)")
        return res["Trajectory"]

    def get_tasks_for_action(self, action: str) -> List[str]:
        res = self.prolog.ensure_all_solutions(f"""kb_call([executes_task({atom(action)}, Task), 
                                                           instance_of(Task, TaskType), 
                                                           subclass_of(TaskType, dul:'Task')])""")
        return [dic["Task"] for dic in res]

    def get_triple_objects(self, subject: str, predicate: str) -> List[str]:
        """
        Catch-all function for getting the 'object' values for a subject-predicate-object triple.
        :param subject: IRI of the 'subject' of the triple
        :param predicate: IRI of the 'predicate' of the triple
        """
        res = self.prolog.ensure_all_solutions(f"""kb_call(holds({atom(subject)}, {atom(predicate)}, X))""")
        if len(res) > 0:
            return list(set([dic["X"] for dic in res]))  # Deduplicate
        else:
            raise NEEMError("Failed to find any objects for triple")

    def get_triple_subjects(self, predicate: str, object: str) -> List[str]:
        """
        Catch-all function for getting the 'subject' values for a subject-predicate-object triple.
        :param predicate: IRI of the 'predicate' of the triple
        :param object: IRI of the 'object' of the triple
        """
        res = self.prolog.ensure_all_solutions(f"""kb_call(holds(X, {atom(predicate)}, {atom(object)}))""")
        if len(res) > 0:
            return list(set([dic["X"] for dic in res]))  # Deduplicate
        else:
            raise NEEMError("Failed to find any subjects for triple")


class Episode:
    """
    Convenience object and context manager for NEEM creation. Can be used in a 'with' statement to automatically
    start and end a NEEM context (episode).
    """

    def __init__(self, neem_interface: NEEMInterface, task_type: str, env_owl: str, env_owl_ind_name: str,
                 env_urdf: str, agent_owl: str, agent_owl_ind_name: str, agent_urdf: str, neem_output_path: str,
                 start_time=None):
        self.neem_interface = neem_interface
        self.task_type = task_type
        self.env_owl = env_owl
        self.env_owl_ind_name = env_owl_ind_name
        self.env_urdf = env_urdf
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
                                                                      self.agent_owl,
                                                                      self.agent_owl_ind_name, self.agent_urdf,
                                                                      self.start_time)
        self.episode_iri = \
            self.neem_interface.prolog.ensure_once(
                f"kb_call(is_setting_for(Episode, {atom(self.top_level_action_iri)}))")[
                "Episode"]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.neem_interface.stop_episode(self.neem_output_path)
