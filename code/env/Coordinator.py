from common.types import RLAlgorithm, ExecutionModes, InsertionHeuristics, DNNArchitecture
from env.data.TimeMatrix import MatrixManager


class Coordinator:
    def __init__(self):
        from ortools.constraint_solver import routing_enums_pb2

        from common.arg_parser import get_parsed_args
        from common.logger import logger
        from common.types import ObjectiveTypes
        from env.data.Location import Location, LocationTypes
        from env.base.online.RoutingSolver import OnlineRoutingSolver

        args = get_parsed_args()
        # initialize the solver
        day_end_time = (24 + 24) * 3_600  # just add 24 hour to be safe !!!

        train_range_start = min(args.train_range)
        train_range_end = max(args.train_range)

        test_range_start = min(args.test_range)
        test_range_end = max(args.test_range)

        # add this validation to avoid test and train data samples are overlapping
        if not (train_range_end <= test_range_start or test_range_end <= train_range_start):
            raise AssertionError("Test and Train sample indices are overlapping !!!")

        self.train_indices = [idx for idx in range(train_range_start, train_range_end)]
        self.test_indices = [idx for idx in range(test_range_start, test_range_end)]

        # depot location of micro transit services
        self.depot_location = Location(
            latitude=args.depot_latitude,
            longitude=args.depot_longitude,
            dwell_time=0,
            loc_type=LocationTypes.DEPOT
        )

        if args.solve_online:
            first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_COST_INSERTION
        else:
            first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC

        self.solver = OnlineRoutingSolver(
            args=args,
            custom_config=
            {
                "day_max_time": day_end_time,
                "first_solution_strategy": first_solution_strategy,
                "meta_heuristic": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
                "depot_location": self.depot_location
            }
        )
        self.processed_model_ids = set()

        if args.execution_mode in [
            ExecutionModes.Train.value,
            ExecutionModes.FineTune.value,
            ExecutionModes.OfflineTrain.value,
            ExecutionModes.GatherExperience.value,
        ]:
            output_dir = (
                f"{args.execution_mode.replace('-', '_')}_"
                f"{args.learn_algorithm}_"
                f"{args.model_arch}_"
                f"{args.model_config_version}_"
                f"{args.model_version}_"
                f"{args.feature_code.replace('-', '_')}_"
                f"{args.polynomial_dimension}_"
                f"{args.look_ahead_horizon}_"
                f"{args.look_ahead_window_size}_"
                f"{str(args.learning_rate).replace('.', '_')}_"
                f"{args.perform_search}"
            )
        else:
            if args.objective == ObjectiveTypes.CustomObjectiveByRL.value and args.test_prefix != "":
                output_dir = (
                    f"{args.execution_mode.replace('-', '_')}_"
                    f"{args.sample_idx}_"
                    f"{args.test_prefix}_"
                    f"{args.perform_search}"
                )
            else:
                if args.test_prefix != "":
                    output_dir = (
                        f"{args.execution_mode.replace('-', '_')}_"
                        f"{args.sample_idx}_"
                        f"{args.objective}_"
                        f"{args.test_prefix}_"
                        f"{args.insertion_approach}_"
                        f"{args.search_approach}_"
                        f"{args.time_ahead}_"
                        f"{args.wait_time_threshold}_"
                        f"{args.perform_search}_"
                        f"{args.solve_online}"
                    )
                else:
                    output_dir = (
                        f"{args.execution_mode.replace('-', '_')}_"
                        f"{args.sample_idx}_"
                        f"{args.objective}_"
                        f"{args.insertion_approach}_"
                        f"{args.search_approach}_"
                        f"{args.time_ahead}_"
                        f"{args.wait_time_threshold}_"
                        f"{args.perform_search}_"
                        f"{args.solve_online}"
                    )

        self.output_dir = output_dir

        if args.execution_mode == ExecutionModes.EvalBest.value:
            status = self.is_evaluation_required()
            if not status:
                logger.warning(f"Model is already evaluated for the sample idx {args.sample_idx}")
                # raise exception, so that sync client will work properly
                raise Exception(f"Model is already evaluated for the sample idx {args.sample_idx}")

        elif args.execution_mode == ExecutionModes.Eval.value:
            self.gather_processed_model_ids(key="E")
        self.environment = self.create_environment()
        self.solver.environment = self.environment
        self.environment.solver = self.solver
        if args.execution_mode == ExecutionModes.Train.value:
            self.max_memory_usage_in_gb = 40  # 40 GB
        else:
            self.max_memory_usage_in_gb = 5  # 5 GB

        if args.execution_mode in [
            ExecutionModes.Train.value,
            ExecutionModes.OfflineTrain.value,
            ExecutionModes.FineTune.value,
            ExecutionModes.GatherExperience.value,
            ExecutionModes.Eval.value,
            ExecutionModes.EvalBest.value
        ]:
            # no need to load travel time matrices for the offline training process
            MatrixManager.instance().init_travel_time_source(
                args.travel_time_source, args.data_dir, args.data_id
            )

    def update_configurations(self, restricted=None):
        # refresh the thread configuration
        from common.arg_parser import get_parsed_args
        from common.types import ObjectiveTypes
        from common.general import directory_exists
        from learn.settings import set_threads

        limited_threads = False
        if restricted and any([directory_exists(res_dir) for res_dir in restricted]):
            limited_threads = True

        args = get_parsed_args()

        if args.objective == ObjectiveTypes.CustomObjectiveByRL.value:
            from learn.settings import set_physical_device

            if limited_threads:
                set_physical_device(number_of_threads=1, use_gpu=args.use_gpu)
            else:
                set_threads(args.dnn_threads, log_changes=True)
                set_physical_device(number_of_threads=args.dnn_threads, use_gpu=args.use_gpu)

        else:
            if not limited_threads:
                set_threads(args.threads, log_changes=True)
        return self

    def run(self):
        from common.arg_parser import get_parsed_args
        args = get_parsed_args()
        self.register_agent()
        match args.execution_mode:
            case ExecutionModes.Train.value:
                self.train()
            case ExecutionModes.FineTune.value:
                self.train()
            case ExecutionModes.OfflineTrain.value:
                self.offline_train()
            case ExecutionModes.GatherExperience.value:
                self.gather_experience()
            case ExecutionModes.Eval.value:
                self.evaluate()
            case ExecutionModes.EvalBest.value:
                self.evaluate(best_episode="T")

    def is_training_required(self):
        """
            check whether the supervised training is required or not based on the availability of the model !!!
        """
        from common.arg_parser import get_parsed_args
        from common.general import directory_exists, file_exists
        from common.types import ObjectiveTypes

        args = get_parsed_args()
        if (args.objective == ObjectiveTypes.CustomObjectiveByRL.value and
                directory_exists(f"{self.output_dir}")):

            if directory_exists(f"{self.output_dir}/models/T"):
                return False
        return True

    def is_evaluation_required(self):
        """
            check whether the supervised trained model is already evaluated or not !!!
        """
        from common.arg_parser import get_parsed_args
        from common.general import directory_exists, file_exists
        from common.types import ObjectiveTypes

        args = get_parsed_args()
        if (args.objective == ObjectiveTypes.CustomObjectiveByRL.value and
                directory_exists(f"{self.output_dir}_SUMMARY")):

            if file_exists(f"{self.output_dir}_SUMMARY/T/{args.sample_idx}/statistics.csv"):
                return False
        return True

    def gather_processed_model_ids(self, key):
        """
            gather model identifiers that are processed already
        """
        from common.arg_parser import get_parsed_args
        from common.general import directory_exists, file_exists
        from common.types import ObjectiveTypes

        args = get_parsed_args()
        if (args.objective == ObjectiveTypes.CustomObjectiveByRL.value and
                directory_exists(f"{self.output_dir}_SUMMARY")):
            # only to evaluate the models from missing ones
            idx = 0
            if len(self.processed_model_ids) > 0:
                idx = max([int(proc_id.replace(key, "")) for proc_id in self.processed_model_ids]) + 1

            while file_exists(f"{self.output_dir}_SUMMARY/E{idx}/{args.sample_idx}/statistics.csv"):
                self.processed_model_ids.add(f"E{idx}")
                idx += 1

    def create_environment(self):
        """
            create environment instance and return the environment instance
        """
        from common.arg_parser import get_parsed_args
        args = get_parsed_args()
        from env.base.online.Environment import OnlineEnvironment
        return OnlineEnvironment(objective=args.objective, output_dir=self.output_dir)

    def register_agent(self):
        """
            register agent based on the model architecture
        """
        from learn.agent.Agent import Agent
        from learn.agent.AgentManager import AgentManager
        from common.arg_parser import get_parsed_args
        args = get_parsed_args()
        agent = Agent(output_dir=self.output_dir)
        AgentManager.instance().register(agent)
        self.solver.agent = agent
        agent.solver = self.solver

    def reset(self, sample_idx, model_idx, chain_idx=-1):
        """
        :param sample_idx: sample index to select one problem instance
        :param model_idx: model index (only applicable for RL based approach)
        :param chain_idx: chain index (only applicable for offline training)

        reset the environment by loading the requests specific to the sample_idx
        where the requests, routes, and dates are stored in file path specified input files
        """
        from common.arg_parser import get_parsed_args
        from env.data.TimeMatrix import MatrixManager
        # clear the node mapping to save some memory
        MatrixManager.instance().clear_node_mapping()

        args = get_parsed_args()
        input_files = {
            key: f"{args.data_dir}/{args.data_id}/base/{key}.csv"
            for key in ["requests", "dates"]
        }
        if chain_idx != -1:
            for key in ["requests", "dates"]:
                input_files[key] = f"{args.data_dir}/{args.data_id}/chains/{chain_idx}/{key}.csv"

        self.environment.reset(
            sample_idx=sample_idx,
            model_idx=model_idx,
            depot_loc=self.depot_location,
            input_files=input_files,
            capacity={
                "am": args.route_am_capacity,
                "wc": args.route_wc_capacity
            },
            time_ahead=args.time_ahead,  # this should be with respect to the data unit
            dwell_time_pickup=args.dwell_time_pickup,
            dwell_time_dropoff=args.dwell_time_dropoff,
            window_size=args.window_size,
            detour_ratio=args.detour_ratio,
            detour_minimum=args.detour_minimum,
        )

    def train(self):
        """
            train the DRL agent
        """
        import copy

        from common.arg_parser import get_parsed_args
        from common.types import ObjectiveTypes, InsertionHeuristics, MetaHeuristics
        from learn.base.Memory import Experience
        from learn.agent.AgentManager import AgentManager

        args = get_parsed_args()

        agent = AgentManager.instance().get_agent()
        agent.save_model(0)
        for sample_idx in range(1):
            self.reset(sample_idx=sample_idx, model_idx=sample_idx, chain_idx=args.dnn_random_seed)

            assert args.solve_online, "Training agent is only supported for online-vrp"
            assert args.objective == ObjectiveTypes.CustomObjectiveByRL.value, \
                "Training agent is only supported for custom objective"
            assert args.insertion_approach == InsertionHeuristics.ESI.value, \
                "Training agent is only supported for custom insertion approach"
            assert args.search_approach == MetaHeuristics.SimulatedAnnealing.value, \
                "Training agent is only supported for custom meta-heuristics"

            post_insertion_prev_state = None
            while not self.environment.is_done():
                instance_info = self.solver.update_state(is_bulk=False)
                pre_insertion_state = copy.deepcopy(self.environment.current_state)
                action, value = agent.act(instance_info)
                self.environment.process_action(action, value)

                post_insertion_state = copy.deepcopy(self.environment.current_state)

                if not self.environment.is_done():
                    next_request = self.environment.get_next_train()
                    if post_insertion_prev_state is not None:
                        experience = Experience(
                            post_insertion_state=post_insertion_prev_state,  # state after insertion
                            reward=1.0 if len(action.changed_route_ids) > 0 else 0.0,
                            immediate_reward=1.0 if len(action.changed_route_ids) > 0 else 0.0,
                            next_request=next_request,  # next request
                            post_insertion_next_state=post_insertion_state,
                            add_duplication_for_improvement=True
                        )

                        agent.add(experience)
                        expected_state_value = 0
                        if args.reward_decay < 1.0:
                            expected_state_value = (
                                    (1 / (1 - args.reward_decay)) *
                                    post_insertion_state.get_service_rate()
                            )
                        agent.tensor_board.log(
                            state_value=agent.get_value(
                                pre_insertion_state, action, objective=ObjectiveTypes.CustomObjectiveByRL.value
                            ),
                            service_rate=post_insertion_state.get_service_rate(),
                            expected_state_value=expected_state_value
                        )
                        # import keras
                        # import gc
                        # keras.utils.clear_session()
                        # gc.collect()

                if args.learn_algorithm in [RLAlgorithm.DQN.value, RLAlgorithm.VFA2DQN.value]:
                    if args.learn_algorithm == RLAlgorithm.VFA2DQN.value:
                        if agent.learn_count > args.fast_estimation_iter:
                            if args.perform_search:
                                self.solver.search(
                                    search_approach=args.search_approach,
                                    fixed_search=args.fixed_search,
                                    fixed_search_duration=args.fixed_search_duration
                                )
                                self.solver.write_stats(execution_mode=args.execution_mode)
                    else:
                        if args.perform_search:
                            self.solver.search(
                                search_approach=args.search_approach,
                                fixed_search=args.fixed_search,
                                fixed_search_duration=args.fixed_search_duration
                            )
                            self.solver.write_stats(execution_mode=args.execution_mode)

                if not self.environment.is_done():
                    post_insertion_prev_state = copy.deepcopy(post_insertion_state)
                else:
                    post_insertion_prev_state = None

    @staticmethod
    def offline_train():
        """
            perform offline training process based of VFA using the collected experience
        """
        from common.arg_parser import get_parsed_args
        from common.logger import logger
        from common.types import FeatureCodes, DNNArchitecture
        from learn.agent.AgentManager import AgentManager

        args = get_parsed_args()

        agent = AgentManager.instance().get_agent()
        max_train_experience_count = 65536
        max_test_experience_count = 65536

        test_experience = agent.memory.quick_memory_load(
            args.experience_dir + "/test",
            start_ptr=0,
            num_experiences=max_test_experience_count,
        )

        logger.info(f"Loaded Test Experiences: {len(test_experience)}")

        if agent.memory.load_from_dir(
                args.experience_dir + "/train",
                quick_load=True,
                start_ptr=0,
                num_experiences=max_train_experience_count,
                reset_memory=True
        ):
            kwargs = {
                "epochs": args.number_of_episodes,
                "save_model": True,
                "test_sample": test_experience
            }
            agent.supervised_learn(**kwargs)
        else:
            logger.error(f"No experience available in {args.experience_dir}")

        del test_experience
        agent.memory.clear()

    def gather_experience(self):
        """
            gather experiences to train the offline agent
        """
        import math
        import keras
        from datetime import datetime
        from common.arg_parser import get_parsed_args
        from common.logger import logger
        from common.types import ObjectiveTypes, InsertionHeuristics, MetaHeuristics
        from learn.base.Memory import Experience
        from learn.agent.AgentManager import AgentManager
        from learn.util_func import convert_to_numeral

        args = get_parsed_args()

        agent = AgentManager.instance().get_agent()
        self.reset(sample_idx=0, model_idx=0, chain_idx=args.dnn_random_seed)

        assert args.solve_online, "Training agent is only supported for online-vrp"
        assert args.objective == ObjectiveTypes.CustomObjectiveByRL.value, \
            "Training agent is only supported for custom objective"
        assert args.insertion_approach == InsertionHeuristics.ESI.value, \
            "Training agent is only supported for custom insertion approach"
        assert args.search_approach == MetaHeuristics.SimulatedAnnealing.value, \
            "Training agent is only supported for custom meta-heuristics"

        reward_array = []
        post_insertion_states = []
        candidate_post_insertion_state = None
        reward_decay = args.reward_decay

        value = 1.0
        best = 1.0
        tolerance_max = args.reward_approx_steps
        last_best = 0
        tolerance = tolerance_max
        steps = 1
        if not (0.0 < reward_decay < 1.0):
            raise ValueError("The reward decay must be strictly between 0 and 1, excluding 0 and 1")

        while True:
            # math.pow could go upto 14/15 significant bits and more precise
            value += math.pow(reward_decay, steps)
            if value > best:
                last_best = steps
                tolerance = tolerance_max
                best = value
            elif value == best:
                tolerance -= 1
            if tolerance == 0:
                break
            steps += 1

        logger.info(f"Number of steps required to approximated the state value {last_best}".upper())
        reward_approx_steps = min(args.reward_approx_steps, last_best)
        logger.info(f"Number of steps chosen to approximated the state value {reward_approx_steps}".upper())

        gamma_array = keras.ops.convert_to_tensor(
            [keras.ops.power(reward_decay, i) for i in range(reward_approx_steps)]
        )
        req_count = 0
        start_time = datetime.now()
        while not self.environment.is_done():
            instance_info = self.solver.update_state(is_bulk=False)
            action, value = agent.act_fixed_policy(instance_info, allow_random=False)
            self.environment.process_action(action, value)
            post_insertion_state = self.environment.current_state

            if not self.environment.is_done():
                curr_reward = 1.0 if len(action.changed_route_ids) > 0 else 0.0

                if len(post_insertion_states) < reward_approx_steps:
                    post_insertion_states.append(post_insertion_state.minimal())
                else:
                    candidate_post_insertion_state = post_insertion_states.pop(0)
                    post_insertion_states.append(post_insertion_state.minimal())

                if len(reward_array) < reward_approx_steps:
                    reward_array.append(curr_reward)
                else:
                    reward_array.pop(0)
                    reward_array.append(curr_reward)

                if len(reward_array) == reward_approx_steps and candidate_post_insertion_state:
                    computed_reward = convert_to_numeral(
                        keras.ops.dot(gamma_array, keras.ops.convert_to_tensor(reward_array))
                    )
                    assert computed_reward <= 1 / (1 - reward_decay)
                    experience = Experience(
                        post_insertion_state=candidate_post_insertion_state,  # state after insertion
                        # future reward for insertion action
                        reward=computed_reward,
                        immediate_reward=reward_array[0],
                        next_request=None,  # next request
                        post_insertion_next_state=post_insertion_states[0],
                        add_duplication_for_improvement=False
                    )
                    agent.add(experience, train=False)
            req_count += 1
            if req_count % 100 == 0:
                time_taken = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"Average time taken for last 100 insertion steps: {round(time_taken / 100, 3)} seconds".upper()
                )
                start_time = datetime.now()

    @staticmethod
    def is_learning_eval():
        """
        :return: whether the input argument specifies an evaluation configuration of learning based approach
        """
        from common.arg_parser import get_parsed_args
        from common.types import ObjectiveTypes, InsertionHeuristics, MetaHeuristics
        args = get_parsed_args()
        return args.objective == ObjectiveTypes.CustomObjectiveByRL.value and \
            args.insertion_approach == InsertionHeuristics.ESI.value and \
            args.search_approach == MetaHeuristics.SimulatedAnnealing.value

    def evaluate(self, test_episodes=None, start_idx=None, best_episode=None):
        """
        :param test_episodes: list of test episodes or single episode number
        :param start_idx: filter few of them in case based on the start-index (anything before start-index is ignored)
        :param best_episode: evaluate the best episode
        """

        from common.arg_parser import get_parsed_args
        from common.logger import logger
        from learn.agent.AgentManager import AgentManager

        args = get_parsed_args()

        agent = AgentManager.instance().get_agent()

        if test_episodes is None:
            test_episodes = [args.sample_idx]

        if isinstance(test_episodes, int):
            test_episodes = [test_episodes]

        model_ids = [-1]
        continuous_evaluation = False
        if self.is_learning_eval():
            if args.model_idx <= -1:

                if best_episode:
                    # for evaluating the best episode
                    if best_episode == "E*":
                        # take the last from sequential training steps
                        model_ids = agent.get_pre_trained_model_identifiers(args.model_dir, "E")
                        if len(model_ids) == 0:
                            return
                        model_ids = [model_ids[-1]]
                    else:
                        model_ids = [best_episode]
                else:
                    model_ids = agent.get_pre_trained_model_identifiers(args.model_dir, "E")
                    continuous_evaluation = True

                if len(model_ids) == 0:
                    raise FileNotFoundError(f"No Models Found in {args.model_dir}")
                else:
                    logger.info(f"Available list of model identifiers from directory {args.model_dir}: {model_ids}")
                    model_ids = [model_id for model_id in model_ids if model_id not in self.processed_model_ids]
                    logger.info(f"Unevaluated list of model identifiers to be evaluated: {model_ids}")
                    if start_idx is not None:
                        model_ids = model_ids[start_idx:]
                        logger.info(f"Filtered list of model identifiers to be evaluated: {model_ids}")
            else:
                model_id = f"E{args.model_idx}"
                if model_id not in self.processed_model_ids:
                    model_ids = [model_id]
                else:
                    model_ids = []
                    logger.info(
                        f"Model is already evaluated for model id {args.model_idx} "
                        f"for sample idx {args.sample_idx}"
                    )

        for model_idx in model_ids:
            self.evaluate_single_model(model_idx, test_episodes)

        if continuous_evaluation:
            # continuously fetch the details of the trained models and evaluate them
            while True:
                model_ids = agent.get_pre_trained_model_identifiers(args.model_dir, "E")
                self.gather_processed_model_ids(key="E")
                model_ids = [model_id for model_id in model_ids if model_id not in self.processed_model_ids]
                if len(model_ids) == 0:
                    break
                for model_idx in model_ids:
                    self.evaluate_single_model(model_idx, test_episodes)

    def evaluate_single_model(
            self,
            model_idx,
            test_episodes,
            model_dir=None,
            skip_reloading=False,
            allow_rejection=False
    ):
        """
        :param model_idx: specific model identifier
        :param test_episodes: list of test episodes or single episode number
        :param model_dir: directory to load the model
        :param skip_reloading: skip reloading last model
        :param allow_rejection: allow rejection of request
        """

        from common.arg_parser import get_parsed_args
        from common.logger import logger
        from learn.agent.AgentManager import AgentManager
        agent = AgentManager.instance().get_agent()

        args = get_parsed_args()

        if self.is_learning_eval():
            if not skip_reloading:
                agent.load_model(
                    model_dir=model_dir if model_dir is not None else args.model_dir,
                    episode=model_idx,
                    exit_on_failure=True,
                    retry_load=False,
                    maximum_retries=3,
                    retry_wait_time=60
                )

        if skip_reloading:
            # only overwrite when evaluating immediately
            agent.allow_rejection = allow_rejection

        for test_episode in test_episodes:
            if self.train_indices[0] <= test_episode < self.train_indices[1]:
                logger.warning(f"Evaluating Sample index: {test_episode}, from Train Dataset")
            elif self.test_indices[0] <= test_episode < self.test_indices[1]:
                logger.info(f"Evaluating Sample index: {test_episode}, from Test Dataset")

            self.reset(sample_idx=test_episode, model_idx=model_idx)

            if args.solve_online:
                while not self.environment.is_done():
                    if self.environment.has_more_than(args.always_accept_percentage):
                        self.__insert(agent)
                    else:
                        self.__insert(agent, skip_rejection=True)
                    self.__search()

                    if args.insertion_approach != InsertionHeuristics.RollingHorizon.value:
                        # summaries are not supported for rolling horizon
                        self.solver.write_stats(execution_mode=args.execution_mode, model_idx=model_idx)

                    # if self.is_learning_eval():
                    #     import keras
                    #     import gc
                    #     keras.utils.clear_session()
                    #     gc.collect()
            else:
                self.solver.offline_insert(
                    approach=args.insertion_approach,
                    solution_limit=args.offline_solution_limit,
                    max_solution_limit=args.max_offline_solution_limit
                )

                self.solver.write_stats(execution_mode=args.execution_mode, model_idx=model_idx)

            if skip_reloading and hasattr(agent, "tensor_board"):
                if allow_rejection:
                    agent.tensor_board.log(
                        service_rate_allow_rejection=self.environment.current_state.get_service_rate(percentage=True)
                    )
                else:
                    agent.tensor_board.log(
                        service_rate=self.environment.current_state.get_service_rate(percentage=True)
                    )

    def __insert(self, agent, skip_rejection=False):
        """
        :param agent: Agent instance
        :param skip_rejection: skip rejection of requests
        perform the insertion
        """
        from common.arg_parser import get_parsed_args
        args = get_parsed_args()
        if self.is_learning_eval():
            instance_info = self.solver.update_state(is_bulk=False)
            action, value = agent.act_eval(instance_info, skip_rejection=skip_rejection)
            self.environment.process_action(action, value)
        else:
            if args.insertion_approach == InsertionHeuristics.RollingHorizon.value:
                self.solver.rh_insert()
            else:
                self.solver.insert(
                    approach=args.insertion_approach,
                    solution_limit=1,
                    max_solution_limit=1
                )

    def __search(self):
        """
            perform the search
        """
        from common.arg_parser import get_parsed_args
        args = get_parsed_args()
        if args.perform_search:
            self.solver.search(
                search_approach=args.search_approach,
                fixed_search=args.fixed_search,
                fixed_search_duration=args.fixed_search_duration
            )

    def log_memory(self):
        """
        log current memory usage (for tracking purpose only)
        """
        from common.general import get_memory
        from common.logger import logger
        resp = get_memory()
        size = resp["size"]
        unit = resp["unit"]
        pids = resp["pids"]

        log_message = f"MEMORY USAGE [NPROC: {len(pids)}] (PIDS: {','.join(pids)}) : {round(size, 2)} {unit}"
        if unit == "GB":
            if size >= self.max_memory_usage_in_gb:
                raise MemoryError(f"[SIGNIFICANT] {log_message}")

        logger.info(log_message)
