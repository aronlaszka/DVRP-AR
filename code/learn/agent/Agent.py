from common.types import ObjectiveTypes, FeatureCodes, DNNArchitecture


class Agent:

    def __init__(self, output_dir):
        import itertools
        from common.arg_parser import get_parsed_args
        from common.types import ObjectiveTypes, DNNArchitecture, RLAlgorithm, ExecutionModes, FeatureCodes

        args = get_parsed_args()
        self.solver = None
        self.output_dir = output_dir
        self.data_id = args.data_id
        self.wait_time_threshold = args.wait_time_threshold
        self.execution_mode = args.execution_mode
        self.train_mode = True if args.execution_mode in [
            ExecutionModes.Train.value, ExecutionModes.FineTune.value
        ] else False
        self.threads = args.threads
        self.dnn_threads = args.dnn_threads
        self.model_dir = args.model_dir
        self.model_arch = args.model_arch
        self.model_config_version = args.model_config_version
        self.model_version = args.model_version
        self._learn_fn_name = "Value"
        self.allow_rejection = args.allow_rejection
        self.use_ve_only_at_decision = args.use_ve_only_at_decision
        self.computed_features = {"train": None, "test": None}
        self.computed_targets = {"train": None, "test": None}
        if args.objective == ObjectiveTypes.CustomObjectiveByRL.value:
            import json
            import keras
            from learn.base.Memory import Memory
            from learn.base.TensorBoard import TensorBoard
            self.dnn_backend = keras.config.backend()
            self.dnn_random_seed = args.dnn_random_seed
            self.number_of_routes = args.number_of_routes
            self.number_of_routes_to_consider = args.number_of_routes_to_consider
            self.look_ahead_horizon = args.look_ahead_horizon
            self.look_ahead_window_size = args.look_ahead_window_size
            self.look_ahead_grid_size = args.look_ahead_grid_size
            self.look_ahead_total_slots = int(self.look_ahead_horizon / self.look_ahead_window_size)
            self.look_ahead_slot_count = min(args.look_ahead_slot_count, self.look_ahead_total_slots)
            self.look_ahead_slot_step_size = args.look_ahead_slot_step_size
            feature_codes = list(set(args.feature_code.split("-")))
            self.polynomial_dim = args.polynomial_dimension
            if self.model_arch in [
                DNNArchitecture.MLP.value, DNNArchitecture.KAN.value
            ]:
                self.model_suffix = (
                    f"{self.look_ahead_grid_size}_{self.look_ahead_slot_count}_"
                    f"{self.look_ahead_slot_step_size}_{self.number_of_routes_to_consider}"
                )
            if self.model_arch == DNNArchitecture.CNN.value:
                assert len(feature_codes) == 1
                self.input_dim = (self.look_ahead_total_slots, 1)
            else:

                # handcrafted features supported by MLPs and KANs
                self.input_dim = 0
                if FeatureCodes.CappedIdleTime.value in feature_codes:
                    self.input_dim += 1

                if FeatureCodes.CappedAvailability.value in feature_codes:
                    possible_size = (self.look_ahead_slot_count * self.number_of_routes_to_consider
                                     * int(self.look_ahead_grid_size ** 2))
                    self.input_dim += possible_size

                base_size = self.input_dim
                for dim in range(2, self.polynomial_dim + 1):
                    self.input_dim += len(
                        [
                            cmb for cmb in
                            itertools.combinations_with_replacement([x for x in range(base_size)], dim)
                        ]
                    )

                self.input_dim = int(self.input_dim)
            self.feature_code = args.feature_code
            self.use_gpu = args.use_gpu
            self.models = {}
            self.optimizers = {}
            self.learning_rate = args.learning_rate
            self.reward_decay = args.reward_decay
            self.__init_models_and_optimizers(sync=True)
            if args.execution_mode in [
                ExecutionModes.Train.value,
                ExecutionModes.FineTune.value,
                ExecutionModes.OfflineTrain.value,
                ExecutionModes.GatherExperience.value
            ]:
                self.memory = Memory(
                    memory_dir=f"{self.output_dir}/memory/",
                    max_memory=args.max_memory,
                    batch_size=args.batch_size,
                    experience_per_store=args.experience_per_store,
                    storage_capacity=args.storage_capacity,
                    save_experiences=args.save_experiences
                )
                self.fast_estimation_iter = args.fast_estimation_iter
                if self.execution_mode == ExecutionModes.Train.value:
                    if args.load_experiences:
                        self.memory.load_from_dir(args.experience_dir)
                    self.load_the_latest_model("E")
                elif self.execution_mode == ExecutionModes.FineTune.value:
                    self.load_model(
                        args.model_dir,
                        "T",  # pre-trained model *
                        load_target=True,
                        sync=True,
                        duplicate_target=True,
                        exit_on_failure=True
                    )
                self.soft_update_factor = args.soft_update_factor
                self.epsilon = args.epsilon
                self.epsilon_min = args.epsilon_min
                self.epsilon_decay = args.epsilon_decay
                self.sync_every = args.sync_every
                self.learn_count = 0
                self.learn_algorithm = args.learn_algorithm
                if self.learn_algorithm == RLAlgorithm.DQN.value:
                    self._learn_fn_name = "Q(s,a)"
                elif self.learn_algorithm == RLAlgorithm.VFA.value:
                    self._learn_fn_name = "V(s)"
                elif self.learn_algorithm == RLAlgorithm.VFA2DQN.value:
                    self._learn_fn_name = "Value"
                self.save_model_every = args.save_model_every
                model_suffix = self.dnn_random_seed
                if self.model_arch in [
                    DNNArchitecture.MLP.value, DNNArchitecture.KAN.value
                ]:
                    model_suffix = self.model_suffix
                self.tensor_board = TensorBoard(random_seed=model_suffix, output_dir=self.output_dir)
                self.max_fixed_search_duration_train = args.max_fixed_search_duration_train
                self.quick_train = args.quick_train
                self.terminate_if_gradient_zero_k_steps = 10  # terminate if the gradient is zero 10 continuous epoch
                self.gradient_zero_counts = 0
                self.write_configuration_summary(args.__dict__)

    def write_configuration_summary(self, configs):
        """
        :param configs: input argument configuration

        save the configuration to files
        """
        import json
        from common.general import create_dir

        create_dir(self.output_dir)
        json_config_file_path = f"{self.output_dir}/config.json"
        with open(json_config_file_path, 'w+') as json_config_file_path:
            json.dump(configs, json_config_file_path, indent=4)

    def build_model(self, is_target=False):
        """
        :param is_target: whether the model is target or not
        :return: generate model instance
        """
        import json
        from common.types import DNNArchitecture

        plot_model = True
        model_suffix = ""
        if self.model_arch == DNNArchitecture.MLP.value:
            from model.MLPModel import MLPModel
            model_class = MLPModel
            model_suffix = f"_{self.model_suffix}" if self.model_suffix != "" else ""
        elif self.model_arch == DNNArchitecture.KAN.value:
            from model.KANModel import KANModel
            model_class = KANModel
            model_suffix = f"_{self.model_suffix}" if self.model_suffix != "" else ""
            plot_model = False  # currently the plot_model function is not supported
        elif self.model_arch == DNNArchitecture.CNN.value:
            from model.CNNModel import CNNModel
            model_class = CNNModel
        else:
            raise NotImplementedError(f"Model Architecture {self.model_arch} not supported")

        prefix = f"target_{self.dnn_backend}" if is_target else f"{self.dnn_backend}"

        from learn.get_config import configs

        layer_config = configs[self.model_arch][self.model_config_version][self.model_version]

        with open(f"learn/norm.json", "rb") as normalize_config_file:
            normalize_config = json.load(normalize_config_file)

        return model_class(
            architecture={
                "dimension": {
                    "input": self.input_dim,
                    "feature_code": self.feature_code,
                    "polynomial": self.polynomial_dim
                },
                "layer_configuration": layer_config,
                "normalization_config": normalize_config,
            },
            file_names={
                "model": "{model_dir}/model" + f"{model_suffix}.keras",
                "weight": "{model_dir}/{episode}/" + f"{prefix}{model_suffix}.weights.h5",
                "norm": "{model_dir}/norm.json",
                "image": "{model_dir}/model.png"
            },
            configs={
                "plot_model": plot_model,
                "random_seed": self.dnn_random_seed,
                "threads": self.threads,
                "dnn_threads": self.dnn_threads,
                "use_gpu": self.use_gpu,
                "model_controls": {
                    "look_ahead_horizon": self.look_ahead_horizon,
                    "look_ahead_window_size": self.look_ahead_window_size,
                    "look_ahead_grid_size": self.look_ahead_grid_size,
                    "look_ahead_total_slots": self.look_ahead_total_slots,
                    "look_ahead_slot_count": self.look_ahead_slot_count,
                    "look_ahead_slot_step_size": self.look_ahead_slot_step_size,
                    "number_of_routes_to_consider": self.number_of_routes_to_consider
                }
            }
        )

    def __act(self, state, training=False, skip_rejection=False):
        # to do generate possible action given state
        # choose the valid actions

        # for insertion heuristics (you could simply check possible ways to perform exhaustive insertion)
        # and choose the action that maximize Q(s,a)
        # INPUT: {Set of Route Plans}, {Current Requests}, {Incoming request}
        # ACTION SPACE:
        # EXHAUSTIVE-SEARCH INSERTION: Possible ways to insert the incoming request
        # OUTPUT: Action as {Set of Updated Route Plans}, and {Acceptance/Rejection} of the incoming request
        from common.types import ExecutionModes

        allow_rejection = self.allow_rejection
        if skip_rejection:
            allow_rejection = False

        actions = self.solver.get_feasible_actions(state, allow_rejection=allow_rejection)

        if len(actions) <= 0:
            # at rejection the number of action equals 1 (which is basically empty action)
            raise AssertionError("Minimum number of actions should be at-least one")

        if training and self.execution_mode == ExecutionModes.Train.value:
            # only use the random when performing simple training
            import random
            rand_prob = random.random()
            if rand_prob <= self.epsilon:
                best_action, chosen_value = self.get_random_best_action(
                    state, actions, is_insertion=True, training=training, use_target=False
                )
                return best_action, chosen_value
        if self.allow_rejection and self.use_ve_only_at_decision:
            return self.act_with_ve_at_decision(state, training=training, use_target=False)
        return self.get_best_action(state, actions, is_insertion=True, training=training, use_target=False)

    def act(self, state):
        return self.__act(state, training=True)

    def act_eval(self, state, skip_rejection=False):
        return self.__act(state, training=False, skip_rejection=skip_rejection)

    def act_fixed_policy(self, state, objective=ObjectiveTypes.IdleTime.value, allow_random=False):
        import keras
        import random
        actions = self.solver.get_feasible_actions(state, allow_rejection=self.allow_rejection)
        values = [
            self.get_value(state, action, objective=objective)
            for action in actions
        ]
        if allow_random:
            rand_prob = random.random()
            if rand_prob <= self.epsilon:
                idx = random.choice(range(len(actions)))
                return actions[idx], values[idx]

        idx = keras.ops.argmax(values)
        chosen_value = keras.ops.max(values)
        best_action = actions[idx]
        return best_action, chosen_value

    def act_with_ve_at_decision(self, state, training=False, use_target=False):
        import keras
        actions = self.solver.get_feasible_actions(state, allow_rejection=True)
        values = [
            self.get_value(state, action, objective=ObjectiveTypes.IdleTime.value)
            for action in actions[:-1]
        ]
        if len(values) > 0:
            idx = keras.ops.argmax(values)
            chosen_value = keras.ops.max(values)
            best_action = actions[idx]

            selected_actions = [best_action, actions[-1]]

            inputs = self.future_transform(state, selected_actions)
            model = self.__get_model(use_target=use_target)
            values = model.inference(inputs, training=True if self.train_mode else training, squeeze=True)

            if values[-1] >= 1.0 + values[0]:
                best_action = actions[-1]
                chosen_value = values[-1]
            return best_action, chosen_value

        inputs = self.future_transform(state, actions)
        model = self.__get_model(use_target=use_target)
        values = model.inference(inputs, training=True if self.train_mode else training, squeeze=True)

        return actions[-1], values

    def _get_model_coef_strs(self):
        """
        :return: model coefficient as string
        """
        import copy
        import itertools
        main_coef_strs = []

        if FeatureCodes.CappedIdleTime.value in self.feature_code:
            main_coef_strs.append("Z")

        if FeatureCodes.CappedAvailability.value in self.feature_code:
            if self.look_ahead_grid_size == 1:
                for tau in range(self.look_ahead_slot_count):
                    for k in range(1, self.number_of_routes_to_consider + 1):
                        main_coef_strs.append(f"X({tau};{k})")
            else:
                for g_x in range(self.look_ahead_grid_size):
                    for g_y in range(self.look_ahead_grid_size):
                        for tau in range(self.look_ahead_slot_count):
                            for k in range(1, self.number_of_routes_to_consider + 1):
                                main_coef_strs.append(f"SX({g_x};{g_y};{tau};{k})")

        coef_strs = copy.deepcopy(main_coef_strs)

        for dim in range(2, self.polynomial_dim + 1):
            svs_x = itertools.combinations_with_replacement(main_coef_strs, dim)
            for sv_xx in svs_x:
                coef_strs.append("*".join(sv_xx))
        return coef_strs

    def transform(self, states):
        """
        :param states: list of states of the environment
        :return: transformation of each state-action pair
        """
        raw_inputs = [
            {
                "current_time": state.current_time,
                "routes": state.routes
            }
            for state in states
        ]
        return self.models["V"].generate_feature_vectors(raw_inputs)

    def future_transform(self, state, actions):
        """
        :param state: state of the environment
        :param actions: single action or list of action that could be taken from the current state
        :return: transformation of each state-action pair
        """
        if isinstance(state, dict):
            state = state["state"]

        if not isinstance(actions, list):
            actions = [actions]

        raw_inputs = [
            {
                "current_time": state.current_time,
                "routes": action.routes_after
            }
            for action in actions
        ]
        return self.models["V"].generate_feature_vectors(raw_inputs)

    def paired_transform(self, states, actions):
        """
        :param states: states of the environment
        :param actions: single best action corresponding to each state in same order
        :return: transformation of each state-action pair
        """
        raw_inputs = [
            {
                "current_time": states[i].current_time,
                "routes": action.routes_after
            }
            for i, action in enumerate(actions)
        ]
        return self.models["V"].generate_feature_vectors(raw_inputs)

    def __get_model(self, use_target=False):
        """
        :param use_target: whether to use the main model or target model
        return either the main model or the target model (if you use_target=True)
        """
        if use_target:
            return self.models["V_target"]
        return self.models["V"]

    def get_best_action(self, state, actions, is_insertion=False, training=False, use_target=False):
        """
        :param state: state of the environment when receiving an incoming request
        :param actions: possible ways to insert the requests without changing the order of previous manifest
        :param is_insertion: whether it is an insertion action or not
        :param training: whether the function is called during training or not
        :param use_target: whether to use the main model or target model
        :return: the best action and value corresponding to the action
        """
        import keras
        inputs = self.future_transform(state, actions)
        model = self.__get_model(use_target=use_target)
        values = model.inference(inputs, training=True if self.train_mode else training, squeeze=True)
        if len(actions) == 1:
            return actions[0], values

        if not (len(actions) == values.shape[0]):
            raise AssertionError(
                f"Each action much receive a value, expected values ({len(actions)}), actual values ({values.shape[0]})"
            )

        if is_insertion:
            if self.allow_rejection:
                immediate_reward = keras.ops.convert_to_tensor(
                    [1.0 if i < len(actions) - 1 else 0.0 for i in range(len(actions))]
                )
            else:
                immediate_reward = keras.ops.convert_to_tensor(
                    [1.0 for _ in range(len(actions))]
                )
            values *= self.reward_decay
            values += immediate_reward

        idx = keras.ops.argmax(values)
        chosen_value = keras.ops.max(values)
        best_action = actions[idx]
        # import gc
        # keras.utils.clear_session()
        # gc.collect()
        return best_action, chosen_value

    def get_random_best_action(self, state, actions, is_insertion=False, training=False, use_target=False):
        """
        :param state: state of the environment when receiving an incoming request
        :param actions: possible ways to insert the requests without changing the order of previous manifest
        :param is_insertion: whether it is an insertion action or not
        :param training: whether the function is called during training or not
        :param use_target: whether to use the main model or target model
        :return: the best action and value corresponding to the action
        """
        from numpy import random
        model = self.__get_model(use_target=use_target)
        if len(actions) == 1:
            chosen_value = model.inference(
                self.future_transform(state, actions), training=True if self.train_mode else training, squeeze=True
            )
            return actions[0], chosen_value
        indices = [idx for idx in range(len(actions))]
        idx = random.choice(indices)
        best_action = actions[idx]
        chosen_value = model.inference(
            self.future_transform(state, [actions[idx]]), training=True if self.train_mode else training, squeeze=True
        )

        if is_insertion:
            chosen_value *= self.reward_decay
            if self.allow_rejection:
                if idx < len(actions) - 1:
                    chosen_value += 1.0

        # import gc
        # keras.utils.clear_session()
        # gc.collect()
        return best_action, chosen_value

    def get_next_action(self, experience, use_target=True):
        """
        :param experience:
        that represents
        * state, action, next_state, reward of the environment after processing the incoming request
          and added to the manifest (if feasible)
        * unique_id: unique identifier of the experience to update the solution
        * processed_time: time taken to process the experience (until this evaluation)
        :param use_target: whether to use the main model or target model
        :return: the best action value corresponding to the action
        """
        import logging

        from common.logger import logger
        from common.types import MetaHeuristics

        # only log critical errors while optimizing routes using anytime algorithm
        logger.setLevel(logging.CRITICAL)

        fixed_search_duration = self.max_fixed_search_duration_train
        if len(experience.post_insertion_improved_state.current_requests) > 0:
            # cap the search duration to maximum possible from given state
            original_search_duration = experience.post_insertion_improved_state.current_requests[-1].max_search_duration
            adjusted_search_duration = original_search_duration - experience.processed_time
            fixed_search_duration = min(max(adjusted_search_duration, 1), self.max_fixed_search_duration_train)

        if self.quick_train:
            # this is a special setting for testing the implementation (not to be used)
            fixed_search_duration = 1

        next_action, updated_post_insertion_state = self.solver.search(
            search_approach=MetaHeuristics.SimulatedAnnealing.value,
            fixed_search=True,  # indicator to overwrite using actual search duration as that is really expensive
            fixed_search_duration=fixed_search_duration,
            state=experience.post_insertion_improved_state,
            use_target=use_target,
            change_environment=False
        )

        # update the solution to use it as warm start in next iteration
        self.memory.update_the_solution(experience.unique_id, updated_post_insertion_state, fixed_search_duration)

        updated_next_state_instance = self.solver.update_state_train(
            updated_post_insertion_state, experience.next_request
        )

        actions = self.solver.get_feasible_actions(updated_next_state_instance)
        next_insertion_action, _ = self.get_best_action(
            updated_next_state_instance, actions, is_insertion=True, training=True, use_target=use_target
        )

        logger.setLevel(logging.INFO)

        # import keras
        # import gc
        # keras.utils.clear_session()
        # gc.collect()
        return next_insertion_action

    def get_value(self, state=None, action=None, **kwargs):
        """
        :param state: state of the environment
            (i.e., current requests, incoming request (if applicable), current route manifests)
        :param action: updated route manifests
        :param kwargs: keyword arguments such as
            consider_threshold (used only when the objective is maximizing idle time)
            selected_route (used only when the objective is from basic route statistics)
        :return: the objective value
        """
        return self._get_value_and_cost(state, action, **kwargs)[self._learn_fn_name]

    def get_cost(self, state=None, action=None, **kwargs):
        """
        :param state: state of the environment
            (i.e., current requests, incoming request (if applicable), current route manifests)
        :param action: updated route manifests
        :param kwargs: keyword arguments such as
            consider_threshold (used only when the objective is maximizing idle time)
            selected_route (used only when the objective is from basic route statistics)
        :return: the objective cost
        """
        return self._get_value_and_cost(state, action, **kwargs)["cost"]

    def _get_value_and_cost(self, state=None, action=None, **kwargs):
        """
        :param state: state of the environment
            (i.e., current requests, incoming request (if applicable), current route manifests)
        :param action: updated route manifests
        :param kwargs: keyword arguments such as
            consider_threshold (used only when the objective is maximizing idle time)
            selected_route (used only when the objective is from basic route statistics)
        :return: the objective value
        """
        import copy
        from common.types import ObjectiveTypes

        if isinstance(state, dict):
            kwargs['current_time'] = state["current_time"]
            routes = state["routes"]
        else:
            kwargs['current_time'] = state.current_time
            routes = state.routes

        objective = kwargs['objective'] if 'objective' in kwargs else None
        use_target = kwargs['use_target'] if 'use_target' in kwargs else False
        if objective == ObjectiveTypes.CustomObjectiveByRL.value:
            model = self.__get_model(use_target=use_target)
            inputs = self.future_transform(state, action)
            value = model.inference(inputs, training=True if self.train_mode else False, squeeze=True)

            # import gc
            # import keras
            # keras.utils.clear_session()
            # gc.collect()
            return {
                self._learn_fn_name: value,
                "cost": -value
            }

        if action:
            kwargs['wait_time_threshold'] = self.wait_time_threshold
            value = action.get_value(**kwargs)
            return {
                self._learn_fn_name: value,
                "cost": -value
            }
        else:
            cost = 0
            for route in routes:
                sub_kwargs = copy.deepcopy(kwargs)
                sub_kwargs['wait_time_threshold'] = self.wait_time_threshold
                cost += route.get_current_cost(**sub_kwargs)
        return {
            self._learn_fn_name: -cost,
            "cost": cost
        }

    def add(self, experience, train=True):
        """
        :param experience: experience from the environment in the format (state, action, next_state, reward)
        :param train: whether to train the model or not

        add the incoming experience to the memory and train the model (if there is sufficient experiences)
        """
        self.memory.add(experience)
        if train:
            self.train()

    def train(self):
        """
        :return: train the model based on the random sample from the memory
        """
        import keras
        from learn.util_func import get_magnitude
        sample = self.memory.sample(random_seed=self.learn_count)
        if len(sample) > 0:
            train_response = self.__train_step(sample)
            self.learn_count += 1
            gradient_magnitude = get_magnitude(train_response["gradients"], order=2)
            self.tensor_board.log(
                actual_state_value_train=keras.ops.mean(train_response["predictions"]),
                expected_state_value_train=keras.ops.mean(train_response["target_values"]),
                actual_state_value_std_train=keras.ops.std(train_response["predictions"]),
                expected_state_value_std_train=keras.ops.std(train_response["target_values"]),
                actual_state_value_var_train=keras.ops.var(train_response["predictions"]),
                expected_state_value_var_train=keras.ops.var(train_response["target_values"]),
                train_mse=train_response["loss"],
                train_rmse=keras.ops.sqrt(train_response["loss"]),
                gradient=gradient_magnitude
            )
            if gradient_magnitude == 0.0:
                self.gradient_zero_counts += 1
                if self.gradient_zero_counts == self.terminate_if_gradient_zero_k_steps:
                    raise ValueError(
                        f"Zero gradient encountered for {self.terminate_if_gradient_zero_k_steps} learning epochs,"
                        f"terminating the training process"
                    )
            else:
                # reset at every time the gradient is not zero
                self.gradient_zero_counts = 0

            if self.learn_count % self.sync_every == 0:
                self.models["V_target"].sync_weights(self.models["V"], self.soft_update_factor)

            if self.learn_count % self.save_model_every == 0:
                self.save_model(f"E{self.learn_count // self.save_model_every}")

            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # import gc
        # keras.utils.clear_session()
        # gc.collect()

    def supervised_learn(self, **kwargs):
        """
        :param kwargs: key word arguments such as epochs (maximum number of epoches of training),
        save_model (save the trained model),
        test_sample (contains raw test experiences to be evaluated on the trained ML model)
        :return: train the model based on the pre collected experiences (with train, test and validation split)
        """
        from datetime import datetime
        from common.logger import logger
        sample = self.memory.full_sample()

        if len(sample) > 0:
            logger.info("Started supervised learning")
            start_time = datetime.now()
            self.__supervised_learn(sample, kwargs.get("epochs", 100))
            time_to_supervised_learn = (datetime.now() - start_time).total_seconds()
            self.tensor_board.log(time_to_supervised_learn=time_to_supervised_learn)
            logger.info(f"Completed supervised learning in {time_to_supervised_learn} seconds")

            # save the trained model
            if 'save_model' in kwargs and kwargs['save_model']:
                self.save_model("T")
                logger.info("Saved trained model")

            # evaluate the test dataset based on the trained model
            if 'test_sample' in kwargs and kwargs['test_sample']:
                logger.info("Started evaluating trained model")
                start_time = datetime.now()
                self.evaluate(kwargs['test_sample'])
                time_to_evaluate_trained = (datetime.now() - start_time).total_seconds()
                self.tensor_board.log(time_to_evaluate_trained=time_to_evaluate_trained)
                logger.info(f"Completed evaluating trained model in {time_to_evaluate_trained} seconds")

            # by default always plot the performance for linear models
            self.plot_performance()

        # import gc
        # import keras
        # keras.utils.clear_session()
        # gc.collect()

    def __dqn(self, sample):
        """
            Q(s,a) (or V(s)) = R(s, a) + γ argmax (a') Q(s', a')
            to estimate Q(s', a') run the anytime algorithm
        """
        import keras
        rewards = keras.ops.convert_to_tensor([experience.reward for experience in sample], dtype="float32")
        state_values = self.transform(states=[experience.post_insertion_state for experience in sample])
        predictions = self.models["V"].inference(inputs=state_values, training=True, squeeze=True)
        next_actions = [self.get_next_action(experience) for experience in sample]
        next_state_action_values = self.paired_transform(
            states=[experience.post_insertion_improved_state for experience in sample],
            actions=next_actions
        )
        target_values = rewards + self.reward_decay * self.models["V_target"].inference(
            inputs=next_state_action_values, training=True, squeeze=True
        )
        train_mse = keras.losses.mean_squared_error(target_values, predictions)
        return {
            "predictions": predictions,
            "target_values": target_values,
            "loss": train_mse
        }

    def __vfa(self, sample):
        """
            V(s) = R(s, a) + γ V(s')
        """
        import keras
        rewards = keras.ops.convert_to_tensor([experience.reward for experience in sample], dtype="float32")
        state_values = self.transform(states=[experience.post_insertion_state for experience in sample])
        predictions = self.models["V"].inference(inputs=state_values, training=True, squeeze=True)
        next_state_values = self.transform(states=[experience.post_insertion_next_state for experience in sample])
        target_values = rewards + self.reward_decay * self.models["V_target"].inference(
            inputs=next_state_values, training=True, squeeze=True
        )
        train_mse = keras.losses.mean_squared_error(target_values, predictions)
        return {
            "predictions": predictions,
            "target_values": target_values,
            "loss": train_mse
        }

    def __init_feature_and_targets(self, key_name, sample):
        """
        :param key_name: name of the key
        :param sample: list of experiences

        precompute the features and targets from input samples
        """
        import keras
        import random
        from datetime import datetime
        from common.logger import logger

        random.seed(self.dnn_random_seed)
        random.shuffle(sample)

        start_time = datetime.now()
        if self.computed_features[key_name] is None:
            self.computed_features[key_name] = self.transform(
                states=[experience.post_insertion_state for experience in sample]
            )
        if self.computed_targets[key_name] is None:
            self.computed_targets[key_name] = keras.ops.convert_to_tensor(
                [experience.reward for experience in sample], dtype="float32"
            )
        time_taken = (datetime.now() - start_time).total_seconds()
        logger.info(f"Generated features and targets for {key_name} with size {len(sample)} in {time_taken} seconds")

    def _reset_feature_and_targets(self, key_names):
        """
        :param key_names: list of keys whose values need to be reset from the temporary cache

        reset the temporary cache that stores the precomputed features and targets
        """
        for key_name in key_names:
            self.computed_features[key_name] = None
            self.computed_targets[key_name] = None

    def __supervised_learn(self, sample, max_number_of_epochs):
        """
        :param sample: list of training experiences
        :param max_number_of_epochs: maximum number of training epochs

        perform supervised training using Keras wrapper function fit()
        and save the best model that achieved after Early stopping based on the validation mean squared error
        """
        import os
        import keras
        import pandas as pd

        from learn.util_func import get_common_keras_metrics

        self.__init_feature_and_targets(key_name="train", sample=sample)

        full_train_inputs = self.computed_features["train"]
        full_train_targets = self.computed_targets["train"]

        self.models["V"].get_model().compile(
            optimizer=self.optimizers["V"],
            loss=keras.losses.MeanSquaredError(),
            metrics=get_common_keras_metrics()
        )
        history = self.models["V"].get_model().fit(
            full_train_inputs, full_train_targets,
            epochs=max_number_of_epochs, verbose=2, validation_split=0.2,
            batch_size=self.memory.batch_size,
            callbacks=
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=10,
                verbose=1,
                mode="auto",
                baseline=None,
                restore_best_weights=True,
                start_from_epoch=0,
            )
        )

        # record the training summaries to a CSV file
        dict_obj = history.history
        df = pd.DataFrame.from_dict(dict_obj)
        column_map = {
            col: col.replace("val_", "validation_") if col.startswith("val_") else "train_" + col
            for col in df.columns
        }
        column_map["loss"] = "train_mean_squared_error"
        column_map["val_loss"] = "validation_mean_squared_error"
        df = df.rename(columns=column_map)

        os.makedirs(f"{self.output_dir}/summary/", exist_ok=True)
        model_suffix = ""
        if self.model_arch in [DNNArchitecture.MLP.value, DNNArchitecture.KAN.value]:
            model_suffix = f"_{self.model_suffix}"
        df.to_csv(f"{self.output_dir}/summary/training{model_suffix}.csv", index=False)

        # import gc
        # keras.utils.clear_session()
        # gc.collect()

    def evaluate(self, sample):
        """
        :param sample: test samples

        evaluated the trained Keras model against the held out test samples
        """
        import keras

        self.__init_feature_and_targets(key_name="test", sample=sample)

        test_inputs = self.computed_features["test"]
        test_targets = self.computed_targets["test"]

        test_predictions = self.models["V"].get_model().predict(test_inputs, batch_size=self.memory.batch_size)
        test_predictions = test_predictions.squeeze()

        test_mse = keras.losses.mean_squared_error(test_targets, test_predictions)

        self.tensor_board.log(
            actual_state_value_test=keras.ops.mean(test_predictions),
            expected_state_value_test=keras.ops.mean(test_targets),
            actual_state_value_std_test=keras.ops.std(test_predictions),
            expected_state_value_std_test=keras.ops.std(test_targets),
            actual_state_value_var_test=keras.ops.var(test_predictions),
            expected_state_value_var_test=keras.ops.var(test_targets),
            test_me=keras.ops.mean(keras.ops.subtract(test_targets, test_predictions)),
            test_mbe=keras.ops.mean(keras.ops.subtract(test_predictions, test_targets)),
            test_mse=test_mse,
            test_rmse=keras.ops.sqrt(test_mse),
            test_r2=self.__r2(test_targets, test_predictions),
        )

    def plot_performance(self):
        """
        plot the performance of x vs f(x) if the models is based on the single feature and its polynomial version
        e.g.,  x vs f(x), x, x^2 vs f(x), etc.
        """
        from common.general import create_dir
        from common.types import DNNArchitecture

        valid_arch = self.model_arch in [DNNArchitecture.MLP.value, DNNArchitecture.KAN.value]

        single_feature = self.feature_code == FeatureCodes.CappedIdleTime.value

        if valid_arch and single_feature:

            import keras
            from pandas import DataFrame
            import matplotlib.pyplot as plt
            from learn.util_func import convert_to_numeral

            y_values = []
            x_values = []
            df = DataFrame()
            for i in range(0, 24 * self.number_of_routes, 1):
                x = round(i / 24, 6)
                feature = [keras.ops.power(x, p + 1) for p in range(self.input_dim)]
                x_values.append(x)
                y_values.append(
                    convert_to_numeral(self.models["V"].inference(keras.ops.convert_to_tensor([feature]))[0][0])
                )

            df["x_value"] = x_values
            df["y_value"] = y_values

            df.to_csv(f"{self.output_dir}/models/T/evaluation.csv", index=False)

            create_dir(f"{self.output_dir}/models/T/")
            df = df[df.x_value <= (self.look_ahead_horizon * self.number_of_routes) / 86400]  # 86400 = 60 * 60 * 24
            plt.plot(df.x_value, df.y_value)
            plt.xlabel("Sum of Total Idle time (in days)")
            plt.ylabel("Value of State")
            plt.legend(['Predicted'])
            plt.savefig(f"{self.output_dir}/models/T/evaluation.png")
            plt.close('all')

    @staticmethod
    def __r2(y_true, y_pred):
        """
        :param y_true: true values
        :param y_pred: predicted values
        """
        import keras
        sum_squares_residuals = keras.ops.sum(keras.ops.square(keras.ops.subtract(y_true, y_pred)))
        sum_squares_totals = keras.ops.sum(keras.ops.square(keras.ops.subtract(y_true, keras.ops.mean(y_true))))
        return keras.ops.subtract(1, keras.ops.divide(sum_squares_residuals, sum_squares_totals))

    def compute_loss(self, sample):
        from common.types import RLAlgorithm

        if self.learn_algorithm == RLAlgorithm.DQN.value:
            return self.__dqn(sample)

        elif self.learn_algorithm == RLAlgorithm.VFA.value:
            return self.__vfa(sample)

        elif self.learn_algorithm == RLAlgorithm.VFA2DQN.value:
            if self.learn_count > self.fast_estimation_iter:
                return self.__dqn(sample)
            else:
                return self.__vfa(sample)
        else:
            raise NotImplementedError(f"Learning algorithm {self.learn_algorithm} not yet supported")

    def __train_step(self, sample, loss_function=None):
        """
        :param sample: training samples
        :param loss_function: loss function definition, if not specified then, the
         loss function is defaulted to self.compute_loss

        perform single epoch of training step
        """
        from common.types import DNNBackend

        if loss_function is None:
            loss_function = self.compute_loss

        if self.dnn_backend == DNNBackend.Tensorflow.value:
            from learn.util_func import train_step_tensorflow
            return train_step_tensorflow(
                loss_function=loss_function,
                loss_function_args={"sample": sample},
                trainable_variables=self.models["V"].get_trainable_weights(),
                optimizer=self.optimizers["V"],
            )
        elif self.dnn_backend == DNNBackend.Torch.value:
            from learn.util_func import train_step_torch
            return train_step_torch(
                loss_function=loss_function,
                loss_function_args={"sample": sample},
                model=self.models["V"],
                trainable_variables=self.models["V"].get_trainable_weights(),
                optimizer=self.optimizers["V"],
                use_gpu=self.use_gpu
            )

    def __init_models_and_optimizers(self, sync=False):
        """
        :param sync: whether to sync the model and target mode
        
        build initial models and optimizers
        """
        import keras
        self.models["V"] = self.build_model(is_target=False)
        self.models["V_target"] = self.build_model(is_target=True)
        self.optimizers["V"] = keras.optimizers.Adam(learning_rate=self.learning_rate)

        if sync:
            self.models["V_target"].sync_weights(self.models["V"])

    def load_model(
            self,
            model_dir,
            episode,
            load_target=False,
            sync=False,
            duplicate_target=False,
            exit_on_failure=True,
            retry_load=False,
            maximum_retries=3,
            retry_wait_time=60,
            **kwargs
    ):
        """
        :param model_dir: directory containing model files
        :param episode: episode number
        :param load_target: whether to load the target model or not
        :param sync: whether to sync the model and target model (only applicable if load_target is True)
        :param duplicate_target: same as sync but do when the target model is absent (or not saved)
        :param exit_on_failure: whether to exit the program on error or not
        :param retry_load: whether to retry load the model or not
        :param maximum_retries: maximum number of retries before loading the model
        :param retry_wait_time: wait time in seconds between retries

        load the model that the keras format
        load the weights that stored in h5 format and
        load any custom parameters or variables that stored in the pickle format
        """
        from time import sleep
        import keras
        from common.logger import logger

        backend = keras.config.backend()
        alternate_backend = backend

        load_status = False
        kwargs['exit_on_failure'] = False
        for attempt in range(maximum_retries if retry_load else 1):
            load_status = self.models["V"].load_model(model_dir, episode, **kwargs)
            if not load_status:
                from common.types import DNNBackend
                for alternate_backend in [item.value for item in DNNBackend]:
                    if alternate_backend != backend:
                        kwargs['alternate_backend'] = alternate_backend
                        if self.models["V"].load_model(model_dir, episode, **kwargs):
                            # break once you figure out the correct backend
                            break
                if not load_status and attempt < maximum_retries:
                    # sleep for a while before loading the model again
                    logger.warning(f"Sleeping the script for {retry_wait_time} seconds, and retry loading model")
                    sleep(retry_wait_time)
            else:
                break

        if not load_status and exit_on_failure:
            raise ValueError(f"Failure while loading the model from {model_dir} of identifier {episode}")

        if load_target and load_status:
            target_load_status = self.models["V_target"].load_model(
                model_dir,
                episode,
                alternate_backend=alternate_backend,
                exit_on_failure=False if duplicate_target else True
            )
            if not target_load_status and duplicate_target:
                self.models["V_target"].build_model()
                self.models["V_target"].sync_weights(self.models["V"])
                logger.warning(
                    f"Duplicated the target {self.models['V_target'].__class__.__name__} with main model weights"
                )

            if sync and target_load_status:
                # optionally sync the target model
                logger.info(
                    f"Match the target {self.models['V_target'].__class__.__name__} with main model weights"
                )
                self.models["V_target"].sync_weights(self.models["V"])

    def save_model(self, episode):
        """
        :param episode: either the current episode number or string representing the current training check point

        save the model in the keras format
        save the weights in h5 format and
        save any custom parameters or variables in the pickle format
        """
        from common.types import ExecutionModes

        self.models["V"].save_model(f"{self.output_dir}/models", episode)
        # only save target models if there is a training or fine-tuning !!!
        if self.execution_mode in [ExecutionModes.Train.value, ExecutionModes.FineTune.value]:
            self.models["V_target"].save_model(f"{self.output_dir}/models", episode)

    def load_the_latest_model(self, episode_prefix=""):
        """
        :param episode_prefix: prefix to filter model types
        load the latest model
        """
        from common.general import directory_exists
        from common.logger import logger

        count = 0
        while directory_exists(f"{self.output_dir}/models/{episode_prefix}{count}"):
            count += 1

        if count > 0:
            count -= 1
            self.load_model(
                f"{self.output_dir}/models/", f"{episode_prefix}{count}",
                load_target=True,
                sync=False,
                exit_on_failure=True
            )
        else:
            logger.warning(f"There is no models found in {self.output_dir}/models")

    @staticmethod
    def get_pre_trained_model_identifiers(model_dir, episode_prefix=""):
        """
        :param model_dir: main model directory
        :param episode_prefix: prefix to filter model types
        :return: list of model identifiers that correspond to available models
        """
        from common.general import directory_exists

        count = 1
        ids = []
        while directory_exists(f"{model_dir}/{episode_prefix}{count}"):
            ids.append(f"{episode_prefix}{count}")
            count += 1
        return ids
