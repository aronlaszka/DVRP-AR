import argparse
from argparse import ArgumentParser

from common.types import *

from common.types import TravelTimeSources


class CustomArgParser(ArgumentParser):
    def __init__(self):
        super(CustomArgParser, self).__init__()
        self.add_argument(
            "--data_dir",
            help="Either absolute path or relative path that points to the main directory that stores the data",
            type=str,
            default="../data",
            required=False
        )

        self.add_argument(
            "--data_id",
            help="Unique identifier to identify the data-set",
            type=str,
            default="MTD",
            required=False
        )

        self.add_argument(
            "--train_range",
            help="Range of sample index that represent the training data",
            action='append',
            type=int,
            default=[],
            required=False
        )

        self.add_argument(
            "--test_range",
            help="Range of sample index that represent the testing data",
            action='append',
            type=int,
            default=[],
            required=False
        )

        self.add_argument(
            "--allow_rejection",
            help="Boolean to indicate whether to acceptance or rejection is a choice",
            action=argparse.BooleanOptionalAction,
            default=False,
            required=False
        )

        self.add_argument(
            "--always_accept_percentage",
            help="Percentage of requests that always accepted",
            type=float,
            default=0.0,
            required=False
        )

        self.add_argument(
            "--use_ve_only_at_decision",
            help="Boolean to indicate whether to use value estimator at decision of accept or reject",
            action=argparse.BooleanOptionalAction,
            default=False,
            required=False
        )

        self.add_argument(
            "--acceptance_value_threshold",
            help="Value that ensure the acceptance threshold",
            type=float,
            default=0.0,
            required=False
        )

        self.add_argument(
            "--sample_idx",
            help="Sample index for indicating the problem instance (only applicable during evaluation)",
            type=int,
            default=1,
            required=False
        )

        self.add_argument(
            "--number_of_routes",
            help="Maximum number of routes at a given time",
            type=int,
            default=4,
            required=False
        )

        self.add_argument(
            "--number_of_requests",
            help="Number of requests need to be processed",
            type=int,
            default=-1,
            required=False
        )

        self.add_argument(
            "--route_start_time",
            help="Route start-time",
            type=int,
            default=0,
            required=False
        )

        self.add_argument(
            "--route_end_time",
            help="Route end-time",
            type=int,
            default=86_400_000,
            required=False
        )

        self.add_argument(
            "--route_am_capacity",
            help="Maximum allowed ambulatory capacity of each vehicle",
            type=int,
            default=8,
            required=False
        )

        self.add_argument(
            "--route_wc_capacity",
            help="Maximum allowed wheel-chair capacity of each vehicle",
            type=int,
            default=2,
            required=False
        )

        self.add_argument(
            "--depot_latitude",
            help="Latitude of Depot Location",
            type=float,
            default=35.057,
            required=False
        )

        self.add_argument(
            "--depot_longitude",
            help="Longitude of Depot Location",
            type=float,
            default=-85.268,
            required=False
        )

        self.add_argument(
            "--time_ahead",
            help="Time-ahead from current time to which the route plan is fixed (seconds)",
            type=int,
            default=0,
            required=False
        )

        self.add_argument(
            "--dwell_time_pickup",
            help="Dwell time at pickup (seconds)",
            type=int,
            default=0,
            required=False
        )

        self.add_argument(
            "--dwell_time_dropoff",
            help="Dwell time at dropoff (seconds)",
            type=int,
            default=0,
            required=False
        )

        self.add_argument(
            "--window_size",
            help="Maximum pickup window size (seconds)",
            type=float,
            default=1800,
            required=False
        )

        self.add_argument(
            "--detour_ratio",
            help="Maximum detour tolerated with respect to direct travel time",
            type=float,
            default=0.0,
            required=False
        )

        self.add_argument(
            "--detour_minimum",
            help="Maximum detour with respect in seconds",
            type=float,
            default=0,
            required=False
        )

        self.add_argument(
            "--wait_time_threshold",
            help="Wait time threshold (seconds)",
            type=int,
            default=0,
            required=False
        )

        self.add_argument(
            "--interpolation_type",
            help="Technique to interpolate the location",
            type=str,
            default=LocationInterpolationTypes.RouteBased.value,
            choices=[item.value for item in LocationInterpolationTypes],
            required=False
        )

        self.add_argument(
            "--result_dir",
            help="Relative path to the directory that used to store the results",
            type=str,
            default="outputs",
            required=False
        )

        self.add_argument(
            "--objective",
            help="Internal objective of that used while performing main goal of maximizing service rate",
            type=str,
            default=ObjectiveTypes.IdleTime.value,
            choices=[item.value for item in ObjectiveTypes],
            required=False
        )

        self.add_argument(
            "--search_objective",
            help="Internal search objective of that used while performing main goal of maximizing service rate",
            type=str,
            default=ObjectiveTypes.IdleTime.value,
            choices=[item.value for item in ObjectiveTypes],
            required=False
        )

        self.add_argument(
            "--insertion_approach",
            help="Insertion Approach",
            type=str,
            default=InsertionHeuristics.ESI.value,
            choices=[item.value for item in InsertionHeuristics],
            required=False
        )

        self.add_argument(
            "--search_approach",
            help="Search Approach",
            type=str,
            default=MetaHeuristics.SimulatedAnnealing.value,
            choices=[item.value for item in MetaHeuristics],
            required=False
        )

        self.add_argument(
            "--perform_search",
            help="Boolean to indicate whether to perform search or not",
            action=argparse.BooleanOptionalAction,
            default=False
        )

        self.add_argument(
            "--offline_solution_limit",
            help="Number of solution for Offline VRP using Google OR Tools",
            type=int,
            default=100,
            required=False
        )

        self.add_argument(
            "--max_offline_solution_limit",
            help="Maximum number of solution for Offline VRP using Google OR Tools",
            type=int,
            default=500,
            required=False
        )

        self.add_argument(
            "--fixed_search",
            help="Boolean to indicate whether to perform search with "
                 "fixed duration or based on actual time of next request",
            action=argparse.BooleanOptionalAction,
            default=False
        )

        self.add_argument(
            "--fixed_search_duration",
            help="Duration of fixed search if fixed_search and perform_search is enabled",
            type=int,
            default=5,
            required=False
        )

        self.add_argument(
            "--max_fixed_search_duration_train",
            help="Maximum Duration of fixed search while training agent",
            type=int,
            default=5,
            required=False
        )

        self.add_argument(
            "--solve_online",
            help="Boolean to indicate whether to solve the problem as online or not",
            action=argparse.BooleanOptionalAction,
            default=False,
            required=False
        )

        self.add_argument(
            "--execution_mode",
            help="indicate different execution mode",
            type=str,
            default=ExecutionModes.Eval.value,
            choices=[item.value for item in ExecutionModes],
            required=False
        )

        self.add_argument(
            "--travel_time_source",
            help="Source of travel time",
            type=str,
            default=TravelTimeSources.OSMnx.value,
            choices=[item.value for item in TravelTimeSources],
            required=False
        )

        self.add_argument(
            "--threads",
            help="Number of worker threads",
            type=int,
            default=1,
            required=False
        )

        self.add_argument(
            "--model_dir",
            help="Main directory for models files",
            type=str,
            default="../model",
            required=False
        )

        self.add_argument(
            "--model_arch",
            help="Model architecture determine Learning Architecture",
            type=str,
            default=DNNArchitecture.MLP.value,
            choices=[item.value for item in DNNArchitecture],
            required=False
        )

        self.add_argument(
            "--learn_algorithm",
            help="Learning algorithm to compute loss for gradient descent",
            type=str,
            default=RLAlgorithm.DQN.value,
            choices=[item.value for item in RLAlgorithm],
            required=False
        )

        self.add_argument(
            "--fast_estimation_iter",
            help="Number of iterations before running complete learning",
            type=int,
            default=2000,
            required=False
        )

        self.add_argument(
            "--feature_code",
            help="Feature Code that describe the input features",
            type=str,
            default=FeatureCodes.CappedIdleTime.value,
            required=False
        )

        self.add_argument(
            "--polynomial_dimension",
            help="Polynomial dimension of input feature (only applicable for Idle Time)",
            type=int,
            default=1,
            required=False
        )

        self.add_argument(
            "--look_ahead_horizon",
            help="Look ahead horizon for idle time computation",
            type=int,
            default=14400,
            required=False
        )

        self.add_argument(
            "--look_ahead_window_size",
            help="Window size of idle time computation",
            type=int,
            default=300,
            required=False
        )

        self.add_argument(
            "--look_ahead_grid_size",
            help="Grid size to divide the entire spatial area",
            type=int,
            default=1,
            required=False
        )

        self.add_argument(
            "--look_ahead_slot_count",  # tau
            help="Different number of slot count for idle time computation",
            type=int,
            default=4,
            required=False
        )

        self.add_argument(
            "--look_ahead_slot_step_size",
            help="Step size to consider look ahead slot",
            type=int,
            default=1,
            required=False
        )

        self.add_argument(
            "--number_of_routes_to_consider",  # tau
            help="Maximum number of routes to consider",
            type=int,
            default=3,
            required=False
        )

        self.add_argument(
            "--model_config_version",
            help="Model configuration version (to select the model versions)",
            type=int,
            default=1,
            required=False
        )

        self.add_argument(
            "--model_version",
            help="Model version to select specific model configuration to organize the inputs",
            type=int,
            default=0,
            required=False
        )

        self.add_argument(
            "--model_idx",
            help="Model Identifier to select specific model to evaluate the problem instances",
            type=int,
            default=-1,
            required=False
        )

        self.add_argument(
            "--dnn_threads",
            help="Number of worker number_of_threads for Deep Neural Network Libraries",
            type=int,
            default=1,
            required=False
        )

        self.add_argument(
            "--dnn_random_seed",
            help="Random seed to initialize weights when using Deep Neural Network Libraries",
            type=int,
            default=0,
            required=False
        )

        self.add_argument(
            "--use_gpu",
            help="Boolean to indicate whether to use gpu or not",
            action=argparse.BooleanOptionalAction,
            default=False,
            required=False
        )

        self.add_argument(
            "--max_memory",
            help="Total number of experiences that are allowed to be stored in the main memory",
            type=int,
            default=1024,
            required=False
        )

        self.add_argument(
            "--batch_size",
            help="Batch size of the experience that need to be sampled during training",
            type=int,
            default=32,
            required=False
        )

        self.add_argument(
            "--reward_approx_steps",
            help="Step to approximate rewards",
            type=int,
            default=100,
            required=False
        )

        self.add_argument(
            "--experience_per_store",
            help="Number of experience to be store in the persistence storage when storing as python object",
            type=int,
            default=128,
            required=False
        )

        self.add_argument(
            "--storage_capacity",
            help="Maximum size of experience storage that can be found in the local disk",
            type=int,
            default=128,
            required=False
        )

        self.add_argument(
            "--experience_dir",
            help="Main directory for experience files",
            type=str,
            default="",
            required=False
        )

        self.add_argument(
            "--sync_every",
            help="Sync target and main model after every k steps of training",
            type=int,
            default=5,
            required=False
        )

        self.add_argument(
            "--save_model_every",
            help="Save main and target model after every k steps of training",
            type=int,
            default=1,
            required=False
        )

        self.add_argument(
            "--number_of_episodes",
            help="Number of episodes of training",
            type=int,
            default=10000,
            required=False
        )

        self.add_argument(
            "--learning_rate",
            help="Learning rate of the optimizer",
            type=float,
            default=0.001,
            required=False
        )

        self.add_argument(
            "--reward_decay",
            help="Reward decay used in the bellman equation",
            type=float,
            default=1.0,
            required=False
        )

        self.add_argument(
            "--soft_update_factor",
            help="Factor at which target network is updated",
            type=float,
            default=0.1,
            required=False
        )

        self.add_argument(
            "--epsilon",
            help="Factor used to allow random-ness in selecting the action",
            type=float,
            default=0.1,
            required=False
        )

        self.add_argument(
            "--epsilon_min",
            help="Factor used to allow minimum random-ness in selecting the action",
            type=float,
            default=0.01,
            required=False
        )

        self.add_argument(
            "--epsilon_decay",
            help="Factor used to decay the random-ness probability",
            type=float,
            default=0.995,
            required=False
        )

        self.add_argument(
            "--load_experiences",
            help="Boolean to indicate whether to load experiences (from previous instance) or not",
            action=argparse.BooleanOptionalAction,
            default=False,
            required=False
        )

        self.add_argument(
            "--save_experiences",
            help="Boolean to indicate whether to save experiences or not",
            action=argparse.BooleanOptionalAction,
            default=False,
            required=False
        )

        self.add_argument(
            "--quick_train",
            help="Boolean to indicate whether to perform a quick training (by reducing search time)",
            action=argparse.BooleanOptionalAction,
            default=False,
            required=False
        )

        self.add_argument(
            "--write_individual_summaries",
            help="Boolean to indicate whether to write detailed individual summaries",
            action=argparse.BooleanOptionalAction,
            default=False,
            required=False
        )

        self.add_argument(
            "--log_search",
            help="Boolean to indicate whether to log search details (summary)",
            action=argparse.BooleanOptionalAction,
            default=False,
            required=False
        )

        self.add_argument(
            "--log_search_detailed",
            help="Boolean to indicate whether to log search details (detailed version with operation details)",
            action=argparse.BooleanOptionalAction,
            default=False,
            required=False
        )

        self.add_argument(
            "--test_prefix",
            help="Providing prefix for evaluation with different configuration of training models",
            type=str,
            default="",
            required=False
        )

        self.add_argument(
            "--rtv_rh",
            help="Rolling Horizon Factor for RTV",
            type=int,
            default=1,
            required=False
        )

        self.add_argument(
            "--rtv_interval",
            help="Interval for RTV",
            type=int,
            default=60,
            required=False
        )

        self.add_argument(
            "--rtv_time_limit",
            help="Time limit to spend on each vehicle for RTV generation (in seconds)",
            type=float,
            default=1.0,
            required=False
        )

        self.add_argument(
            "--rtv_bin_name",
            help="Name of RTV binary",
            type=str,
            default="rolling_horizon_m1",
            required=False
        )


def get_parsed_args():
    """
    :return: provide custom argument parser with few fixed configurations
    """
    args = CustomArgParser().parse_args()

    if args.data_id == "NYC":
        args.depot_latitude = 40.7405826
        args.depot_longitude = -73.9893518
        args.travel_time_source = TravelTimeSources.OSMnx.value
    elif args.data_id == "MTD":
        args.depot_latitude = 35.057
        args.depot_longitude = -85.268
        args.travel_time_source = TravelTimeSources.OSMnx.value
    elif args.data_id == "ABS":
        # this is specific to the current setting, please change it if you change
        # the abstract environment
        args.depot_latitude = 1250
        args.depot_longitude = 1250
        args.travel_time_source = TravelTimeSources.Euclidean.value
    else:
        raise ValueError(f"Invalid data set id, {args.data_id} is not supported")

    # these are default configurations for the AAAI-26
    args.route_am_capacity = 8
    args.route_wc_capacity = 0
    args.time_ahead = 0
    args.dwell_time_pickup = 0
    args.dwell_time_dropoff = 0
    args.window_size = 1800
    args.detour_ratio = 0.5
    args.detour_minimum = 900
    args.wait_time_threshold = 0
    args.train_range = list([0, 40])
    args.test_range = list([40, 60])
    args.solve_online = True  # solve the problem as dynamic VRPs (only this configuration is supported for AAAI-26)

    # don't change these configurations (when evaluating models)
    args.look_head_horizon = 14400
    args.look_ahead_window_size = 30

    if args.quick_train:
        args.fixed_search = True
        args.fixed_search_duration = 1
    feature_codes = list(set(args.feature_code.split("-")))
    for code in feature_codes:
        if code not in [item.value for item in FeatureCodes]:
            raise ValueError(f"Feature code {code} is not valid, Exiting !!!")
    return args