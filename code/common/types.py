from enum import Enum


class ObjectiveTypes(Enum):
    IdleTime = "idle-time"
    DriveTime = "drive-time"
    TravelTime = "travel-time"
    CustomObjectiveByRL = "rl-custom"


class TravelTimeSources(Enum):
    OSMnx = "osmnx"  # faster
    Euclidean = "euc"  # even faster


class LocationInterpolationTypes(Enum):
    RouteBased = "route-based"


class RequestActions(Enum):
    Pickup = "pickup"
    Dropoff = "dropoff"


class SolvingModes(Enum):
    INSERT = "insert"
    BULK_INSERT = "bulk_insert"
    RESTORE = "restore"
    SEARCH = "search"


class InsertionModes(Enum):
    SINGLE = "single"
    BULK = "bulk"


class InsertionHeuristics(Enum):
    RoutingAPI = "routing"  # re-generate the solution from scratch (if possible) and try to insert the new request
    ESI = "exhaustive"  # exhaustive search to perform insertion without altering prior assignments
    ESIVerified = "exhaustive-verified"  # ESI verified with LCCI
    RollingHorizon = "rolling-horizon"  # Rolling Horizon (baseline) (actually offline VRP)


class MetaHeuristics(Enum):
    RoutingAPI = "routing"  # improve the given solution for the objective specified
    SimulatedAnnealing = "sim_anneal"  # improve the given solution for the objective specified by more configurable
    NoSearch = "no_search"


class DNNBackend(Enum):
    Tensorflow = "tensorflow"  # default backend support for Keras
    Torch = "torch"  # for memory efficient training or inferences go for it


class DNNArchitecture(Enum):
    MLP = "mlp"  # multi-layer perceptron
    CNN = "cnn"  # convolution neural network based custom model
    KAN = "kan"  # Kolmogorov-Arnold Networks


class FeatureCodes(Enum):
    CappedIdleTime = "cia"
    CappedAvailability = "cac"


class RLAlgorithm(Enum):
    DQN = "dqn"
    VFA = "vfa"
    VFA2DQN = "vfa2dqn"


class ExecutionModes(Enum):
    Train = "training"
    OfflineTrain = "offline-train"
    FineTune = "fine-tune"
    Eval = "evaluation"
    EvalBest = "evaluation-best"
    GatherExperience = "gather-experience"


class TimeUnits(Enum):
    Day = "day"
    Hour = "hour"
    Minute = "minute"
    HectoSecond = "hecto-second"
    DecaSecond = "deca-second"
    Second = "second"
