from common.Singleton import Singleton


class AgentManager(metaclass=Singleton):
    _instance = None

    @classmethod
    def __init__(cls):
        cls.__agent = None

    @classmethod
    def register(cls, agent):
        """
        :param agent: Agent instance
        register the agent instance
        """
        cls._instance.__agent = agent

    @classmethod
    def get_agent(cls):
        """
        get the registered agent
        """
        return cls._instance.__agent

    @classmethod
    def get_cost(cls, state=None, action=None, **kwargs):
        """
        :param state: state of the environment
            (i.e., current requests, incoming request (if applicable), current route manifests)
        :param action: updated route manifests
        :param kwargs: keyword arguments such as
            consider_threshold (used only when the objective is maximizing idle time)
            selected_route (used only when the objective is from basic route statistics)
        :return: the objective cost
        """
        return cls._instance.__agent.get_cost(state, action, **kwargs)

    @classmethod
    def get_value(cls, state=None, action=None, **kwargs):
        """
        :param state: state of the environment
            (i.e., current requests, incoming request (if applicable), current route manifests)
        :param action: updated route manifests
        :param kwargs: keyword arguments such as
            consider_threshold (used only when the objective is maximizing idle time)
            selected_route (used only when the objective is from basic route statistics)
        :return: the objective value
        """
        return cls._instance.__agent.get_value(state, action, **kwargs)

    @classmethod
    def get_best_action(cls, state, actions, is_insertion=False, training=False, use_target=False):
        """
        :param state: state of the environment when receiving an incoming request
        :param actions: possible ways to insert the requests without changing the order of previous manifest
        :param is_insertion: whether it is an insertion action or not
        :param training: whether the function is called during training or not
        :param use_target: whether to use the main model or target model
        :return: the best action and value corresponding to the action
        """
        return cls._instance.__agent.get_best_action(state, actions, is_insertion, training, use_target)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = AgentManager()
        return cls._instance
