from datetime import datetime


class RoutingMonitor:
    from ortools.constraint_solver import pywrapcp

    def __init__(self, model: pywrapcp.RoutingModel):
        self.model = model
        self.iteration_compute_times = []
        self.objective_values = []
        self.previous_time = datetime.now()

    def __call__(self):
        curr_time = datetime.now()
        self.iteration_compute_times.append((curr_time - self.previous_time).total_seconds())
        self.objective_values.append(self.model.CostVar().Min())
        self.previous_time = curr_time


class RoutingInstance:
    def __init__(self, model, manager, solution):
        self.model = model
        self.manager = manager
        self.solution = solution
