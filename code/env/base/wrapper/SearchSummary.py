class SearchSummary:
    def __init__(self, precision=4):
        self.__precision = precision
        self.__start_costs = []
        self.__end_costs = []
        self.__number_of_improvements = 0
        self.__improvement = []
        self.__improvement_percentage = []
        self.__search_durations = []
        self.__operations = []

    def add_entry(self, start_cost, end_cost, search_duration, operation_stats=None):
        """
        :param start_cost: cost before performing the search
        :param end_cost: cost after performing the search
        :param search_duration: total time consumed to perform the search
        :param operation_stats: statistics about the operation
        :return: dictionary object that represents the improvement during the search
        """
        from learn.util_func import convert_to_numeral
        start_cost = convert_to_numeral(start_cost)
        end_cost = convert_to_numeral(end_cost)
        improvement = start_cost - end_cost
        improvement_percentage = improvement / abs(start_cost) if start_cost != 0 else 0
        improvement = convert_to_numeral(improvement)
        improvement_percentage = convert_to_numeral(improvement_percentage)
        self.__start_costs.append(start_cost)
        self.__end_costs.append(end_cost)
        self.__number_of_improvements += 1 if improvement > 0 else 0
        self.__improvement.append(improvement)
        self.__improvement_percentage.append(improvement_percentage)
        self.__search_durations.append(search_duration)
        self.__operations.append(operation_stats)

        # storing so that in future if we want to access it
        return {
            "improvement": improvement,
            "improvement_percentage": improvement_percentage
        }

    def clear(self):
        self.__start_costs = []
        self.__end_costs = []
        self.__number_of_improvements = 0
        self.__search_durations = []
        self.__improvement = []
        self.__improvement_percentage = []

    def improvement_statistics(self):
        contains_improvement_records = False
        if len(self.__operations) > 0:
            contains_improvement_records = max([len(operations) for operations in self.__operations])
        summaries = []
        for i, improvement in enumerate(self.__improvement):
            if len(self.__operations[i]) > 0 or (not contains_improvement_records):
                summary_entry = {
                    "iteration": i,
                    "start_cost": round(self.__start_costs[i], self.__precision),
                    "end_cost": round(self.__end_costs[i], self.__precision),
                    "improvement": round(improvement, self.__precision),
                    "improvement_percentage": round(self.__improvement_percentage[i], self.__precision),
                    "search_duration": round(self.__search_durations[i], self.__precision),
                }

                from collections import defaultdict
                aggregate_operation_dict = defaultdict(int)
                for operation_count_dict in self.__operations[i]:
                    for key, value in operation_count_dict.items():
                        aggregate_operation_dict[key] += value

                for key, value in aggregate_operation_dict.items():
                    simplified_key = key.lower().replace("-", "_")
                    summary_entry[simplified_key] = value if int(value) == value else round(value, self.__precision)

                summaries.append(summary_entry)
        return summaries

    def summary(self):
        import numpy as np

        average_improvement_percentage = 0
        if len(self.__improvement_percentage) > 0:
            average_improvement_percentage = np.mean(self.__improvement_percentage)

        total_improvement = 0
        if len(self.__improvement) > 0:
            total_improvement = sum(self.__improvement)

        return {
            "total_improvements": self.__number_of_improvements,
            "improvement": round(total_improvement, self.__precision),
            "average_improvement_percentage": round(average_improvement_percentage, self.__precision)
        }
