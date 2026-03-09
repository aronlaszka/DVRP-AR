import multiprocessing

maximum_worker_count = multiprocessing.cpu_count() - 1


class ConcurrentWrapper:
    """
        wrapper class for concurrent processing
    """

    def __init__(self, _function, _arguments, _number_of_workers, _wait_for_response=False, _flatten=False):
        self._function = _function
        self._arguments = _arguments
        self._number_of_workers = min(len(_arguments), maximum_worker_count, _number_of_workers)
        self._wait_for_responses = _wait_for_response
        self._flatten = _flatten

    def _execute_single_proc(self):
        """
            special case when it was assigned to run in single process
        """
        responses = []
        for _args in self._arguments:
            response = self._function(_args)
            if self._flatten and isinstance(response, list):
                responses.extend(response)
            else:
                responses.append(response)
        return responses

    def execute(self):
        if self._number_of_workers > 1:
            responses = self._execute()
            if self._wait_for_responses or self._flatten:
                # need to copy into another list as the iterator
                # for generator functions provides empty list at the end
                final_responses = []
                for response in responses:
                    if self._flatten and isinstance(response, list):
                        final_responses.extend(response)
                    else:
                        final_responses.append(response)
            else:
                final_responses = responses
        elif self._number_of_workers == 1:
            final_responses = self._execute_single_proc()
        else:
            raise ValueError("Number of workers should be greater than 0")
        return final_responses

    def _execute(self):
        raise NotImplementedError


def thread_pool_executor_wrapper(function, arguments, number_of_workers, wait_for_response=False, flatten=False):
    """
    This is a Wrapper function that can vehicle_route the instances in the multiprocessing mode and return results
    :param function: function to evaluate
    :param arguments: contains the list of arguments
    :param number_of_workers: provide the number of workers
    :param wait_for_response: by enabling this builder function wait-for process completion
    :param flatten: optionally flatten if the output of the function is list
    :return: return the results at the end
    """
    from concurrent.futures import ThreadPoolExecutor

    class TPEWrapper(ConcurrentWrapper):
        """
            builder class for concurrent evaluations using thread pool executor
        """

        def __init__(self, _function, _arguments, _number_of_workers, _wait_for_response=False, _flatten=False):
            super(TPEWrapper, self).__init__(_function, _arguments, _number_of_workers, _wait_for_response, _flatten)

        def _execute(self):
            with ThreadPoolExecutor(self._number_of_workers) as executor:
                responses = executor.map(self._function, self._arguments)
            return responses

    return TPEWrapper(function, arguments, number_of_workers, wait_for_response, flatten).execute()


def process_pool_executor_wrapper(function, arguments, number_of_workers, wait_for_response=False, flatten=False):
    """
    This is a Wrapper function that can vehicle_route the instances in the multiprocessing mode and return results
    :param function: function to evaluate
    :param arguments: contains the list of arguments
    :param number_of_workers: provide the number of workers
    :param wait_for_response: by enabling this builder function wait-for process completion
    :param flatten: optionally flatten if the output of the function is list
    :return: return the results at the end
    """

    from concurrent.futures import ProcessPoolExecutor

    class PPEWrapper(ConcurrentWrapper):
        """
            builder class for concurrent evaluations using process pool executor
        """

        def __init__(self, _function, _arguments, _number_of_workers, _wait_for_response=False, _flatten=False):
            super(PPEWrapper, self).__init__(_function, _arguments, _number_of_workers, _wait_for_response, _flatten)

        def _execute(self):
            with ProcessPoolExecutor(max_workers=self._number_of_workers) as executor:
                responses = executor.map(self._function, self._arguments)
            return responses

    return PPEWrapper(function, arguments, number_of_workers, wait_for_response, flatten).execute()
