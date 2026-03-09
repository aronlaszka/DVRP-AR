class MappedList(object):
    """
        Implementation of List with O(1) compute time
    """

    def __init__(self, inp_list=None, discard_duplicate=False):
        self.__list = []
        self.__map = {}
        self.__discard_duplicate = discard_duplicate
        self.append(inp_list)

    def copy(self):
        return MappedList(self.__list)

    def append(self, value):
        """
        :param value: new value or new values as a list
        add a new element to the new list
        """
        if value is None:
            return

        if isinstance(value, list):
            self.extend(value)
        else:
            if value in self:
                if self.__discard_duplicate:
                    # discard the duplicates
                    return
                raise ValueError(f"{value} is already presents in the list {self.__list}")
            self.__list.append(value)
            self.__map[value] = len(self.__list) - 1

    def extend(self, values):
        """
        :param values: list of values
        extend the current list with new values
        """
        if values is None:
            return

        if len(values) > 0:
            for k, value in enumerate(values):
                self.__map[value] = len(self.__list) + k
            self.__list.extend(values)

    def __remove(self, value):
        """
        :param value: specific value
        :return: remove specific value (this is an internal function)
        """
        if value in self:
            idx = self.__map[value]

            # swapping
            last_val = self.__list[-1]
            self.__list[-1] = value
            self.__list[idx] = last_val
            self.__map[last_val] = idx

            self.__list.pop()
            self.__map.pop(value)
        else:
            raise ValueError(f"value {value} is not present")

    def __delitem__(self, index):
        """
        :param index: index of element to remove

        delete the item specified by index
        """
        self.__remove(self.__list[index])

    def remove(self, value):
        """
        :param value: specific value
        :return: remove the specific value
        """
        if isinstance(value, list):
            if len(value) > 0:
                for item in value:
                    self.__remove(item)
        else:
            self.__remove(value)

    def __contains__(self, value):
        """
        This is a pseudo implementation to use .contains() / in call
        :param value: value
        :return: whether the value in the list or not
        """
        return value in self.__map

    def index(self, value):
        """
        This is pseudo implementation to use .index() call
        :param value: value
        :return: the index of the value
        """
        if value in self:
            return self.__map[value]
        raise ValueError(f"value {value} not exists in the list {self.__list}")

    def __len__(self):
        """
        :return: the size of the list
        """
        return len(self.__list)

    def __iter__(self):
        """
        :return: get the iterator
        """
        it = list.__iter__(self.__list)
        return it

    def __getitem__(self, idx):
        """
        :param idx: index of item in the list
        :return: return the item if the index is within the bound, else through index error
        """
        if isinstance(idx, int):
            if idx < len(self.__list):
                return self.__list[idx]
            raise IndexError(f"{idx} is out of the bounds")
        else:
            return self.__list[idx]

    def sort(self, key=None, reverse=False):
        """
        :param key: key function to sort the list
        :param reverse: indicates whether to reverse the order or not

        perform sorting based on the key function (if provided)
        """
        if len(self.__list) > 0:
            self.__list.sort(key=key, reverse=reverse)
            for i, value in enumerate(self.__list):
                self.__map[value] = i

    def get_list(self):
        """
        :return: the list instance
        """
        return self.__list
