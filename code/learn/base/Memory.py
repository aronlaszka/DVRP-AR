import copy
import uuid


class Experience:
    def __init__(
            self,
            post_insertion_state,
            immediate_reward,
            reward=None,
            next_request=None,
            post_insertion_next_state=None,
            add_duplication_for_improvement=False
    ):
        # store the minimum required details to compute feature vectors
        self.post_insertion_state = post_insertion_state.minimal()
        self.reward = reward
        self.immediate_reward = immediate_reward
        self.post_insertion_improved_state = None
        if add_duplication_for_improvement:
            # store the complete details for ease of computation purpose, memory wise inefficient !!!
            self.post_insertion_improved_state = copy.deepcopy(post_insertion_state)
        # store the minimum required details to compute feature vectors
        self.post_insertion_next_state = post_insertion_next_state.minimal()
        self.next_request = copy.deepcopy(next_request)
        self.processed_time = 0
        self.unique_id = str(uuid.uuid4())[:6]  # to consider when performing updates for the next_state

    def compress(self, enabled=True):
        """
        :return: compress the experience
        """
        import zlib
        import pickle
        if not enabled:
            return self
        return zlib.compress(pickle.dumps(self))


class Memory:
    def __init__(
            self,
            memory_dir,
            max_memory=1024,
            batch_size=32,
            experience_per_store=128,
            storage_capacity=128,
            save_experiences=False
    ):
        self.unique_id = str(uuid.uuid4())[:6]
        self.memory_dir = memory_dir
        self.max_memory = max_memory
        self.batch_size = batch_size
        self.save_experiences = save_experiences
        self.experience_per_store = experience_per_store
        self.storage_capacity = storage_capacity
        self.memory = [None for _ in range(self.max_memory)]
        self.memory_map = {}
        self.experience_counter = 0
        self.storage_counter = 0
        self.storage_map = {}
        # moving towards minimalistic memory storage and disabled the compression for now
        self.enable_compression = False

    def clear(self):
        self.memory = []
        self.memory_map = {}
        self.experience_counter = 0
        self.storage_counter = 0
        self.storage_map = {}

    def decompress(self, compressed_experience):
        """
        :param compressed_experience: compressed experience instance
        :return: decompressed experience instance
        """
        import pickle
        import zlib
        if not self.enable_compression:
            return compressed_experience
        return pickle.loads(zlib.decompress(compressed_experience))

    def add(self, experience, persist_storage=True):
        """
        :param experience: Single experience in the format of (state, action, next_state, reward and terminal/not)
        :param persist_storage: whether to store experience to local disk

        add the experience to the fixed size memory, at the overflow remove the memory in FIFO fashion
        """
        store_pointer = self.experience_counter % self.max_memory
        if self.save_experiences and persist_storage:
            self._save()

        self.memory_map[experience.unique_id] = store_pointer
        del self.memory[store_pointer]
        self.memory[store_pointer:store_pointer] = [experience.compress(self.enable_compression)]
        del experience
        self.experience_counter += 1

    def add_bulk(self, experiences):
        """
        add bulk of experiences to the memory, without updating any details
        """
        self.memory.extend(experiences)

    def _save(self):
        """
        save experience to local disk
        """
        if self.experience_counter % self.experience_per_store == 0 and self.experience_counter > 0:
            from common.general import dump_obj
            from datetime import datetime
            self.storage_counter = self.storage_counter % self.storage_capacity
            start_ptr = (self.experience_counter - self.experience_per_store) % self.max_memory
            experiences = self.memory[start_ptr:start_ptr + self.experience_per_store]
            time_stamp = int(datetime.now().timestamp())
            self.storage_map[self.storage_counter] = time_stamp
            dump_obj(experiences, f"{self.memory_dir}/{self.unique_id}/exp_{self.storage_counter}.pickle")
            dump_obj(self.storage_map, f"{self.memory_dir}/{self.unique_id}/storage_map.pickle")
            self.storage_counter += 1

    @staticmethod
    def quick_memory_load(dir_name, start_ptr, num_experiences):
        """
        :param dir_name: path to directory that store the experience
        :param start_ptr: start index of experience
        :param num_experiences: first n-files only
        load experience from local disk and return the experiences (only used for supervised learning)
        """
        from common.logger import logger
        from common.general import directory_exists, extract, load_obj
        all_experiences = []
        if directory_exists(dir_name):
            storage_map_files = extract(dir_name, "storage_map.pickle")
            experience_paths = extract(dir_name, ".pickle")
            filtered_files = []
            for file_path in experience_paths:
                if file_path not in storage_map_files:
                    filtered_files.append(file_path)

            if len(filtered_files) == 0:
                logger.error(f"No experiences files found in {dir_name}")

            start_ptr = start_ptr % len(filtered_files)

            success_count = 0
            all_experiences = []
            for file_path in filtered_files[start_ptr:]:
                experiences = load_obj(file_path)
                all_experiences.extend(experiences)
                success_count += 1
                if success_count >= num_experiences:
                    break
        return all_experiences

    def load(self, file_path, quick=False):
        """
        :param file_path: path to file that store the experience
        :param quick: whether to load experience quickly
        load experience from local disk
        """
        from common.general import load_obj
        from common.logger import logger
        experiences = load_obj(file_path)
        if len(experiences) == 0:
            logger.warning(f"No experience stored in the file: {file_path}")
            return False
        else:
            if quick:
                self.add_bulk(experiences)
            else:
                experiences = [self.decompress(experience) for experience in experiences]
                for experience in experiences:
                    self.add(experience, persist_storage=False)
            logger.info(f"Loaded experience from file: {file_path}")
        return True

    def load_from_dir(self, dir_name=None, quick_load=False, start_ptr=0, num_experiences=-1, reset_memory=False):
        """
        :param dir_name: path to directory that store the experience
        :param quick_load: whether to load experience quickly or not
        :param start_ptr: start index of experience
        :param num_experiences: first n-files only
        :param reset_memory: whether to reset memory before loading experience
        load experience from local disk
        """
        from common.logger import logger
        from common.general import directory_exists, extract, load_obj
        if reset_memory:
            self.memory = []

        if dir_name is None or dir_name == "" or not directory_exists(dir_name):
            dir_name = self.memory_dir

        if directory_exists(dir_name):
            storage_map_files = extract(dir_name, "storage_map.pickle")
            experience_paths = extract(dir_name, ".pickle")
            time_stamped_paths = {}
            if quick_load:
                filtered_files = []
                for file_path in experience_paths:
                    if file_path not in storage_map_files:
                        filtered_files.append(file_path)

                if start_ptr > len(filtered_files) - 1:
                    start_ptr = start_ptr % len(filtered_files)

                success_count = 0
                for file_path in filtered_files[start_ptr:]:
                    if self.load(file_path, quick=True):
                        success_count += 1
                    if success_count >= num_experiences:
                        break
            else:
                if len(storage_map_files) >= 1:
                    success_count = 0
                    for storage_map_file in storage_map_files:
                        storage_map = load_obj(storage_map_file)
                        unique_id = storage_map_file.split("/")[-2]
                        for file_path in experience_paths:
                            if "storage_map" not in file_path and unique_id in file_path:
                                ptr = int((file_path.split("/")[-1].replace(".pickle", "").split("_")[-1]))
                                if ptr in storage_map:
                                    time_stamped_paths[storage_map[ptr]] = file_path

                    for time_stamp in sorted(time_stamped_paths.keys(), reverse=False):
                        if self.load(time_stamped_paths[time_stamp]):
                            success_count += 1

                    logger.info(f"Successfully loaded {success_count} experiences")
                else:
                    for file_path in experience_paths:
                        self.load(file_path)
        else:
            logger.warning(f"Directory: {dir_name} does not exists")
        return True if len(self.memory) > 0 else False

    def update_the_solution(self, unique_id, updated_state, process_time=0):
        """
        :param unique_id: unique identifier for the experience
        :param updated_state: updated state (after applying the action)
        :param process_time: updated process time

        update the experience.post_insertion_state for better utilization in future training
        """
        if unique_id in self.memory_map.keys():
            pointer = self.memory_map[unique_id]
            experience = self.decompress(self.memory[pointer])
            # update the vehicle routes, use as warm start
            experience.post_insertion_improved_state = copy.deepcopy(updated_state)
            experience.processed_time += process_time
            self.memory[pointer] = experience.compress(self.enable_compression)

    def sample(self, random_seed=0):
        """
        :param random_seed: random seed to specify the randomness of sampling function
        :return: sample experience equal to the size of batch_size
        """
        valid_memory = [experience for experience in self.memory if experience]
        if len(valid_memory) < self.batch_size:
            return []
        import random
        random.seed(random_seed)
        sample = random.sample(valid_memory, self.batch_size)
        return [self.decompress(experience) for experience in sample]

    def sample_sequential(self, random_seed=0):
        """
        :param random_seed: random seed to specify the randomness of sampling function
        :return: sample experience equal to the size of batch_size
        """
        import math
        valid_memory = [experience for experience in self.memory if experience]
        if len(valid_memory) < self.batch_size:
            return []
        random_seed = random_seed % math.floor(len(valid_memory) / self.batch_size)
        sample = valid_memory[random_seed * self.batch_size:(random_seed + 1) * self.batch_size]
        return [self.decompress(experience) for experience in sample]

    def full_sample(self):
        return [experience for experience in self.memory if experience]
