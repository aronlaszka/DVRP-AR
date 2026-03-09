def file_exists(file_name, extension=""):
    """
    :param file_name: file name with full path
    :param extension: file extension
    :return: returns whether the file exists or not
    """
    import os
    exists = False
    if file_name is not None:
        if os.path.exists(file_name + extension):
            exists = True
    return exists


def directory_exists(dir_name):
    """
    :param dir_name: directory name with the full path
    :return: returns whether directory exists or not
    """
    exists = False
    if file_exists(dir_name):
        exists = True
    return exists


def create_dir(dir_name, allow_file_path=True):
    """
    :param dir_name: name of directory, which need to be created
    :param allow_file_path: whether to allow the file path or not

    if enabled, create the directory that ensures the input file can be created without any
    exception
    """
    import os
    if allow_file_path:
        if "." in dir_name:
            last_slash = dir_name.rfind("/")
            if last_slash != -1:
                dir_name = dir_name[:last_slash + 1]
    if not directory_exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def extract(main_dir, file_ending=None, file_contains=None):
    """
    :param main_dir: name of the main directory
    :param file_ending: the ending of the file
    :param file_contains: the part of string that should be present in the file path
    :return: returns the list of files with corresponding file ending in the specified main directory
    """
    import os
    _files = []
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            if file_ending:
                if file.endswith(file_ending):
                    __file_path = os.path.join(root, file)
                    _files.append(__file_path)
            elif file_contains:
                __file_path = os.path.join(root, file)
                if file_contains in __file_path:
                    _files.append(__file_path)
            else:
                # always consider
                __file_path = os.path.join(root, file)
                _files.append(__file_path)
    return _files


def extract_dirs(main_dir, additional_filter=""):
    """
    :param main_dir: name of the main directory
    :param additional_filter: list of additional filters
    :return: returns the list of directories within main directory
    """
    import os
    if os.path.exists(main_dir):
        dirs = filter(os.path.isdir, [os.path.join(main_dir, f) for f in os.listdir(main_dir)])
        return [dir for dir in dirs if additional_filter in dir]
    return []


def get_df_from_dict(dict_obj, orient='columns'):
    """
    :param dict_obj: dictionary object or list of dictionary object
    :param orient: orient for orient while calling pd.DataFrame.from_dict()
    :return: instance of dataframe
    """
    import pandas as pd
    if isinstance(dict_obj, dict):
        dict_obj = [dict_obj]
    return pd.DataFrame.from_dict(dict_obj, orient=orient)


def load_data_as_pandas_df(file_path, encoding="utf-8", sep=",", file_type="csv", low_memory=True, exit_on_error=False):
    """
    :param file_path: path of the input file
    :param encoding: encoding used in the input file
    :param sep: seperator used in the input file to divide the values in the single row to set of columns
    :param file_type: file type of the input file, currently this code supported for CSV file type only
    :param low_memory: setting to limit memory usage
    :param exit_on_error: setting to exit on error
    function to load the data as pandas dataframe
    """
    if file_type == "csv":
        if file_exists(file_path):
            import pandas as pd
            return pd.read_csv(file_path, encoding=encoding, sep=sep, low_memory=low_memory)
        else:
            if exit_on_error:
                raise FileNotFoundError(f"File {file_path} does not exists")
            else:
                from common.logger import logger
                logger.error(f"File {file_path} does not exists")


def dump_obj(obj, file_name, extension=".pickle"):
    """
    :param obj: python object
    :param file_name: file to save the python object
    :param extension: file extension
    """
    import pickle
    if not file_name.endswith(extension):
        file_name = file_name + extension
    create_dir(file_name)
    with open(file_name, "wb") as dump_file:
        pickle.dump(obj, dump_file)


def load_obj(file_name, extension=".pickle"):
    """
    :param file_name: file to load the python object
    :param extension: file extension
    :return: python object
    """
    import pickle
    if file_exists(file_name):
        with open(file_name, "rb") as dump_file:
            obj = pickle.load(dump_file)
    elif file_exists(file_name + extension):
        with open(file_name + extension, "rb") as dump_file:
            obj = pickle.load(dump_file)
    else:
        raise FileNotFoundError(f"python byte-object is not found at the location {file_name}")
    return obj


def np_load_matrix(file_name, deliminator=",", ceil=True, dtype=None):
    """
    :param file_name: file to load the python object
    :param deliminator: deliminator used in the input file to divide the values in the single row to set of columns
    :param ceil: whether to ceil the loaded contents, ceiling ensures the triangular inequality
    and make the memory consumption low
    :param dtype: output data type

    load the matrix saved in text or csv file into 2D array
    """
    import numpy as np
    if dtype is None:
        dtype = np.int32
    base_matrix = np.loadtxt(file_name, delimiter=deliminator)
    if ceil:
        base_matrix = np.ceil(base_matrix)
    return base_matrix.astype(dtype)


def load_toml_file(file_name):
    """
    :param file_name: .configs file path
    :return: dictionary object with loaded configurations
    """
    import sys
    if sys.version_info.major == 3 and sys.version_info.minor <= 10:
        import toml
        read_mode = "r"
    else:
        import tomllib as toml
        read_mode = "rb"

    with open(file_name, read_mode) as toml_file_pointer:
        config = toml.load(toml_file_pointer)
    return config


def get_memory():
    """
    :return: total RSS memory usage for the process and its sub process
    """
    import os
    import psutil
    p_main = psutil.Process(os.getpid())
    size = p_main.memory_info().rss
    pids = [str(os.getpid())]
    for child in p_main.children(recursive=True):
        size += psutil.Process(child.pid).memory_info().rss
        pids.append(str(child.pid))
    unit = "MB"
    size = size / 1024 ** 2
    if size >= 1024:
        size /= 1024
        unit = "GB"

    return {
        "size": size, "unit": unit, "pids": pids
    }


def decode_time(formatted_time):
    """
    :param formatted_time: time as HHMMSS
    :return: returns time as seconds assuming that 12:00AM as 0 seconds
    """
    if not isinstance(formatted_time, int):
        try:
            formatted_time = int(formatted_time)
        except ValueError:
            raise ValueError(f"Invalid time value {formatted_time} of type {type(formatted_time)}")
    assert formatted_time >= 0, "time value should be greater than or equal to zero"
    _min, _sec = divmod(formatted_time, 100)
    _hour, _min = divmod(_min, 100)
    assert _min <= 59 and _sec <= 59, "minutes and seconds should be less than 60"
    return int(3600 * _hour + 60 * _min + _sec)


def convert_time_to_sec(time_str):
    """
    :param time_str: time as HH:MM:SS or HHMMSS
    :return: returns time as seconds assuming that 12:00AM as 0 seconds
    """
    if isinstance(time_str, str):
        if ":" in time_str:
            splits = time_str.split(":")
            if len(splits) == 3:
                hour, minute, sec = splits
            else:
                sec = 0
                hour, minute = splits
            return int(hour) * 3600 + int(minute) * 60 + int(sec)
        else:
            return decode_time(time_str)
    else:
        return decode_time(time_str)


def convert_sec_to_hh_mm_ss(seconds):
    """
    :param seconds: time in seconds, assuming that 12:00AM as 0 seconds
    :return: convert the seconds to the format hh:mm:ss
    """
    if seconds >= 0:
        _min, _sec = divmod(seconds, 60)
        _hour, _min = divmod(_min, 60)
        return "%d:%02d:%02d" % (_hour, _min, _sec)
    else:
        return "-" + convert_sec_to_hh_mm_ss(abs(seconds))


def measure_time(func):
    from datetime import datetime
    from functools import wraps

    @wraps(func)
    def compute_time_taken(*args, **kwargs):
        start = datetime.now()
        res = func(*args, **kwargs)
        end = datetime.now()
        time_taken = (end - start).total_seconds()
        if hasattr(args[0], 'store_compute_times') and hasattr(args[0], 'compute_times'):
            if args[0].store_compute_times:
                args[0].compute_times.append((func.__name__, time_taken))
        if isinstance(res, dict):
            res["compute_time"] = time_taken
        return res

    return compute_time_taken


def time_unit_conversion(input_unit, output_unit):
    from common.types import TimeUnits
    seconds_multipliers = {
        TimeUnits.Day.value: 86400,
        TimeUnits.Hour.value: 3600,
        TimeUnits.HectoSecond.value: 100,
        TimeUnits.Minute.value: 60,
        TimeUnits.DecaSecond.value: 10,
        TimeUnits.Second.value: 1
    }
    return seconds_multipliers[input_unit] / seconds_multipliers[output_unit]


def generate_experience(
        df,
        request_arrival_rate=28,
        hour_multiplier=3600,
        average_time_diff=7200,
        hours=40000,
        agency='MTD',
        date='2024-01-01',
        chains=1
):
    for k, seed in enumerate(range(chains)):
        generate_experience_for_seed(
            df,
            request_arrival_rate=request_arrival_rate,
            hour_multiplier=hour_multiplier,
            average_time_diff=average_time_diff,
            hours=hours,
            agency=agency,
            date=date,
            k=k,
            seed=seed
        )


def generate_experience_for_seed(
        df,
        request_arrival_rate=28,
        hour_multiplier=3600,
        average_time_diff=7200,
        hours=40000,
        agency='MTD',
        date='2024-01-01',
        k=0,
        seed=0
):
    import os
    import copy
    import random
    import numpy as np
    import pandas as pd

    columns = [
        "sample_date",
        "am_count",
        "wc_count",
        "origin_lat",
        "origin_lon",
        "destination_lat",
        "destination_lon",
        "arrival_time",
        "scheduled_pickup"
    ]
    output_key = "base"
    os.makedirs(f'../data/{agency}/{output_key}/', exist_ok=True)
    np.random.seed(seed)
    dfs = []
    df_sub = copy.deepcopy(
        df.sample(n=request_arrival_rate * hours, random_state=seed * 50000 + k * 5000000, replace=True)
    )
    arrival_times = []
    time_diffs = []
    for h in range(hours):
        arrival_time_raw = np.random.exponential(1 / request_arrival_rate, size=request_arrival_rate)
        arrival_time = arrival_time_raw.cumsum()
        arrival_time = (arrival_time - arrival_time.min()) * 3600 / (arrival_time.max() - arrival_time.min())
        arrival_time = (h + 1) * hour_multiplier + arrival_time
        arrival_time = arrival_time.tolist()
        time_diff = np.random.exponential(1 / request_arrival_rate, size=request_arrival_rate)
        time_diff = time_diff.cumsum()
        time_diff = (time_diff - time_diff.min()) * average_time_diff / (time_diff.max() - time_diff.min())
        arrival_times.extend(arrival_time)
        time_diffs.extend(time_diff)

    random.shuffle(time_diffs)
    df_sub['arrival_time'] = arrival_times
    df_sub["time_diff"] = time_diffs
    df_sub['scheduled_pickup'] = df_sub["arrival_time"] + df_sub["time_diff"]
    dfs.append(df_sub)

    df = pd.concat(dfs, ignore_index=True)
    df["sample_date"] = date
    os.makedirs(f'../data/{agency}/chains/', exist_ok=True)
    os.makedirs(f'../data/{agency}/chains/{k}/', exist_ok=True)

    df['arrival_time'] = df['arrival_time'].astype(int)
    df['scheduled_pickup'] = df['scheduled_pickup'].astype(int)
    df = df[columns]
    df.to_csv(f'../data/{agency}/chains/{k}/requests.csv', index=False)

    df_dates = pd.DataFrame()
    df_dates["sample_date"] = df['sample_date'].unique()
    df_dates["sample_idx"] = [i for i in range(len(df_dates))]
    df_dates = df_dates[["sample_idx", "sample_date"]]
    df_dates.to_csv(f'../data/{agency}/chains/{k}/dates.csv', index=False)
