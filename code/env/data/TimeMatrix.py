import copy
import math

from common.Singleton import Singleton
from common.general import directory_exists
from common.types import TravelTimeSources
from env.data.Location import LOCATION_PRECISION
from env.structures.MappedList import MappedList

def load_graph_by_bounding_box(box_dimensions, file_name):
    """
    :param box_dimensions: name of the location
    :param file_name: file name of osmnx graph (name of the existing file to load, or new file name to create one)
    :return: graph
    """

    import osmnx as ox
    from osmnx import settings

    from common.general import file_exists

    settings.log_console = True
    if not file_exists(file_name, ".graphml"):
        # this is processed locations
        graph = ox.graph_from_bbox(
            north=box_dimensions["north"],
            east=box_dimensions["east"],
            south=box_dimensions["south"],
            west=box_dimensions["west"],
            network_type='drive',
            simplify=True,
            truncate_by_edge=True,
            retain_all=False,
        )
        graph = configure_graph(graph)
        ox.save_graphml(graph, filepath=f'{file_name}.graphml')
        return graph
    else:
        graph = ox.load_graphml(f"{file_name}.graphml")
        graph = configure_graph(graph)
        return graph


def configure_graph(graph):
    from osmnx import settings
    from osmnx import utils_graph
    from osmnx import speed

    settings.log_console = False
    hwy_speeds = {
        'motorway': 90, 'motorway_link': 45, 'trunk': 85,
        'trunk_link': 40, 'primary': 65, 'primary_link': 30,
        'secondary': 55, 'secondary_link': 25, 'tertiary': 40,
        'tertiary_link': 20, 'unclassified': 25, 'residential': 25,
        'living_street': 10, 'service': 15
    }

    fallback = 10
    # to get largest strongly connected graph
    graph = utils_graph.get_largest_component(graph, strongly=True)
    graph = speed.add_edge_speeds(graph, hwy_speeds=hwy_speeds, fallback=fallback)
    return speed.add_edge_travel_times(graph)


def get_nodes_and_edges(graph):
    import osmnx as ox
    nodes, edges = ox.utils_graph.graph_to_gdfs(graph)

    # format nodes
    nodes['osmid'] = nodes.index
    nodes.index = range(len(nodes))
    nodes['node_id'] = nodes.index
    nodes['lon'] = nodes['x']
    nodes['lat'] = nodes['y']
    nodes = nodes[['node_id', 'osmid', 'lat', 'lon']]
    nodes['node_id'] = nodes['node_id'].astype(int)
    nodes['osmid'] = nodes['osmid'].astype(int)
    nodes['lat'] = nodes['lat'].astype(float)
    nodes['lon'] = nodes['lon'].astype(float)

    # format edges
    edges = edges.reset_index()
    edges['source_osmid'] = edges['u']
    edges['target_osmid'] = edges['v']
    edges['source_node'] = edges['source_osmid'].apply(
        lambda x: nodes.loc[nodes['osmid'] == x, 'node_id'].values[0])
    edges['target_node'] = edges['target_osmid'].apply(
        lambda x: nodes.loc[nodes['osmid'] == x, 'node_id'].values[0])
    edges = edges.sort_values(by=['travel_time'])
    edges = edges.drop_duplicates(subset=['source_node', 'target_node'])
    edges = edges[['source_osmid', 'target_osmid', 'source_node', 'target_node', 'travel_time']]
    edges['source_osmid'] = edges['source_osmid'].astype(int)
    edges['target_osmid'] = edges['target_osmid'].astype(int)
    edges['source_node'] = edges['source_node'].astype(int)
    edges['target_node'] = edges['target_node'].astype(int)
    edges['travel_time'] = edges['travel_time'].astype(int)
    return nodes, edges


class TimeMatrix:
    """
    class to store time-matrix object and locations associated with time-matrix
    """

    def __init__(self, locations, init_matrix=True):
        """
        :param locations: set of locations
        :param init_matrix: initialize the matrix or not
        """
        from common.arg_parser import get_parsed_args
        args = get_parsed_args()
        self.locations = MappedList(locations)
        self.travel_time_source = args.travel_time_source
        self.dwell_time_pickup = args.dwell_time_pickup
        self.dwell_time_dropoff = args.dwell_time_dropoff
        self.raw_matrix = []
        self.dt_matrix = []
        self.matrix = []
        if init_matrix:
            self.init_matrix()

    def init_matrix(self):
        """
        initialize the matrix
        """

        if self.dwell_time_pickup > 0 or self.dwell_time_dropoff > 0:
            self.dt_matrix = [[0 for _ in self.locations] for _ in self.locations]
            from env.data.Location import Location
            loc_i: Location
            loc_j: Location
            for i, loc_i in enumerate(self.locations):
                for j, loc_j in enumerate(self.locations):
                    if i != j:
                        self.dt_matrix[i][j] = self.get_dwell_time(loc_i, loc_j)
            self.raw_matrix = self.get_duration_matrix(self.locations)
            self.matrix = self.add_matrix(self.raw_matrix, self.dt_matrix)
        else:
            self.matrix = self.get_duration_matrix(self.locations)

    @staticmethod
    def get_duration_matrix(locations):
        """
        :param locations: list of locations
        :return: return time matrix
        """
        import math
        matrix = [[0 for _ in locations] for _ in locations]
        for i, loc_i in enumerate(locations):
            for j, loc_j in enumerate(locations):
                if i != j:
                    matrix[i][j] = math.ceil(loc_i.duration(loc_j))
        return matrix

    @staticmethod
    def add_matrix(raw_matrix, dt_matrix):
        # this will perform element wise addition
        import numpy as np
        raw_matrix = np.array(raw_matrix, dtype=np.int32)
        dt_matrix = np.array(dt_matrix, dtype=np.int32)
        return (raw_matrix + dt_matrix).tolist()

    @staticmethod
    def get_dwell_time(start_loc, end_loc):
        from env.data.Location import LocationTypes
        dwell_time = 0
        if end_loc.loc_type in [LocationTypes.PICKUP, LocationTypes.DROPOFF]:
            dwell_time += end_loc.dwell_time
        if start_loc.loc_type in [LocationTypes.PICKUP_MIDDLE, LocationTypes.DROPOFF_MIDDLE]:
            dwell_time -= start_loc.captured_dwell_time
        return dwell_time

    def __del__(self):
        self.reset()

    def reset(self):
        del self.locations
        del self.raw_matrix
        del self.dt_matrix
        del self.matrix
        self.locations = None
        self.raw_matrix = None
        self.dt_matrix = None
        self.matrix = None

    def copy(self, reuse_matrix=False):
        # locations never deep-copied
        copy_matrix = TimeMatrix(self.locations, init_matrix=reuse_matrix)
        if not reuse_matrix:
            copy_matrix.raw_matrix = copy.deepcopy(self.raw_matrix)
            copy_matrix.dt_matrix = copy.deepcopy(self.dt_matrix)
            copy_matrix.matrix = copy.deepcopy(self.matrix)
        return copy_matrix


class MatrixManager(metaclass=Singleton):
    _instance = None

    @classmethod
    def init_travel_time_source(cls, source, data_dir, data_id):
        from common.logger import logger
        from common.general import load_obj, np_load_matrix
        if hasattr(cls._instance, "source") and cls._instance.source == source:
            return
        cls._instance.source = source
        matrix_path = f"{data_dir}/{data_id}/matrix"
        networks_path = f"{data_dir}/{data_id}/networks"
        if cls._instance.source == TravelTimeSources.OSMnx.value:
            import osmnx as ox
            import numpy as np
            loaded_graph = ox.load_graphml(f"{matrix_path}/graph.graphml")
            cls._instance.graph = configure_graph(loaded_graph)
            cls._instance.node_map = load_obj(f"{matrix_path}/node_map")
            cls._instance.time_matrix = np_load_matrix(f"{matrix_path}/time_matrix.csv")
            cls._instance.dist_matrix = np_load_matrix(f"{matrix_path}/dist_matrix.csv")
            if directory_exists(networks_path):
                cls._instance.locations = load_obj(f"{networks_path}/locations")
                lats = [lat for lat, _ in cls._instance.locations]
                lons = [lon for _, lon in cls._instance.locations]
                min_lat, max_lat = min(lats), max(lats)
                min_lon, max_lon = min(lons), max(lons)
                cls._instance.min_lat = min_lat
                cls._instance.min_lon = min_lon
                cls._instance.diff_lat = max_lat - min_lat
                cls._instance.diff_lon = max_lon - min_lon
                cls._instance.nearest_nodes_special = load_obj(f"{networks_path}/nearest_nodes_spl")
            cls._instance.node_mapping = {}
            cls._instance.node_idx_mapping = {}
        elif cls._instance.source == TravelTimeSources.Euclidean.value:
            import json
            with open(f"{data_dir}/{data_id}/grid_info.json") as grid_info_file:
                json_grid_info = json.load(grid_info_file)
                grid_size = json_grid_info["grid_size"]
            cls._instance.min_lat = 0
            cls._instance.min_lon = 0
            cls._instance.diff_lat = grid_size
            cls._instance.diff_lon = grid_size

        logger.info("LOADED TRAVEL MATRICES")

    @classmethod
    def clear_node_mapping(cls):
        """
        clear the node mapping to avoid memory issues
        """
        if hasattr(cls._instance, "node_mapping"):
            del cls._instance.node_mapping
            cls._instance.node_mapping = {}
        if hasattr(cls._instance, "node_idx_mapping"):
            del cls._instance.node_idx_mapping
            cls._instance.node_idx_mapping = {}

    @classmethod
    def get_grid_coordinates(cls, raw_coordinates, grid_size=2):
        """
        :param raw_coordinates: raw latitude and longitude coordinates
        :param grid_size: size of the square grid

        return the x, y grid coordinates
        """
        import math
        lat = math.floor((raw_coordinates[0] - cls._instance.min_lat) * grid_size / cls._instance.diff_lat)
        lon = math.floor((raw_coordinates[1] - cls._instance.min_lon) * grid_size / cls._instance.diff_lon)

        # handling the corner cases
        if lat >= grid_size:
            lat = grid_size - 1
        if lon >= grid_size:
            lon = grid_size - 1
        return [lat, lon]

    @classmethod
    def set_nearest_node(cls, raw_coordinates):
        """
        :param raw_coordinates: raw latitude and longitude coordinates

        cache the nearest node for fast access
        """
        from osmnx.distance import nearest_nodes
        if raw_coordinates not in cls._instance.node_mapping:
            nearest_node = nearest_nodes(
                cls._instance.graph, [raw_coordinates[1]], [raw_coordinates[0]], return_dist=False
            )[0]
            cls._instance.node_mapping[raw_coordinates] = nearest_node
            cls._instance.node_idx_mapping[raw_coordinates] = cls._instance.node_map[nearest_node]

    @classmethod
    def get_node(cls, raw_coordinates):
        """
        :param raw_coordinates: raw latitude and longitude coordinates
        :return: nearest node
        """
        loc = round(raw_coordinates[0], LOCATION_PRECISION), round(raw_coordinates[1], LOCATION_PRECISION)
        if loc not in cls._instance.locations:
            raise ValueError(f"Location {loc} does not exists")
        else:
            if len(cls._instance.nearest_nodes_special) < cls._instance.locations.index(loc):
                raise ValueError(f"No nearest available for the location {loc}")
        return cls._instance.nearest_nodes_special[cls._instance.locations.index(loc)]

    @classmethod
    def get_nearest_node(cls, raw_coordinates):
        """
        :param raw_coordinates: raw latitude and longitude coordinates
        :return: nearest node
        """
        cls._instance.set_nearest_node(raw_coordinates)
        return cls._instance.node_mapping[raw_coordinates]

    @classmethod
    def get_nearest_node_idx(cls, raw_coordinates):
        """
        :param raw_coordinates: raw latitude and longitude coordinates
        :return: nearest node index
        """
        cls._instance.set_nearest_node(raw_coordinates)
        return cls._instance.node_idx_mapping[raw_coordinates]

    @classmethod
    def get_travel_time(cls, start_loc, end_loc, no_ceil=False):
        """
        :param start_loc: start location
        :param end_loc: end location
        :param no_ceil: whether to ceil the travel time or not
        :return: obtain the travel time
        """
        from env.data.Location import Location
        if not isinstance(start_loc, Location):
            start_loc = start_loc.loc
        if not isinstance(end_loc, Location):
            end_loc = end_loc.loc
        match cls._instance.source:
            case TravelTimeSources.OSMnx.value:
                start_idx = cls._instance.get_nearest_node_idx(start_loc.get_raw())
                end_idx = cls._instance.get_nearest_node_idx(end_loc.get_raw())
                duration = cls._instance.time_matrix[start_idx][end_idx]
            case TravelTimeSources.Euclidean.value:
                duration = start_loc.get_euclidean(end_loc)
            case _:  # default option
                raise ValueError(f"Invalid travel time source {cls._instance.source}")
        duration += start_loc.time_adjustment + end_loc.time_adjustment  # accommodate the time discrepancies
        if no_ceil:
            return duration
        return math.ceil(duration)

    @classmethod
    def get_travel_distance(cls, start_loc, end_loc):
        """
        :param start_loc: start location
        :param end_loc: end location
        :return: obtain the travel distance
        """
        meters_to_miles = 0.000621371
        from env.data.Location import Location
        if not isinstance(start_loc, Location):
            start_loc = start_loc.loc
        if not isinstance(end_loc, Location):
            end_loc = end_loc.loc
        match cls._instance.source:
            case TravelTimeSources.OSMnx.value:
                start_idx = cls._instance.get_nearest_node_idx(start_loc.get_raw())
                end_idx = cls._instance.get_nearest_node_idx(end_loc.get_raw())
                return cls._instance.dist_matrix[start_idx][end_idx] * meters_to_miles
            case TravelTimeSources.Euclidean.value:
                return start_loc.get_euclidean(end_loc)
            case _:
                raise ValueError(f"Invalid travel time source {cls._instance.source}")

    @classmethod
    def get_routes(cls, start_loc, end_loc):
        """
        :param start_loc: start location of the route
        :param end_loc: end location of the route
        :return: list of nodes that represents the shortest path based on the 'travel_time'
                 from the start location to the end location
        """
        match cls._instance.source:
            case TravelTimeSources.OSMnx.value:
                import networkx as nx
                start_node = cls._instance.get_nearest_node(start_loc.get_raw())
                end_node = cls._instance.get_nearest_node(end_loc.get_raw())
                path = nx.shortest_path(
                    cls._instance.graph, start_node, end_node, weight='travel_time'
                )
                locations = []
                durations = []
                for node in path:
                    locations.append([cls._instance.graph.nodes[node]['x'], cls._instance.graph.nodes[node]['y']])

                for i, node in enumerate(path[:-1]):
                    node_idx = cls._instance.node_map[node]
                    next_node_idx = cls._instance.node_map[path[i + 1]]
                    durations.append(cls._instance.time_matrix[node_idx][next_node_idx])

                return locations, durations

    @classmethod
    def load(cls, locations, init_matrix=True):
        """
        :param locations: list of locations
        :param init_matrix: whether to load initial matrix or not
        """

        cls._instance.init_matrix = init_matrix
        if cls._instance.init_matrix:
            cls._instance.main = TimeMatrix(locations)
            cls._instance.temp = None

    @classmethod
    def get_matrix(cls, temp=False):
        """
        :param temp: whether to return temporary matrix or not (like before insertion of incoming requests)
        :return: matrix that covers all the locations listed
        """
        if cls._instance.init_matrix:
            if temp:
                if cls._instance.temp:
                    return cls._instance.temp
                return cls._instance.main
            return cls._instance.main
        return None

    @classmethod
    def temporary_reconstruct(cls, locations=None, update_temp=True):
        """
        :param locations: updated locations
        :param update_temp: update temporary matrix object with original matrix object
        """
        if cls._instance.init_matrix:
            if update_temp:
                cls._instance.__update_temp()
            if locations is not None:
                from env.structures.MappedList import MappedList
                cls._instance.temp.locations = locations
            cls._instance.temp.init_matrix()

    @classmethod
    def update(cls):
        """
        update the main matrix object to match with temporary matrix object
        """
        if cls._instance.init_matrix:
            cls._instance.main = cls._instance.temp.copy(reuse_matrix=True)

    @classmethod
    def __update_temp(cls):
        """
        update the temporary matrix object to match with original matrix object
        """
        if cls._instance.init_matrix:
            cls._instance.temp = cls._instance.main.copy(reuse_matrix=True)

    @classmethod
    def reset_temp(cls):
        """
        reset the temporary matrix object
        """
        if cls._instance.init_matrix:
            del cls._instance.temp
            cls._instance.temp = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = MatrixManager()
        return cls._instance
