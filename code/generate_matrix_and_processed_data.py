import math
import os
from collections import defaultdict
from heapq import heappush, heappop

import networkx as nx
import numpy as np
import pandas as pd
from osmnx import nearest_nodes

from common.general import load_obj, dump_obj
from env.data.TimeMatrix import load_graph_by_bounding_box, get_nodes_and_edges

lat_adjust = 0.01  # configure values greater than 0.0 for maps with extended road-network
lon_adjust = 0.01  # configure values greater than 0.0 for maps with extended road-network
detour_ratio = 0.5
window = 1800
detour_minimum = 900
test_train_split_idx = 40
cities = ['MTD', 'NYC']
depot_location = {
    'MTD': [35.057, -85.268],
    'NYC':[40.7405826, -73.9893518]
}
for data_set_id in cities:
    os.makedirs(f"../data/{data_set_id}/matrix", exist_ok=True)
    os.makedirs(f"../data/{data_set_id}/networks", exist_ok=True)
    os.makedirs(f"../data/{data_set_id}/processed", exist_ok=True)

    df = pd.read_csv(f"../data/{data_set_id}/base/requests.csv")
    df_chain = pd.read_csv(f"../data/{data_set_id}/chains/0/requests.csv")
    df = pd.concat([df, df_chain], ignore_index=True)

    box_dimensions = {
        "north": max(df.origin_lat.max(), df.destination_lat.max()) + lat_adjust,
        "south": min(df.origin_lat.min(), df.destination_lat.min()) - lat_adjust,
        "east": max(df.origin_lon.max(), df.destination_lon.max()) + lon_adjust,
        "west": min(df.origin_lon.min(), df.destination_lon.min()) - lon_adjust,
    }

    graph = load_graph_by_bounding_box(box_dimensions, f"../data/{data_set_id}/matrix/graph")

    # save the travel time matrices
    pointer = 0
    node_map = {}
    for node_id, node_obj in graph.nodes.items():
        node_map[node_id] = pointer
        pointer += 1
    dump_obj(node_map, f"../data/{data_set_id}/matrix/node_map.pickle")

    travel_time_matrix = nx.floyd_warshall_numpy(graph, weight='travel_time')
    travel_time_matrix = travel_time_matrix.astype(np.int32)

    def four_travel_times(map_directory):
        # author: Matthew Zalesak mdz32@cornell.edu
        input_file_name = f'{map_directory}/edges.csv'
        output_file_name = f'{map_directory}/times.csv'

        print('\tBuilding internal network...')
        nodecount = 0
        edges = {}
        with open(input_file_name) as fin:
            for line in fin:
                origin, destination, length = map(int, line[:-1].split(','))
                nodecount = max(nodecount, origin, destination)
                origin -= 1
                destination -= 1
                edges[(origin, destination)] = length  # int(round(MULTIPLIER * length))

        network = defaultdict(list)
        for (origin, destination), cost in edges.items():
            network[origin].append((destination, cost))

        def dijkstras(i):
            q = [(0, i, ())]  # (distance, id, path)
            visited = set()
            bests = {i: 0}

            while q:
                (cost, o, path) = heappop(q)  # Closest node on the frontier.
                if o not in visited:  # For each source... (if unprocessed)
                    visited.add(o)  # Note this has been processed and then extend reverse path.
                    path = (o, path)  # Path does nothing here.

                    for d, cost_d in network.get(o, ()):  # For each destination
                        if d not in visited:  # If it might be on the frontier...
                            prev = bests.get(d, None)  # Try to get existing estimate of distance.
                            new = cost + cost_d  # Judge distance via waypoint o.
                            if prev is None or new < prev:  # If this is best path found so far...
                                bests[d] = new  # This is the best distance!
                                heappush(q, (new, d, path))  # Pop it on the queue!

            return bests

        # Instead of this old way, will do more space efficient version!
        # results = 10000 * np.ones(dtype = int, shape = (nodecount, nodecount))

        print('\tComputing all pairs shortest distances and writing results to file...')
        print('\tThere are', nodecount, 'rows to process.')
        with open(output_file_name, 'w') as fout:
            for origin in range(nodecount):
                results = 10000 * np.ones(dtype=int, shape=(nodecount,))
                if origin % 100 == 0:
                    print('\t\tWorking on line', origin)
                paths = dijkstras(origin)
                for destination, cost in paths.items():
                    results[destination] = cost
                fout.write(','.join(map(str, results)))
                fout.write('\n')


    matrix_dir = f"../data/{data_set_id}/networks/"

    nodes, edges = get_nodes_and_edges(graph)
    nodes['node_id'] = nodes['node_id'].apply(lambda x: x + 1)
    node_to_osmnx_map = {node.osmid: int(node.node_id) for i, node in nodes.iterrows()}
    nodes = nodes[['node_id', 'lat', 'lon']]
    file_path = f"{matrix_dir}/nodes.csv"
    nodes.to_csv(file_path, header=False, index=False)

    edges['source_node'] = edges['source_node'].apply(lambda x: x + 1)
    edges['target_node'] = edges['target_node'].apply(lambda x: x + 1)
    edges['travel_time'] = edges['travel_time'].apply(lambda x: math.ceil(x))
    edges = edges[['source_node', 'target_node', 'travel_time']]
    file_path = f"{matrix_dir}/edges.csv"
    edges.to_csv(file_path, header=False, index=False)
    four_travel_times(matrix_dir)

    locations = [(round(depot_location[data_set_id][0], 6), round(depot_location[data_set_id][1], 6))]
    for i, entry in df.iterrows():
        locations.append((round(entry['origin_lat'], 6), round(entry['origin_lon'], 6)))
        locations.append((round(entry['destination_lat'], 6), round(entry['destination_lon'], 6)))

    locations = list(set(locations))

    dump_obj(locations, f"{matrix_dir}/locations")

    locations = load_obj(f"{matrix_dir}/locations")
    X = [lon_i for (lat_i, lon_i) in locations]
    Y = [lat_i for (lat_i, lon_i) in locations]
    nearest_nodes_dict = nearest_nodes(graph, X, Y, return_dist=False)
    updated_nearest_nodes_dict = [node_to_osmnx_map[osmid] for osmid in nearest_nodes_dict]

    dump_obj(updated_nearest_nodes_dict, f"{matrix_dir}/nearest_nodes_spl")

    node_map = load_obj(f'../data/{data_set_id}/matrix/node_map.pickle')
    keys = []
    values = []
    dicts = []
    for key, value in node_map.items():
        dicts.append({
            'osm_id': key,
            'ttm_id': value
        })
    df_node_map = pd.DataFrame(dicts)
    df_node_map.to_csv(f"../data/{data_set_id}/matrix/node_map.csv", index=False)

    with open(f'../data/{data_set_id}/matrix/time_matrix.csv', 'wb') as f:
        np.savetxt(f, np.ceil(travel_time_matrix).astype(np.int32), delimiter=',', fmt='%i')

    df = pd.read_csv(f"../data/{data_set_id}/base/requests.csv")
    sample_dates_dict = {sample_date: idx for idx, sample_date in enumerate(list(df.sample_date.unique()))}
    df['chain_id'] = df.apply(lambda x: sample_dates_dict[x.sample_date], axis=1)
    chains = []
    for chain_id in df.chain_id.unique():
        df_single_chain = df[df.chain_id == chain_id]
        df_single_chain = df_single_chain.sort_values(by=["arrival_time"])
        df_single_chain['chain_order'] = [i for i in range(len(df_single_chain))]
        chains.append(df_single_chain)
    df = pd.concat(chains, ignore_index=True)
    df = df.sort_values(by=["chain_id", "arrival_time"])
    origin_lats = df.origin_lat.to_list()
    origin_lons = df.origin_lon.to_list()
    destination_lats = df.destination_lat.to_list()
    destination_lons = df.destination_lon.to_list()

    origin_node_ids = nearest_nodes(graph, origin_lons, origin_lats)
    origin_node_reverse_ids = [node_map[origin_node_id] for origin_node_id in origin_node_ids]
    dest_node_ids = nearest_nodes(graph, destination_lons, destination_lats)
    dest_node_reverse_ids = [node_map[dest_node_id] for dest_node_id in dest_node_ids]
    df['pickup_node_id'] = origin_node_reverse_ids
    df['dropoff_node_id'] = dest_node_reverse_ids
    df['pickup_time_since_midnight'] = df['scheduled_pickup']
    df['direct_time'] = df.apply(
        lambda row: math.ceil(travel_time_matrix[row.pickup_node_id][row.dropoff_node_id]),
        axis=1
    )
    df['pickup_end_time_since_midnight'] = df.apply(
        lambda row: int(row.pickup_time_since_midnight + window),
        axis=1
    )
    df['dropoff_start_time_since_midnight'] = df.apply(
        lambda row: int(row.pickup_time_since_midnight + row.direct_time),
        axis=1
    )
    df['dropoff_time_since_midnight'] = df.apply(
        lambda row: int(
            row.pickup_time_since_midnight + row.direct_time + window +
            min(row.direct_time * detour_ratio, detour_minimum)
        ),
        axis=1
    )
    df['request_arrive_time'] = df['arrival_time']
    df_entries = []
    for chain_id in df.chain_id.unique():
        df_sub = df[df.chain_id == chain_id]
        df_sub = df_sub.sort_values(by='request_arrive_time')
        df_sub['request_next_arrive_time'] = df_sub['arrival_time'].to_list()[1:] + [df_sub.iloc[-1]['pickup_time_since_midnight']]
        df_sub['search_time'] = df_sub.apply(lambda row: row.request_next_arrive_time - row.request_arrive_time, axis=1)
        df_entries.append(df_sub)

    df = pd.concat(df_entries, ignore_index=True)

    selected_columns = [
        'chain_id', 'chain_order', 'pickup_node_id', 'dropoff_node_id', 'pickup_time_since_midnight',
        'pickup_end_time_since_midnight', 'dropoff_start_time_since_midnight',
        'dropoff_time_since_midnight', 'am_count', 'wc_count', 'direct_time', 'request_arrive_time', 'search_time'
    ]

    df = df[selected_columns]
    df_test = df[df.chain_id >= test_train_split_idx]
    df_train = df[df.chain_id < test_train_split_idx]
    df_test.to_csv(f"../data/{data_set_id}/processed/test.csv", index=False)
    df_train.to_csv(f"../data/{data_set_id}/processed/train.csv", index=False)
