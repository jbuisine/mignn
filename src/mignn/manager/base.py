from mignn.container.base import LightGraphContainer
from typing import List

from mignn.entities.graph import LightGraph
from mignn.entities.connection import RayConnection


import random
import numpy as np

class LightGraphManager():

    @staticmethod
    def fusion(light_graphs: List[LightGraphContainer]) -> LightGraphContainer:

        if len(light_graphs) < 2:
            return light_graphs[0]

        # quick from params init
        # TODO: improve this part
        final_dict_graph = light_graphs[0].from_params(light_graphs[0])

        for g_container in light_graphs:

            if not isinstance(g_container, LightGraphContainer):
                continue

            # if final_dict_graph.keys() == g_container.keys():
            
            # for every keys (add new key or add graphs to existing one)
            for k in g_container.keys():
                final_dict_graph.add_graphs(k, g_container.graphs_at(k))

            # TODO: improve this part
            final_dict_graph._n_built_nodes += g_container._n_built_nodes
            final_dict_graph._n_built_connections += g_container._n_built_connections

            # remove from memory managed graph
            light_graphs.remove(g_container)

        return final_dict_graph

    @staticmethod
    def vstack(light_graph: LightGraphContainer, verbose: bool=True) -> LightGraphContainer:
        """From a light graph container, stack for each key all graphs into one

        Args:
            light_graph (LightGraphContainer): light graph container to stack

        Returns:
            LightGraphContainer: light graph container with only one graph per key
        """

        final_graph = light_graph.from_params(light_graph)

        n_elements = len(light_graph.keys())
        step = n_elements // 100

        # for each key stack all associated graph
        for idx, (key, graphs) in enumerate(light_graph.items()):

            # track all graphs data
            origin_list = []
            targets_list = []
            nodes_list = []
            connections_list = []

            # keep only one origin point
            origin_nodes = []

            for graph in graphs:

                origin_nodes.append(graph.get_node_by_index(0))

                origin_list.append(graph.origin)
                targets_list.append(graph.targets)
                nodes_list += graph.nodes
                connections_list += graph.connections

            # origin is from mean of graph origins
            origin = np.mean(origin_list, axis=0)
            targets = np.mean(targets_list, axis=0)

            current_graph = LightGraph(origin=origin, targets=targets)

            # randomly choose an origin node (only one from graphs)
            random_origin_node = random.choice(origin_nodes)
            current_graph.add_node(random_origin_node)

            # add nodes and connections into new graph
            for node in nodes_list:

                if not node in origin_nodes:
                    current_graph.add_node(node)

            for connection in connections_list:

                # check connection and connect it from one origin node only
                c_connection = None

                # do not use this kind of connection
                if connection.from_node in origin_nodes and connection.to_node in origin_nodes:
                    continue

                if connection.from_node in origin_nodes:
                    c_connection = RayConnection(random_origin_node, connection.to_node, \
                        connection.data, connection.tag)
                elif connection.to_node in origin_nodes:
                    c_connection = RayConnection(connection.from_node, random_origin_node, \
                        connection.data, connection.tag)
                else:
                    c_connection = connection

                current_graph.add_connection(c_connection)

            # add final graph to expected key
            final_graph.add_graph(key, current_graph)

            if verbose and (idx % step == 0 or idx >= n_elements - 1):
                print(f'[Stack] -- progress: {(idx + 1) / n_elements * 100.:.2f}%', \
                        end='\r' if idx + 1 < n_elements else '\n')

        return final_graph
