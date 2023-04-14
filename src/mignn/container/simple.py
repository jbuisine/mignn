from .base import LightGraphContainer
from mignn.entities.graph import LightGraph
from mignn.entities.node import RayNode
from mignn.entities.connection import RayConnection

from mignn.entities.connection import RayConnection, ConnectionTag

import mitsuba as mi
import numpy as np
import math
import random

class SimpleLightGraphContainer(LightGraphContainer):
    
    def __init__(self, scene_file: str, reference: np.ndarray=None, \
        variant: str='scalar_rgb'):
        
        super().__init__(scene_file, reference, variant)
 
        
    def _build_pos_connections(self, scene, pos, n_graphs, n_nodes_per_graphs, n_neighbors):
        """
        For each position from current film, new connections are tempted to be build:
        - n_graphs: number of graphs to update
        - n_nodes_per_graphs: expected number of nodes to get new connections (chosen randomly)
        - n_neighbors: number of neighbors graph to take in account 
        """
        
        # use of mitsuba in order to build new connections    
        pos = tuple(pos)
        if pos in self._graphs:
            
            pos_graphs = self._graphs[pos]
            
            # for each graph, try to create new connection
            for graph in random.choices(pos_graphs, k=n_graphs):
                
                selected_nodes = random.choices(graph.nodes, k=n_nodes_per_graphs)
                potential_neighbors = [ g for g in pos_graphs if g is not graph ]
                
                # check if there is at least 1 potential neighbor
                if len(potential_neighbors) > 0:
                    neighbors_graphs = random.choices([ g for g in pos_graphs if g is not graph], \
                                            k=n_neighbors)

                    # try now to create connection
                    for node in selected_nodes:
                        
                        # select randomly one neighbor graph
                        selected_graph = random.choice(neighbors_graphs)

                        # randomly select current neighbor graph node for the connection
                        neighbor_selected_node = random.choice(selected_graph.nodes)

                        # do not continue the selected nodes are primary rays or origin
                        if node.primary and neighbor_selected_node.primary:
                            continue
                        
                        # create Ray from current node
                        origin, point = mi.Vector3f(node.position), mi.Vector3f(neighbor_selected_node.position)

                        # get direction and create new ray
                        direction = point - origin
                        normalized_d = direction / np.sqrt(np.sum(direction ** 2))
                        ray = mi.Ray3f(origin, normalized_d)

                        # try intersect using this ray
                        si = scene.ray_intersect(ray)
                        
                        # if connections exists, then the node is also attached to the graph
                        # new bi-directionnal connections are created between `node` and 
                        #  `neighbor_selected_node` with distance data
                        if si.is_valid() and si.t >= math.dist(point, origin):

                            # add connection into current graph (from -> to)
                            connection = RayConnection(node, neighbor_selected_node, \
                                {'distance': si.t}, ConnectionTag.BUILT)
                            self._n_built_nodes += graph.add_node(neighbor_selected_node)
                            self._n_built_connections += graph.add_connection(connection)

                            # add connection into neighbor graph (to -> from)
                            connection = RayConnection(neighbor_selected_node, node, \
                                {'distance': si.t}, ConnectionTag.BUILT)
                            self._n_built_nodes += selected_graph.add_node(node)
                            self._n_built_connections += selected_graph.add_connection(connection)
                            
            return True
            
        return False
        
    def _extract_light_grath(self, line, coord_reverse):

        data = line.replace('\n', '').split(';')

        # TODO: need to reverse sample pos (height / width), check if always required or specific to scene
        # get origin
        if coord_reverse:
            sample_pos = list(map(int, map(float, data[0].split(','))))[::-1]
        else:
            sample_pos = list(map(int, map(float, data[0].split(','))))
        
        position = list(map(float, data[1].split(',')))
        # simulated camera normal (direction vector)
        normal = list(map(float, data[2].split(',')))

        # get luminance
        c_luminance = list(map(float, data[-1].split(',')))

        # prepare new graph
        # TODO: use of reference image if exists
        if self._reference is not None:
            pos_x, pos_y = sample_pos
            target_luminance = list(np.array(self._reference[pos_x, pos_y, :]))
        else:
            raise ValueError('Expected reference image for this kind of loader!')
            
        graph = LightGraph(position, target_luminance)

        # default origin node
        # zero radiance by default
        prev_node = RayNode(position, normal, c_luminance, primary=True)

        graph.add_node(prev_node)

        del data[0:3]
        del data[-1]
        
        # cannot build connection (only one Node)
        # There is no graph
        for n_i, node in enumerate(data):
            node_data = node.split('::')

            distance = float(node_data[0])
            valid = bool(int(node_data[1]))
            has_next = bool(int(node_data[2]))
            
            # bsdf_weight = list(map(float, node_data[1].split(',')))
            position = list(map(float, node_data[4].split(',')))
            normal = list(map(float, node_data[5].split(',')))
            
            # only is there is next computation
            if has_next:
                radiance = list(map(float, node_data[7].split(',')))
            else:
                radiance = [0, 0, 0]

            # set RayNode as primary
            if n_i == 0:
                node = RayNode(position, normal, radiance, primary=True)
            else:
                node = RayNode(position, normal, radiance)
                
            graph.add_node(node)
            
            # build connection (unilateral)
            connection = RayConnection(prev_node, node, \
                {'distance': distance, 'valid': valid}, ConnectionTag.ORIGINAL)
            graph.add_connection(connection)

            prev_node = node
        
        return sample_pos, graph
 
    def __str__(self) -> str:
        return f'SimpleLightGraphContainer: {super().__str__()}'