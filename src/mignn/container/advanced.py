from .base import LightGraphContainer
from mignn.entities.graph import LightGraph
from mignn.entities.node import RayNode
from mignn.entities.connection import RayConnection

import mitsuba as mi
import numpy as np
import math
import random

class AdvancedLightGraphContainer(LightGraphContainer):
    
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
        
        sensor = scene.sensors()[0]
        sampler = sensor.sampler()
        bsdf_ctx = mi.BSDFContext()
        
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

                        # create Ray from current node
                        o, p = mi.Vector3f(node.position), mi.Vector3f(neighbor_selected_node.position)

                        # get direction and create new ray
                        d = p - o
                        normalized_d = d / np.sqrt(np.sum(d ** 2))
                        ray = mi.Ray3f(o, normalized_d)

                        # try intersect using this ray
                        si = scene.ray_intersect(ray)
                        
                        # if connections exists, then the node is also attached to the graph
                        # new connection is created between `node` and 
                        #  `neighbor_selected_node` with distance data
                        if si.is_valid() and si.t >= math.dist(p, o):

                            # TODO: check how to update throughput or necessary to remove it
                            # => depends on BRDF and hence directions...
                            bsdf = si.bsdf(ray)

                            # TODO: expected another `wo` which is from the previous node
                            # get connections where node is the target one and select one randomlyus
                            # => use of eval_pdf (same as integrator and update throuput using previous one and BSDF)
                            wo = si.to_local(ray.d)
                            node_index = graph.get_node_index(node)
                            
                            
      
                            return

                            # add connection into current graph
                            self._n_built_nodes += graph.add_node(neighbor_selected_node)
                            # self._n_built_connections += graph.add_connection(node, neighbor_selected_node, si.t)

                            # add connection into current graph
                            # self._n_built_nodes += selected_graph.add_node(node)
                            #self._n_built_connections += selected_graph.add_connection(neighbor_selected_node, node, si.t)

            return True
            
        return False
        
    @classmethod
    def _extract_light_grath(cls, line, coord_reverse):

        data = line.replace('\n', '').split(';')

        # get origin
        sample_pos = list(map(int, map(float, data[0].split(','))))
        origin = list(map(float, data[1].split(',')))
        # simulated camera normal (direction vector)
        normal = list(map(float, data[2].split(',')))

        # get luminance
        obtained_luminance = list(map(float, data[-1].split(',')))

        # prepare new graph
        graph = LightGraph(origin, obtained_luminance)

        # default origin node
        prev_node = RayNode(origin, normal)

        graph.add_node(prev_node)

        del data[0:3]
        del data[-1]

        for _, node in enumerate(data):
            node_data = node.split('::')

            distance = float(node_data[0])
            
            # TODO: take into account bsdf
            # bsdf_weight = list(map(float, node_data[1].split(',')))
            point = list(map(float, node_data[2].split(',')))
            normal = list(map(float, node_data[3].split(',')))

            node = RayNode(point, normal)
            graph.add_node(node)
            
            # build connection (unilateral)
            connection = RayConnection(prev_node, node, {'distance': distance})
            graph.add_connection(connection)

            prev_node = node
        
        return sample_pos, graph
 
    def __str__(self) -> str:
        return f'SimpleLightGraphContainer: {super().__str__()}'