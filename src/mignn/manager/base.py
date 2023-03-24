from mignn.container.base import LightGraphContainer
from typing import List

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
                
            if final_dict_graph.keys() == g_container.keys():
                for k in final_dict_graph.keys():
                    final_dict_graph.add_graphs(k, g_container.graphs_at(k))
                
                # TODO: improve this part
                final_dict_graph._n_built_nodes += g_container._n_built_nodes
                final_dict_graph._n_built_connections += g_container._n_built_connections
                    
        return final_dict_graph
    
    @staticmethod
    def vstack(light_graph: LightGraphContainer) -> LightGraphContainer:
        pass
