from mignn.container.base import LightGraphContainer
from typing import List

class LightGraphManager():
    
    @staticmethod
    def fusion(light_graphs: List[LightGraphContainer]) -> LightGraphContainer:
                
        if len(light_graphs) < 2:
            return light_graphs[0]
        
        # quick from params init
        final_dict_graph = light_graphs[0].from_params(light_graphs[0])
        
        for g_dict in light_graphs:
            
            if not isinstance(g_dict, LightGraphContainer):
                continue
                
            if final_dict_graph.keys() == g_dict.keys():
                for k in final_dict_graph.keys():
                    final_dict_graph.add_graphs(k, g_dict.get_graphs(k))
                    
        return final_dict_graph
    
    @staticmethod
    def vstack(light_graph: LightGraphContainer) -> LightGraphContainer:
        pass
