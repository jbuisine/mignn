"""Abstract Encoder"""
from abc import ABC, abstractmethod
from typing import List

from itertools import chain


class Encoder(ABC):
    """Generic Encoder class"""
    
    def encode(self, properties: List[float], \
        filter_pattern: List[bool]=None):
        """Encoder properties using or not a filter pattern

        Args:
            properties (List[float]): expected properties

        Raises:
            ValueError: filter_pattern and properties not identical in size

        Returns:
            _type_: encoded properties
        """
        
        if filter_pattern is None:
            return list(chain(*[ self._encode_property(p) for p in properties ]))
        else:
            if len(filter_pattern) != len(properties):
                raise ValueError('Expected a filter pattern and '\
                            'properties of identical size')
            else:
                return list(chain(*[ self._encode_property(p) if filter_pattern[p_i] \
                    else p for p_i, p in enumerate(properties)]))

    @abstractmethod
    def _encode_property(self, property: float) -> float:
        """Specific way to encode a property
        """ 
