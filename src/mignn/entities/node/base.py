"""Abstract Node"""
from abc import ABC, abstractmethod
from typing import List

class Node(ABC):
    """Abstract Node class
    """
    
    @property
    @abstractmethod
    def properties(self) -> List[float]:
        """Get all properties describing this node
        """     
