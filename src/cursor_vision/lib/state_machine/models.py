"""
Modelos de datos para la máquina de estados.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class ItemContext:
    """Contexto de un item siendo cargado."""
    sku: str
    description: str = ""
    quantity_expected: int = 0
    quantity_loaded: int = 0
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    
    @property
    def is_complete(self) -> bool:
        return self.ended_at is not None
    
    @property
    def duration(self) -> Optional[float]:
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sku": self.sku,
            "description": self.description,
            "quantity_expected": self.quantity_expected,
            "quantity_loaded": self.quantity_loaded,
            "is_complete": self.is_complete,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration": self.duration
        }


@dataclass
class HUPlan:
    """Plan de items esperados para un HU."""
    hu_id: str
    items: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def total_items(self) -> int:
        return len(self.items)
    
    @property
    def total_quantity(self) -> int:
        return sum(item.get('quantity', 0) for item in self.items)


@dataclass
class OrderPlan:
    """Plan completo de una orden."""
    order_id: str
    description: str = ""
    hus: List[HUPlan] = field(default_factory=list)
    
    @property
    def total_hus(self) -> int:
        return len(self.hus)
    
    def get_hu_plan(self, hu_id: str) -> Optional[HUPlan]:
        for hu in self.hus:
            if hu.hu_id == hu_id:
                return hu
        return None
    
    def get_hu_ids(self) -> List[str]:
        return [hu.hu_id for hu in self.hus]

