"""
Enums para la máquina de estados.
"""

from enum import Enum, auto


class HUState(Enum):
    """
    Estados posibles de un Handling Unit (HU).
    
    Transiciones válidas:
        UNAVAILABLE -> PASSIVE  (cuando ORDER_START incluye este HU)
        PASSIVE -> ACTIVE       (cuando ITEM_START apunta a este HU)
        ACTIVE -> PASSIVE       (cuando ITEM_END y hay más items en otros HUs)
        ACTIVE -> ACTIVE        (cuando ITEM_START del mismo HU, nuevo item)
        PASSIVE -> UNAVAILABLE  (cuando ORDER_END)
        ACTIVE -> UNAVAILABLE   (cuando ORDER_END)
    """
    UNAVAILABLE = "UNAVAILABLE"  # HU no pertenece a la orden actual
    PASSIVE = "PASSIVE"          # HU pertenece a la orden pero no está activo
    ACTIVE = "ACTIVE"            # HU actualmente recibiendo items


class OrderState(Enum):
    """Estados posibles de una orden."""
    IDLE = "IDLE"                # Sin orden activa
    IN_PROGRESS = "IN_PROGRESS"  # Orden en proceso
    PAUSED = "PAUSED"            # Orden pausada
    COMPLETED = "COMPLETED"      # Orden completada


class EventType(Enum):
    """Tipos de eventos que puede procesar la máquina de estados."""
    ORDER_START = "ORDER_START"
    ORDER_END = "ORDER_END"
    ITEM_START = "ITEM_START"
    ITEM_END = "ITEM_END"
    ORDER_PAUSE = "ORDER_PAUSE"
    ORDER_RESUME = "ORDER_RESUME"

