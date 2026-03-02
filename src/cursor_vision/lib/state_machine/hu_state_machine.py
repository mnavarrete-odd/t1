"""
HU State Machine - Máquina de estados para un Handling Unit individual.

Cada HU tiene su propia instancia de esta clase que mantiene:
- Estado actual (UNAVAILABLE, PASSIVE, ACTIVE)
- Item actualmente en carga (si hay)
- Historial de items cargados
- Timestamps de transiciones
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
import logging

from .enums import HUState, EventType
from .models import ItemContext


logger = logging.getLogger(__name__)


@dataclass
class StateTransition:
    """Registro de una transición de estado."""
    from_state: HUState
    to_state: HUState
    trigger: EventType
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


class HUStateMachine:
    """
    Máquina de estados para un Handling Unit (HU) individual.
    
    Mantiene el estado del HU y el tracking de items cargados.
    
    Estados:
        UNAVAILABLE: HU no pertenece a ninguna orden activa
        PASSIVE: HU pertenece a la orden pero no está recibiendo items
        ACTIVE: HU está activamente recibiendo un item
    
    Ejemplo de uso:
        hu = HUStateMachine("HU-001")
        hu.activate("SKU-123", "Producto X", 5)  # ITEM_START
        hu.complete_item(5)                       # ITEM_END
        hu.deactivate()                           # Vuelve a PASSIVE
    """
    
    def __init__(
        self,
        hu_id: str,
        initial_state: HUState = HUState.UNAVAILABLE,
        on_state_change: Optional[Callable[['HUStateMachine', HUState, HUState], None]] = None
    ):
        """
        Inicializa la máquina de estados del HU.
        
        Args:
            hu_id: Identificador único del HU
            initial_state: Estado inicial (default: UNAVAILABLE)
            on_state_change: Callback cuando cambia el estado
        """
        self.hu_id = hu_id
        self._state = initial_state
        self._on_state_change = on_state_change
        
        # Item actualmente en carga
        self._current_item: Optional[ItemContext] = None
        
        # Historial de items cargados en esta sesión
        self._items_loaded: List[ItemContext] = []
        
        # Historial de transiciones
        self._transitions: List[StateTransition] = []
        
        # Timestamps
        self._activated_at: Optional[datetime] = None
        self._last_activity: datetime = datetime.now()
    
    # ─────────────────────────────────────────────────────────────────
    # Propiedades
    # ─────────────────────────────────────────────────────────────────
    
    @property
    def state(self) -> HUState:
        """Estado actual del HU."""
        return self._state
    
    @property
    def is_active(self) -> bool:
        """True si el HU está activo (recibiendo items)."""
        return self._state == HUState.ACTIVE
    
    @property
    def is_passive(self) -> bool:
        """True si el HU está pasivo (en orden pero sin actividad)."""
        return self._state == HUState.PASSIVE
    
    @property
    def is_unavailable(self) -> bool:
        """True si el HU no está disponible."""
        return self._state == HUState.UNAVAILABLE
    
    @property
    def is_in_order(self) -> bool:
        """True si el HU pertenece a una orden (ACTIVE o PASSIVE)."""
        return self._state in (HUState.ACTIVE, HUState.PASSIVE)
    
    @property
    def has_open_item(self) -> bool:
        """True si hay un item actualmente en carga."""
        return self._current_item is not None and not self._current_item.is_complete
    
    @property
    def current_item(self) -> Optional[ItemContext]:
        """Item actualmente en carga (None si no hay)."""
        return self._current_item
    
    @property
    def current_sku(self) -> Optional[str]:
        """SKU del item actual (None si no hay item en carga)."""
        return self._current_item.sku if self._current_item else None
    
    @property
    def current_quantity(self) -> int:
        """Cantidad esperada del item actual (0 si no hay item)."""
        return self._current_item.quantity_expected if self._current_item else 0
    
    @property
    def items_loaded(self) -> List[ItemContext]:
        """Lista de items cargados en esta sesión."""
        return self._items_loaded.copy()
    
    @property
    def items_loaded_count(self) -> int:
        """Número de items cargados."""
        return len(self._items_loaded)
    
    @property
    def total_quantity_loaded(self) -> int:
        """Cantidad total de unidades cargadas."""
        return sum(item.quantity_loaded for item in self._items_loaded)
    
    @property
    def transitions(self) -> List[StateTransition]:
        """Historial de transiciones de estado."""
        return self._transitions.copy()
    
    # ─────────────────────────────────────────────────────────────────
    # Transiciones de Estado
    # ─────────────────────────────────────────────────────────────────
    
    def _set_state(self, new_state: HUState, trigger: EventType, context: Dict[str, Any] = None):
        """
        Cambia el estado interno y registra la transición.
        
        Args:
            new_state: Nuevo estado
            trigger: Evento que causó la transición
            context: Contexto adicional
        """
        old_state = self._state
        
        if old_state == new_state:
            return  # No hay cambio
        
        # Registrar transición
        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            trigger=trigger,
            timestamp=datetime.now(),
            context=context or {}
        )
        self._transitions.append(transition)
        
        # Actualizar estado
        self._state = new_state
        self._last_activity = datetime.now()
        
        # Callback
        if self._on_state_change:
            self._on_state_change(self, old_state, new_state)
    
    def join_order(self) -> bool:
        """
        Añade el HU a una orden (UNAVAILABLE -> PASSIVE).
        
        Se llama cuando ORDER_START incluye este HU.
        
        Returns:
            True si la transición fue exitosa
        """
        if self._state != HUState.UNAVAILABLE:
            return False
        
        self._set_state(HUState.PASSIVE, EventType.ORDER_START)
        self._items_loaded.clear()  # Limpiar historial de sesión anterior
        self._activated_at = None
        return True
    
    def activate(
        self,
        sku: str,
        description: str = "",
        quantity: int = 0,
        item_index: int = 0
    ) -> bool:
        """
        Activa el HU para recibir un item (PASSIVE/ACTIVE -> ACTIVE).
        
        Se llama cuando ITEM_START apunta a este HU.
        
        Args:
            sku: SKU del producto
            description: Descripción del producto
            quantity: Cantidad esperada
            item_index: Índice del item en la lista
        
        Returns:
            True si la transición fue exitosa
        """
        if self._state == HUState.UNAVAILABLE:
            return False
        
        # Si había un item previo sin cerrar, cerrarlo
        if self._current_item and not self._current_item.is_complete:
            self._current_item.ended_at = datetime.now()
            self._items_loaded.append(self._current_item)
        
        # Crear nuevo contexto de item
        self._current_item = ItemContext(
            sku=sku,
            description=description,
            quantity_expected=quantity,
            started_at=datetime.now()
        )
        
        # Transición a ACTIVE
        was_passive = self._state == HUState.PASSIVE
        self._set_state(
            HUState.ACTIVE,
            EventType.ITEM_START,
            {"sku": sku, "quantity": quantity, "item_index": item_index}
        )
        
        if was_passive:
            self._activated_at = datetime.now()
        
        return True
    
    def complete_item(self, quantity_loaded: int = None) -> bool:
        """
        Completa el item actual (permanece en ACTIVE).
        
        Se llama cuando ITEM_END para este HU.
        
        Args:
            quantity_loaded: Cantidad realmente cargada (default: la esperada)
        
        Returns:
            True si había un item que completar
        """
        if not self._current_item:
            return False
        
        # Completar item
        self._current_item.quantity_loaded = (
            quantity_loaded if quantity_loaded is not None 
            else self._current_item.quantity_expected
        )
        self._current_item.ended_at = datetime.now()
        
        # Agregar a historial
        self._items_loaded.append(self._current_item)
        
        # Limpiar item actual (pero seguimos ACTIVE hasta que otro HU tome el control)
        completed_item = self._current_item
        self._current_item = None
        self._last_activity = datetime.now()
        
        return True
    
    def deactivate(self) -> bool:
        """
        Desactiva el HU (ACTIVE -> PASSIVE).
        
        Se llama cuando otro HU se activa o manualmente.
        
        Returns:
            True si la transición fue exitosa
        """
        if self._state != HUState.ACTIVE:
            return False  # Ya no está activo
        
        # Si hay item abierto, cerrarlo
        if self._current_item and not self._current_item.is_complete:
            self._current_item.ended_at = datetime.now()
            self._items_loaded.append(self._current_item)
            self._current_item = None
        
        self._set_state(HUState.PASSIVE, EventType.ITEM_START)  # Otro HU tomó control
        return True
    
    def leave_order(self) -> bool:
        """
        Remueve el HU de la orden (PASSIVE/ACTIVE -> UNAVAILABLE).
        
        Se llama cuando ORDER_END.
        
        Returns:
            True si la transición fue exitosa
        """
        if self._state == HUState.UNAVAILABLE:
            return False  # Ya está fuera
        
        # Cerrar item si está abierto
        if self._current_item and not self._current_item.is_complete:
            self._current_item.ended_at = datetime.now()
            self._items_loaded.append(self._current_item)
            self._current_item = None
        
        self._set_state(HUState.UNAVAILABLE, EventType.ORDER_END)
        return True
    
    def reset(self):
        """Resetea el HU a su estado inicial."""
        self._state = HUState.UNAVAILABLE
        self._current_item = None
        self._items_loaded.clear()
        self._transitions.clear()
        self._activated_at = None
        self._last_activity = datetime.now()
    
    # ─────────────────────────────────────────────────────────────────
    # Serialización
    # ─────────────────────────────────────────────────────────────────
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa el estado del HU a diccionario."""
        return {
            "hu_id": self.hu_id,
            "state": self._state.value,
            "is_active": self.is_active,
            "is_passive": self.is_passive,
            "is_in_order": self.is_in_order,
            "has_open_item": self.has_open_item,
            "current_item": self._current_item.to_dict() if self._current_item else None,
            "current_sku": self.current_sku,
            "current_quantity": self.current_quantity,
            "items_loaded_count": self.items_loaded_count,
            "total_quantity_loaded": self.total_quantity_loaded,
            "items_loaded": [item.to_dict() for item in self._items_loaded],
            "activated_at": self._activated_at.isoformat() if self._activated_at else None,
            "last_activity": self._last_activity.isoformat()
        }
    
    def __repr__(self) -> str:
        item_info = f", item={self.current_sku}" if self.current_sku else ""
        return f"HU({self.hu_id}, state={self._state.value}{item_info})"

