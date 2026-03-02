"""
Order State Machine - Orquestador de la máquina de estados.

Coordina múltiples HUs y procesa eventos del picker manteniendo
la coherencia del estado global de la orden.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
import logging

from .enums import HUState, OrderState, EventType
from .models import ItemContext, HUPlan, OrderPlan
from .hu_state_machine import HUStateMachine


logger = logging.getLogger(__name__)


@dataclass
class OrderSummary:
    """Resumen de una orden completada."""
    order_id: str
    total_hus: int
    total_items: int
    total_quantity: int
    duration_seconds: float
    hus_summary: List[Dict[str, Any]]
    started_at: datetime
    completed_at: datetime


class OrderStateMachine:
    """
    Máquina de estados que orquesta una orden de picking completa.
    
    Gestiona múltiples HUStateMachine y procesa eventos del sistema SAP
    para mantener el estado coherente de toda la orden.
    
    Uso típico:
        sm = OrderStateMachine()
        
        # Procesar eventos
        sm.process_event("ORDER_START", {...})
        sm.process_event("ITEM_START", {...})
        sm.process_event("ITEM_END", {...})
        ...
        sm.process_event("ORDER_END", {...})
        
        # Consultar estado
        print(sm.get_active_hu())
        print(sm.get_state())
    """
    
    MAX_HUS = 4  # Máximo de HUs por orden
    MAX_COMPLETED_ORDERS = 100  # Límite máximo de órdenes completadas en historial (prevención de OOM)
    
    def __init__(
        self,
        on_state_change: Optional[Callable[['OrderStateMachine'], None]] = None,
        on_hu_state_change: Optional[Callable[[HUStateMachine, HUState, HUState], None]] = None
    ):
        """
        Inicializa el orquestador.
        
        Args:
            on_state_change: Callback cuando cambia el estado de la orden
            on_hu_state_change: Callback cuando cambia el estado de un HU
        """
        self._on_state_change = on_state_change
        self._on_hu_state_change = on_hu_state_change
        
        # Estado de la orden
        self._order_state = OrderState.IDLE
        self._order_id: Optional[str] = None
        self._order_plan: Optional[OrderPlan] = None
        
        # HUs (hasta 4)
        self._hus: Dict[str, HUStateMachine] = {}
        
        # HU activo actual
        self._active_hu_id: Optional[str] = None
        
        # Timestamps
        self._started_at: Optional[datetime] = None
        self._completed_at: Optional[datetime] = None
        
        # Historial de órdenes completadas (con límite para prevenir OOM)
        self._completed_orders: List[OrderSummary] = []
        
        # Contador de eventos procesados
        self._event_count = 0
    
    # ─────────────────────────────────────────────────────────────────
    # Propiedades
    # ─────────────────────────────────────────────────────────────────
    
    @property
    def order_state(self) -> OrderState:
        """Estado actual de la orden."""
        return self._order_state
    
    @property
    def order_id(self) -> Optional[str]:
        """ID de la orden actual."""
        return self._order_id
    
    @property
    def is_idle(self) -> bool:
        """True si no hay orden activa."""
        return self._order_state == OrderState.IDLE
    
    @property
    def is_in_progress(self) -> bool:
        """True si hay una orden en progreso."""
        return self._order_state == OrderState.IN_PROGRESS

    @property
    def is_paused(self) -> bool:
        """True si la orden está pausada."""
        return self._order_state == OrderState.PAUSED
    
    @property
    def active_hu(self) -> Optional[HUStateMachine]:
        """HU actualmente activo (None si ninguno)."""
        if self._active_hu_id and self._active_hu_id in self._hus:
            return self._hus[self._active_hu_id]
        return None
    
    @property
    def active_hu_id(self) -> Optional[str]:
        """ID del HU activo."""
        return self._active_hu_id
    
    @property
    def hus(self) -> Dict[str, HUStateMachine]:
        """Diccionario de todos los HUs."""
        return self._hus.copy()
    
    @property
    def hu_count(self) -> int:
        """Número de HUs en la orden actual."""
        return len(self._hus)
    
    @property
    def event_count(self) -> int:
        """Número de eventos procesados."""
        return self._event_count
    
    # ─────────────────────────────────────────────────────────────────
    # Acceso a HUs
    # ─────────────────────────────────────────────────────────────────
    
    def get_hu(self, hu_id: str) -> Optional[HUStateMachine]:
        """Obtiene un HU por su ID."""
        return self._hus.get(hu_id)
    
    def get_active_hus(self) -> List[HUStateMachine]:
        """Retorna lista de HUs en estado ACTIVE."""
        return [hu for hu in self._hus.values() if hu.is_active]
    
    def get_passive_hus(self) -> List[HUStateMachine]:
        """Retorna lista de HUs en estado PASSIVE."""
        return [hu for hu in self._hus.values() if hu.is_passive]
    
    def get_hus_in_order(self) -> List[HUStateMachine]:
        """Retorna lista de HUs que pertenecen a la orden (ACTIVE o PASSIVE)."""
        return [hu for hu in self._hus.values() if hu.is_in_order]
    
    # ─────────────────────────────────────────────────────────────────
    # Procesamiento de Eventos
    # ─────────────────────────────────────────────────────────────────
    
    def process_event(self, event_type: str, context: Dict[str, Any]) -> bool:
        """
        Procesa un evento del picker.
        
        Este es el método principal para alimentar la máquina de estados.
        
        Args:
            event_type: Tipo de evento (ORDER_START, ITEM_START, etc.)
            context: Contexto del evento con los datos relevantes
        
        Returns:
            True si el evento fue procesado correctamente
        
        Ejemplo:
            sm.process_event("ORDER_START", {
                "order_id": "ORD-001",
                "total_hus": 2,
                "hus": [{"hu_id": "HU-001", "items": [...]}]
            })
        """
        self._event_count += 1
        
        try:
            event = EventType(event_type)
        except ValueError:
            return False
        
        # Dispatch según tipo
        handlers = {
            EventType.ORDER_START: self._handle_order_start,
            EventType.ORDER_END: self._handle_order_end,
            EventType.ITEM_START: self._handle_item_start,
            EventType.ITEM_END: self._handle_item_end,
            EventType.ORDER_PAUSE: self._handle_order_pause,
            EventType.ORDER_RESUME: self._handle_order_resume
        }
        
        handler = handlers.get(event)
        if handler:
            result = handler(context)
            if self._on_state_change:
                self._on_state_change(self)
            return result
        
        return False

    def _handle_order_pause(self, context: Dict[str, Any]) -> bool:
        """Procesa ORDER_PAUSE."""
        if self._order_state != OrderState.IN_PROGRESS:
            return False
        
        self._order_state = OrderState.PAUSED
        return True

    def _check_not_paused(self, action_name: str) -> bool:
        """
        Verifica que la orden no esté pausada.
        Retorna True si se puede continuar, False si está pausada.
        """
        if self._order_state == OrderState.PAUSED:
            return False
        return True
        
    def _handle_order_resume(self, context: Dict[str, Any]) -> bool:
        """Procesa ORDER_RESUME."""
        if self._order_state != OrderState.PAUSED:
            return False
        
        self._order_state = OrderState.IN_PROGRESS
        return True

    def _handle_order_start(self, context: Dict[str, Any]) -> bool:
        """Procesa ORDER_START."""
        order_id = context.get("order_id")
        hus_data = context.get("hus", [])
        
        if not order_id:
            logger.error("ORDER_START sin order_id")
            return False
        
        # Si hay orden previa, cerrarla
        if self._order_state == OrderState.IN_PROGRESS:
            self._finalize_order()
        
        # Inicializar nueva orden
        self._order_id = order_id
        self._order_state = OrderState.IN_PROGRESS
        self._started_at = datetime.now()
        self._completed_at = None
        self._active_hu_id = None
        
        # Crear plan de orden
        self._order_plan = OrderPlan(
            order_id=order_id,
            description=context.get("description", "")
        )
        
        # Limpiar HUs anteriores y crear nuevos
        self._hus.clear()
        
        for hu_data in hus_data[:self.MAX_HUS]:
            hu_id = hu_data.get("hu_id")
            if hu_id:
                # Crear HU
                hu = HUStateMachine(
                    hu_id=hu_id,
                    initial_state=HUState.UNAVAILABLE,
                    on_state_change=self._on_hu_state_change
                )
                hu.join_order()  # UNAVAILABLE -> PASSIVE
                self._hus[hu_id] = hu
                
                # Agregar al plan
                hu_plan = HUPlan(
                    hu_id=hu_id,
                    items=hu_data.get("items", [])
                )
                self._order_plan.hus.append(hu_plan)
        
        return True
    
    def _handle_item_start(self, context: Dict[str, Any]) -> bool:
        """Procesa ITEM_START."""
        if not self._check_not_paused("ITEM_START"):
            return False
        
        hu_id = context.get("hu_id")
        sku = context.get("sku", "")
        description = context.get("description", "")
        quantity = context.get("quantity", 0)
        item_index = context.get("item_index", 0)
        
        if not hu_id:
            logger.error("ITEM_START sin hu_id")
            return False
        
        if self._order_state != OrderState.IN_PROGRESS:
            return False
        
        # Validar que no haya un item ya activo (regla: no múltiples items simultáneos)
        active_hu = self.active_hu
        if active_hu and active_hu.has_open_item:
            return False
        
        # Obtener o crear HU
        hu = self._hus.get(hu_id)
        if not hu:
            hu = HUStateMachine(
                hu_id=hu_id,
                on_state_change=self._on_hu_state_change
            )
            hu.join_order()
            self._hus[hu_id] = hu
        
        # Desactivar HU anterior si es diferente
        if self._active_hu_id and self._active_hu_id != hu_id:
            old_hu = self._hus.get(self._active_hu_id)
            if old_hu:
                old_hu.deactivate()
        
        # Activar el nuevo HU
        hu.activate(sku, description, quantity, item_index)
        self._active_hu_id = hu_id
        
        return True
    
    def _handle_item_end(self, context: Dict[str, Any]) -> bool:
        """Procesa ITEM_END."""
        if not self._check_not_paused("ITEM_END"):
            return False

        hu_id = context.get("hu_id")
        sku = context.get("sku", "")
        quantity = context.get("quantity", context.get("quantity_loaded", 0))
        
        if not hu_id:
            logger.error("ITEM_END sin hu_id")
            return False
        
        hu = self._hus.get(hu_id)
        if not hu:
            return False
        
        # Completar item
        hu.complete_item(quantity)
        
        return True
    
    def _handle_order_end(self, context: Dict[str, Any]) -> bool:
        """Procesa ORDER_END."""
        if not self._check_not_paused("ORDER_END"):
            return False    

        order_id = context.get("order_id", self._order_id)
        
        if self._order_state != OrderState.IN_PROGRESS:
            return False
        
        # Validar que no haya items abiertos (regla: no cerrar orden con item activo)
        active_hu = self.active_hu
        if active_hu and active_hu.has_open_item:
            return False
        
        self._finalize_order()
        
        return True
    
    
    def _finalize_order(self):
        """Finaliza la orden actual y genera resumen."""
        self._completed_at = datetime.now()
        self._order_state = OrderState.COMPLETED
        
        # Cerrar todos los HUs
        for hu in self._hus.values():
            hu.leave_order()
        
        # Generar resumen
        if self._started_at and self._completed_at:
            duration = (self._completed_at - self._started_at).total_seconds()
            
            hus_summary = []
            total_items = 0
            total_quantity = 0
            
            for hu in self._hus.values():
                hu_data = {
                    "hu_id": hu.hu_id,
                    "items_count": hu.items_loaded_count,
                    "total_quantity": hu.total_quantity_loaded,
                    "items": [item.to_dict() for item in hu.items_loaded]
                }
                hus_summary.append(hu_data)
                total_items += hu.items_loaded_count
                total_quantity += hu.total_quantity_loaded
            
            summary = OrderSummary(
                order_id=self._order_id,
                total_hus=len(self._hus),
                total_items=total_items,
                total_quantity=total_quantity,
                duration_seconds=duration,
                hus_summary=hus_summary,
                started_at=self._started_at,
                completed_at=self._completed_at
            )
            
            # Agregar resumen con límite para prevenir crecimiento ilimitado de memoria
            self._completed_orders.append(summary)
            
            # Mantener solo las últimas N órdenes (eliminar las más antiguas)
            if len(self._completed_orders) > self.MAX_COMPLETED_ORDERS:
                self._completed_orders = self._completed_orders[-self.MAX_COMPLETED_ORDERS:]
        
        self._active_hu_id = None
    
    # ─────────────────────────────────────────────────────────────────
    # Estado y Resumen
    # ─────────────────────────────────────────────────────────────────
    
    def get_state(self) -> Dict[str, Any]:
        """Retorna el estado completo de la máquina de estados."""
        return {
            "order_state": self._order_state.value,
            "order_id": self._order_id,
            "active_hu_id": self._active_hu_id,
            "hu_count": len(self._hus),
            "hus": {hu_id: hu.to_dict() for hu_id, hu in self._hus.items()},
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "event_count": self._event_count
        }
    
    def get_current_context(self) -> Dict[str, Any]:
        """
        Retorna el contexto actual para otros módulos.
        
        Útil para que el contador sepa qué HU está activo y qué SKU se espera.
        """
        active_hu = self.active_hu
        
        return {
            "order_id": self._order_id,
            "order_state": self._order_state.value,
            "active_hu_id": self._active_hu_id,
            "active_hu_state": active_hu.state.value if active_hu else None,
            "current_sku": active_hu.current_sku if active_hu else None,
            "current_quantity": active_hu.current_quantity if active_hu else 0,
            "item_open": active_hu.has_open_item if active_hu else False,
            "hus_states": {
                hu_id: {
                    "state": hu.state.value,
                    "item_open": hu.has_open_item,
                    "current_sku": hu.current_sku
                }
                for hu_id, hu in self._hus.items()
            }
        }
    
    def get_order_summary(self) -> Optional[Dict[str, Any]]:
        """Retorna resumen de la orden actual o última completada."""
        if self._completed_orders:
            summary = self._completed_orders[-1]
            return {
                "order_id": summary.order_id,
                "total_hus": summary.total_hus,
                "total_items": summary.total_items,
                "total_quantity": summary.total_quantity,
                "duration_seconds": summary.duration_seconds,
                "hus_summary": summary.hus_summary,
                "started_at": summary.started_at.isoformat(),
                "completed_at": summary.completed_at.isoformat()
            }
        return None
    
    def reset(self):
        """Resetea completamente la máquina de estados."""
        self._order_state = OrderState.IDLE
        self._order_id = None
        self._order_plan = None
        self._hus.clear()
        self._active_hu_id = None
        self._started_at = None
        self._completed_at = None
        self._event_count = 0
        # Limpiar historial de órdenes completadas para liberar memoria
        self._completed_orders.clear()
    
    def __repr__(self) -> str:
        return (
            f"OrderStateMachine(order={self._order_id}, "
            f"state={self._order_state.value}, "
            f"hus={len(self._hus)}, "
            f"active={self._active_hu_id})"
        )

