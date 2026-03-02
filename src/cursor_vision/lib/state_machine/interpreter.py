"""
State Interpreter - Intérprete de estados para generar mensajes legibles.

Convierte los datos y estados de la máquina de estados en mensajes
formateados para logging y visualización.

Uso:
    from lib.state_machine import StateInterpreter
    
    interpreter = StateInterpreter()
    msg = interpreter.format_order_status(state_machine, order_data)
    logger.info(msg)
"""

from typing import Dict, Any, List, Optional
from .enums import HUState, OrderState, EventType


class StateInterpreter:
    """
    Intérprete de estados que genera mensajes formateados.
    
    Convierte datos de OrderStateMachine y HUStateMachine en strings
    legibles para logs y visualización.
    
    Atributos configurables:
        line_width: Ancho de las líneas separadoras
        symbols: Diccionario de símbolos para estados
    """
    
    # Símbolos por defecto
    DEFAULT_SYMBOLS = {
        "active": "►",
        "completed": "✓",
        "pending": "○",
        "paused": "⏸",
        "error": "✗",
        "success": "✓",
        "star": "★",
        "arrow": "→"
    }
    
    def __init__(self, line_width: int = 50, symbols: Dict[str, str] = None):
        """
        Inicializa el intérprete.
        
        Args:
            line_width: Ancho de las líneas separadoras
            symbols: Diccionario personalizado de símbolos (opcional)
        """
        self.line_width = line_width
        self.symbols = symbols or self.DEFAULT_SYMBOLS.copy()
    
    # ─────────────────────────────────────────────────────────────────
    # Formateo de Órdenes
    # ─────────────────────────────────────────────────────────────────
    
    def format_order_breakdown(self, order_data: Dict[str, Any]) -> str:
        """
        Genera el desglose completo de una orden.
        
        Args:
            order_data: Diccionario con datos de la orden
                {
                    "order_id": str,
                    "description": str,
                    "hus": [{"hu_id": str, "items": [...]}]
                }
        
        Returns:
            String formateado con el desglose
        """
        if not order_data:
            return ""
        
        lines = []
        hus = order_data.get("hus", [])
        
        lines.append("")
        lines.append("=" * self.line_width)
        lines.append(f"ORDEN: {order_data.get('order_id', 'N/A')}")
        lines.append(f"{order_data.get('description', '')}")
        lines.append(f"Total HUs: {len(hus)}")
        lines.append("=" * self.line_width)
        
        for i, hu in enumerate(hus, 1):
            items = hu.get("items", [])
            lines.append("")
            lines.append(f"HU {i}/{len(hus)}: {hu.get('hu_id', 'N/A')}")
            lines.append(f"Items: {len(items)}")
            lines.append("-" * 40)
            
            for j, item in enumerate(items, 1):
                lines.append(f"  {j}. {item.get('sku', 'N/A')}")
                lines.append(f"     {item.get('description', '')}")
                lines.append(f"     Cantidad: {item.get('quantity', 0)}")
        
        lines.append("")
        lines.append("=" * self.line_width)
        
        return "\n".join(lines)
    
    def format_order_status(
        self, 
        order_state: str,
        order_id: str,
        hus_status: List[Dict[str, Any]]
    ) -> str:
        """
        Genera el estado actual de la orden con progreso de HUs.
        
        Args:
            order_state: Estado actual de la orden (IDLE, IN_PROGRESS, etc.)
            order_id: ID de la orden
            hus_status: Lista de estados de HUs
                [{"hu_id": str, "completed": int, "total": int, "is_active": bool}]
        
        Returns:
            String formateado con el estado
        """
        lines = []
        
        lines.append("")
        lines.append("-" * self.line_width)
        lines.append(f"ESTADO: {order_state}")
        lines.append(f"Orden: {order_id}")
        lines.append("-" * self.line_width)
        
        for hu in hus_status:
            line = self.format_hu_status_line(
                hu_id=hu.get("hu_id", "N/A"),
                completed=hu.get("completed", 0),
                total=hu.get("total", 0),
                is_active=hu.get("is_active", False)
            )
            lines.append(line)
        
        lines.append("-" * self.line_width)
        
        return "\n".join(lines)
    
    def format_order_summary(self, summary: Dict[str, Any]) -> str:
        """
        Genera el resumen de una orden completada.
        
        Args:
            summary: Diccionario con el resumen de la orden
                {
                    "order_id": str,
                    "total_hus": int,
                    "total_items": int,
                    "total_quantity": int,
                    "duration_seconds": float,
                    "hus_summary": [...]
                }
        
        Returns:
            String formateado con el resumen
        """
        if not summary:
            return ""
        
        lines = []
        
        lines.append("")
        lines.append("=" * self.line_width)
        lines.append(f"{self.symbols['star']} ORDEN COMPLETADA")
        lines.append("=" * self.line_width)
        lines.append(f"Orden: {summary.get('order_id', 'N/A')}")
        lines.append(f"HUs procesados: {summary.get('total_hus', 0)}")
        lines.append(f"Items totales: {summary.get('total_items', 0)}")
        lines.append(f"Cantidad total: {summary.get('total_quantity', 0)}")
        lines.append(f"Duración: {summary.get('duration_seconds', 0):.2f} segundos")
        
        hus_summary = summary.get('hus_summary', [])
        if hus_summary:
            lines.append("")
            lines.append("Detalle por HU:")
            for hu in hus_summary:
                lines.append(
                    f"  {hu.get('hu_id', 'N/A')}: "
                    f"{hu.get('items_count', 0)} items, "
                    f"{hu.get('total_quantity', 0)} unidades"
                )
        
        lines.append("=" * self.line_width)
        
        return "\n".join(lines)
    
    # ─────────────────────────────────────────────────────────────────
    # Formateo de HUs
    # ─────────────────────────────────────────────────────────────────
    
    def format_hu_status_line(
        self, 
        hu_id: str, 
        completed: int, 
        total: int, 
        is_active: bool
    ) -> str:
        """
        Genera una línea de estado para un HU.
        
        Args:
            hu_id: ID del HU
            completed: Items completados
            total: Total de items
            is_active: Si el HU está activo
        
        Returns:
            String formateado, ej: "► HU-001: 2/5 items"
        """
        if is_active:
            symbol = self.symbols["active"]
        elif completed == total and total > 0:
            symbol = self.symbols["completed"]
        else:
            symbol = self.symbols["pending"]
        
        return f"{symbol} {hu_id}: {completed}/{total} items"
    
    def format_hu_transition(
        self, 
        hu_id: str, 
        old_state: HUState, 
        new_state: HUState
    ) -> str:
        """
        Genera mensaje de transición de estado de un HU.
        
        Args:
            hu_id: ID del HU
            old_state: Estado anterior
            new_state: Nuevo estado
        
        Returns:
            String formateado, ej: "HU-001: PASSIVE → ACTIVE"
        """
        return f"{hu_id}: {old_state.value} {self.symbols['arrow']} {new_state.value}"
    
    # ─────────────────────────────────────────────────────────────────
    # Formateo de Eventos
    # ─────────────────────────────────────────────────────────────────
    
    def format_event_received(
        self, 
        event_type: str, 
        event_count: int,
        context: Dict[str, Any]
    ) -> str:
        """
        Genera mensaje de evento recibido.
        
        Args:
            event_type: Tipo de evento (ORDER_START, ITEM_START, etc.)
            event_count: Número de evento
            context: Contexto del evento
        
        Returns:
            String formateado con detalles del evento
        """
        lines = []
        
        lines.append("")
        lines.append("-" * self.line_width)
        lines.append(f"EVENTO #{event_count}: {event_type}")
        lines.append("-" * self.line_width)
        
        if event_type == EventType.ORDER_START.value:
            lines.append(f"Orden: {context.get('order_id', 'N/A')}")
            lines.append(f"Total HUs: {context.get('total_hus', 0)}")
            
        elif event_type == EventType.ITEM_START.value:
            lines.append(f"HU: {context.get('hu_id', 'N/A')}")
            lines.append(f"SKU: {context.get('sku', 'N/A')}")
            lines.append(f"{context.get('description', '')} x{context.get('quantity', 0)}")
            
        elif event_type == EventType.ITEM_END.value:
            lines.append(f"HU: {context.get('hu_id', 'N/A')}")
            lines.append(f"SKU: {context.get('sku', 'N/A')} x{context.get('quantity', 0)}")
            
        elif event_type == EventType.ORDER_END.value:
            lines.append(f"Orden: {context.get('order_id', 'N/A')} completada")
            
        elif event_type == EventType.ORDER_PAUSE.value:
            lines.append(f"{self.symbols['paused']} Orden pausada")
            
        elif event_type == EventType.ORDER_RESUME.value:
            lines.append(f"{self.symbols['active']} Orden reanudada")
        
        return "\n".join(lines)
    
    def format_event_result(
        self, 
        event_type: str, 
        success: bool, 
        context: Dict[str, Any]
    ) -> str:
        """
        Genera mensaje de resultado de evento.
        
        Args:
            event_type: Tipo de evento
            success: Si fue exitoso
            context: Contexto del evento
        
        Returns:
            String formateado con el resultado
        """
        symbol = self.symbols["success"] if success else self.symbols["error"]
        status = "procesado" if success else "rechazado"
        
        if event_type == EventType.ITEM_START.value:
            return f"{symbol} ITEM INICIADO: {context.get('sku', 'N/A')} en {context.get('hu_id', 'N/A')}"
        
        elif event_type == EventType.ITEM_END.value:
            return f"{symbol} ITEM COMPLETADO: {context.get('sku', 'N/A')}"
        
        elif event_type == EventType.ORDER_START.value:
            return f"{symbol} ORDEN INICIADA: {context.get('order_id', 'N/A')}"
        
        elif event_type == EventType.ORDER_END.value:
            return f"{symbol} ORDEN FINALIZADA: {context.get('order_id', 'N/A')}"
        
        elif event_type == EventType.ORDER_PAUSE.value:
            return f"{self.symbols['paused']} ORDEN PAUSADA"
        
        elif event_type == EventType.ORDER_RESUME.value:
            return f"{self.symbols['active']} ORDEN REANUDADA"
        
        return f"{symbol} {event_type} {status}"
    
    # ─────────────────────────────────────────────────────────────────
    # Formateo de Errores/Advertencias
    # ─────────────────────────────────────────────────────────────────
    
    def format_invalid_action(
        self, 
        action: str, 
        reason: str, 
        suggestion: str
    ) -> str:
        """
        Genera mensaje de acción inválida.
        
        Args:
            action: La acción que se intentó
            reason: Razón por la que falló
            suggestion: Sugerencia de qué hacer
        
        Returns:
            String formateado con el error
        """
        lines = []
        lines.append(f"{self.symbols['error']} No se puede: {action}")
        lines.append(f"  Razón: {reason}")
        lines.append(f"  Sugerencia: {suggestion}")
        return "\n".join(lines)
    
    def format_warning(self, message: str) -> str:
        """Genera mensaje de advertencia."""
        return f"⚠ {message}"
    
    # ─────────────────────────────────────────────────────────────────
    # Formateo de Estado Actual (para logging)
    # ─────────────────────────────────────────────────────────────────
    
    def format_current_state(self, context: Dict[str, Any]) -> str:
        """
        Genera mensaje del estado actual de la state machine.
        
        Args:
            context: Resultado de state_machine.get_current_context()
                {
                    "order_id": str,
                    "order_state": str,
                    "active_hu_id": str,
                    "current_sku": str,
                    "current_quantity": int,
                    "item_open": bool,
                    "hus_states": {...}
                }
        
        Returns:
            String formateado con el estado actual
        """
        lines = []
        
        lines.append("-" * 40)
        lines.append(f"Estado: {context.get('order_state', 'N/A')}")
        lines.append(f"HU activo: {context.get('active_hu_id') or 'ninguno'}")
        
        if context.get("current_sku"):
            lines.append(f"Item actual: {context['current_sku']} x{context.get('current_quantity', 0)}")
            lines.append(f"Item abierto: {context.get('item_open', False)}")
        
        hus_states = context.get("hus_states", {})
        if hus_states:
            lines.append("HUs:")
            for hu_id, hu_state in hus_states.items():
                state = hu_state.get("state", "N/A")
                if state == "ACTIVE":
                    icon = "🟢"
                elif state == "PASSIVE":
                    icon = "🟡"
                else:
                    icon = "⚫"
                
                item_info = f" [{hu_state.get('current_sku')}]" if hu_state.get("current_sku") else ""
                lines.append(f"  {icon} {hu_id}: {state}{item_info}")
        
        return "\n".join(lines)
    
    def format_plan_received(self, order_id: str, total_hus: int, hus: List[Dict]) -> str:
        """
        Genera mensaje de plan recibido.
        
        Args:
            order_id: ID de la orden
            total_hus: Total de HUs
            hus: Lista de HUs con sus items
        
        Returns:
            String formateado
        """
        lines = []
        
        lines.append("")
        lines.append("-" * self.line_width)
        lines.append(f"PLAN RECIBIDO: {order_id}")
        lines.append("-" * self.line_width)
        lines.append(f"Total HUs: {total_hus}")
        
        for hu in hus:
            hu_id = hu.get("hu_id", "N/A")
            items = hu.get("items", [])
            lines.append(f"└── {hu_id}: {len(items)} items")
        
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Funciones helper (para uso directo sin instanciar la clase)
# ─────────────────────────────────────────────────────────────────────

# Instancia global por defecto
_default_interpreter = StateInterpreter()

def format_hu_status_line(hu_id: str, completed: int, total: int, is_active: bool) -> str:
    """Wrapper de conveniencia para format_hu_status_line."""
    return _default_interpreter.format_hu_status_line(hu_id, completed, total, is_active)

def format_hu_transition(hu_id: str, old_state: HUState, new_state: HUState) -> str:
    """Wrapper de conveniencia para format_hu_transition."""
    return _default_interpreter.format_hu_transition(hu_id, old_state, new_state)

def format_event_result(event_type: str, success: bool, context: Dict[str, Any]) -> str:
    """Wrapper de conveniencia para format_event_result."""
    return _default_interpreter.format_event_result(event_type, success, context)

def format_invalid_action(action: str, reason: str, suggestion: str) -> str:
    """Wrapper de conveniencia para format_invalid_action."""
    return _default_interpreter.format_invalid_action(action, reason, suggestion)

