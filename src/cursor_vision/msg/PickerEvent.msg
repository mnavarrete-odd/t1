# PickerEvent - Evento del picker (ORDER_START, ITEM_START, ITEM_END, ORDER_END, etc.)

# Constantes para tipos de evento
string ORDER_START=ORDER_START
string ORDER_END=ORDER_END
string ITEM_START=ITEM_START
string ITEM_END=ITEM_END
string ORDER_PAUSE=ORDER_PAUSE
string ORDER_RESUME=ORDER_RESUME
string WAIT=WAIT

# Datos del evento
string event_type
string order_id
string hu_id
string sku
string description
int32 quantity
int32 item_index
int32 hu_index
int32 total_hus
builtin_interfaces/Time timestamp

