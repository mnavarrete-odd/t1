# CursorVision - Guia de Setup, Build y Ejecucion

## Prerequisitos

### 1. ROS2 Humble (o Jazzy)

```bash
# Verificar que ROS2 esta instalado
source /opt/ros/humble/setup.bash   # o /opt/ros/jazzy/setup.bash
ros2 --version
```

Si no esta instalado, seguir la guia oficial: https://docs.ros.org/en/humble/Installation.html

### 2. Dependencias ROS2

```bash
sudo apt update
sudo apt install -y \
  ros-${ROS_DISTRO}-cv-bridge \
  ros-${ROS_DISTRO}-message-filters \
  ros-${ROS_DISTRO}-sensor-msgs \
  ros-${ROS_DISTRO}-std-msgs \
  python3-colcon-common-extensions \
  python3-rosdep
```

### 3. Dependencias Python (GPU/Vision)

```bash
pip3 install \
  ultralytics>=8.0.0 \
  torch>=2.0.0 \
  torchvision>=0.15.0 \
  scipy>=1.10.0 \
  numpy>=1.24.0 \
  opencv-python>=4.8.0 \
  pyyaml>=6.0 \
  transformers
```

> **Nota sobre torch**: Si el PC tiene GPU NVIDIA, instalar la version CUDA
> de PyTorch: https://pytorch.org/get-started/locally/
>
> Ejemplo para CUDA 12.1:
> ```bash
> pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. GPU y CUDA (recomendado)

- NVIDIA GPU con al menos 4GB VRAM (recomendado 8GB+)
- CUDA 11.8+ y cuDNN instalados
- Verificar: `nvidia-smi`

Si no hay GPU disponible, el sistema funciona en CPU (mas lento).
Ajustar `counter_device: "cpu"` en la config.

---

## Estructura del Proyecto

```
cencosud-cursor/
  SETUP_GUIDE.md          <-- Este archivo
  src/
    cursor_vision/        <-- Paquete ROS2
      CMakeLists.txt
      package.xml
      config/             <-- Archivos de configuracion
      core/               <-- Librerias de CounterVision (detector, photographer, counter)
      launch/             <-- Launch files
      lib/                <-- Componentes nuevos (pipelines, bridges, controllers)
      models/             <-- Modelos YOLO + DINO (hay que copiarlos aqui)
      msg/                <-- Mensajes ROS2 custom
      nodes/              <-- Nodos ejecutables
      tests/              <-- Tests unitarios
```

---

## Paso 1: Copiar el Proyecto al PC Destino

```bash
# Opcion A: Copiar directamente
scp -r cencosud-cursor/ usuario@pc-destino:/ruta/deseada/

# Opcion B: Si usas git
cd cencosud-cursor
git init && git add . && git commit -m "Initial commit"
# Luego clonar en el PC destino
```

---

## Paso 2: Colocar los Modelos

Los modelos NO estan incluidos en el repositorio (son archivos pesados).
Copiarlos al directorio `models/` del paquete:

```bash
cd cencosud-cursor/src/cursor_vision/

# Crear directorio de modelos si no existe
mkdir -p models/dino

# Copiar el modelo YOLO
cp /ruta/a/11-NEW.pt models/
# O si tienes la version TensorRT (mas rapido):
cp /ruta/a/11-NEW.engine models/

# Copiar el modelo DINO (embedder para el counter)
cp -r /ruta/a/dino/* models/dino/
```

> **Donde encontrar los modelos**: Estan en el proyecto `cencosud-counter-vision`
> en su directorio `models/`. O pregunta al equipo.
>
> **Alternativa sin DINO**: Si no tienes el modelo DINO o quieres ahorrar VRAM,
> edita `config/counter_default.yaml` y cambia:
> ```yaml
> embedder:
>   type: hist    # en vez de "dino" -- usa histogramas, sin GPU
> ```

---

## Paso 3: Configurar

### 3.1 Nombres de camaras

Editar `config/inventory_node_config.yaml`:

```yaml
# Cambiar los nombres segun los topics de tus camaras RealSense/USB
camera_names: "primary_camera,secondary_camera"
```

Estos nombres se usan para suscribirse a:
- `/{camera_name}/color/image_raw`
- `/{camera_name}/depth/image_raw`

### 3.2 Modelo YOLO

```yaml
detector_model_path: "models/11-NEW.pt"   # relativo al paquete share, o ruta absoluta
detector_confidence: 0.5
```

### 3.3 Counter (GPU vs CPU)

```yaml
counter_device: "cuda"   # o "cpu" si no hay GPU
```

### 3.4 Debug (opcional)

```yaml
debug_save_keyframes: false       # true para guardar KFs a disco
debug_save_counter_video: false   # true para generar video de tracking
debug_output_dir: "/tmp/cursor_debug"
```

### 3.5 SAP Event API

Editar `config/sap_event_config.yaml` si necesitas cambiar el puerto:

```yaml
server:
  host: "0.0.0.0"
  port: 1616          # Puerto donde escucha la API HTTP
api:
  endpoint: "/api/sap-info"
```

---

## Paso 4: Build con colcon

```bash
# 1. Ir a la raiz del workspace (donde esta src/)
cd cencosud-cursor/

# 2. Sourcer ROS2
source /opt/ros/humble/setup.bash

# 3. Instalar dependencias con rosdep (primera vez)
sudo rosdep init 2>/dev/null || true
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# 4. Build
colcon build --packages-select cursor_vision --symlink-install
```

> **`--symlink-install`** permite editar archivos Python sin re-buildear.
> Solo necesitas re-buildear si cambias CMakeLists.txt, package.xml, o .msg files.

### Si el build falla

**Error: `rosidl_default_generators` no encontrado**
```bash
sudo apt install ros-${ROS_DISTRO}-rosidl-default-generators
```

**Error: `ament_cmake_python` no encontrado**
```bash
sudo apt install ros-${ROS_DISTRO}-ament-cmake-python
```

**Error: modulo Python no encontrado (torch, ultralytics, etc)**
```bash
pip3 install -r src/cursor_vision/requirements.txt
```

---

## Paso 5: Sourcer el Workspace

Despues de cada build exitoso:

```bash
source install/setup.bash
```

> **Tip**: Agregar ambos source a tu `.bashrc`:
> ```bash
> echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
> echo "source ~/cencosud-cursor/install/setup.bash" >> ~/.bashrc
> ```

---

## Paso 6: Ejecutar

### Opcion A: Launcher completo (recomendado)

Lanza ambos nodos (`inventory_node` + `sap_event_node`):

```bash
ros2 launch cursor_vision cursor_vision.launch.py
```

### Opcion B: Nodos individuales

Terminal 1 - Inventory Node:
```bash
ros2 run cursor_vision inventory_node --ros-args \
  --params-file $(ros2 pkg prefix cursor_vision)/share/cursor_vision/config/inventory_node_config.yaml
```

Terminal 2 - SAP Event Node:
```bash
ros2 run cursor_vision sap_event_node
```

### Verificar que esta corriendo

```bash
# Ver nodos activos
ros2 node list

# Ver topics publicados
ros2 topic list

# Escuchar conteo de una camara
ros2 topic echo /camera/primary_camera/count

# Escuchar el resumen de la state machine
ros2 topic echo /state_machine/summary
```

---

## Paso 7: Enviar Eventos de Prueba

El `sap_event_node` expone una API HTTP. Enviar eventos con curl:

### Iniciar una orden

```bash
curl -X POST http://localhost:1616/api/sap-info \
  -H "Content-Type: application/json" \
  -d '{
    "action": "order_start",
    "order_id": "ORD-001",
    "description": "Orden de prueba",
    "hus": [
      {
        "hu_id": "HU-A",
        "items": [
          {"sku": "SKU-001", "description": "Producto A", "quantity": 10},
          {"sku": "SKU-002", "description": "Producto B", "quantity": 5}
        ]
      }
    ]
  }'
```

### Iniciar una tarea (activa la deteccion y el conteo)

```bash
curl -X POST http://localhost:1616/api/sap-info \
  -H "Content-Type: application/json" \
  -d '{
    "action": "item_start",
    "order_id": "ORD-001",
    "hu_id": "HU-A",
    "sku": "SKU-001",
    "quantity": 10
  }'
```

### Finalizar una tarea

```bash
curl -X POST http://localhost:1616/api/sap-info \
  -H "Content-Type: application/json" \
  -d '{
    "action": "item_end",
    "order_id": "ORD-001",
    "hu_id": "HU-A",
    "sku": "SKU-001",
    "quantity": 10
  }'
```

### Finalizar la orden

```bash
curl -X POST http://localhost:1616/api/sap-info \
  -H "Content-Type: application/json" \
  -d '{
    "action": "order_end",
    "order_id": "ORD-001"
  }'
```

---

## Ciclo de Vida del Sistema

```
ORDER_START  -->  ITEM_START  -->  [deteccion activa, keyframes, conteo]
                                        |
                  ITEM_END   <----------+  (KF final forzado, pipelines se desactivan)
                      |
                  ITEM_START  -->  [nuevo ciclo de conteo]
                      |
                  ITEM_END
                      |
ORDER_END    -->  [resultado final publicado, todo se reinicia]
```

- **ITEM_START**: Genera un keyframe inicial forzado, activa el Photographer.
- **Durante la tarea**: Cada keyframe que produce el Photographer se pasa al Counter.
- **ITEM_END**: Genera un keyframe final forzado, desactiva el Photographer.
- **ORDER_END**: Publica el resumen final en `/inventory/summary` y resetea todo.

---

## Topics ROS2 Publicados

| Topic | Tipo | Descripcion |
|-------|------|-------------|
| `/camera/{name}/count` | `cursor_vision/CountResult` | Conteo por cada keyframe procesado |
| `/inventory/summary` | `std_msgs/String` (JSON) | Resumen final al terminar la orden |
| `/state_machine/summary` | `std_msgs/String` (JSON) | Estado actual de la state machine |
| `/picker/events` | `cursor_vision/PickerEvent` | Eventos publicados por sap_event_node |
| `/picker/plan` | `cursor_vision/PickerPlan` | Plan de la orden (solo en order_start) |

---

## Troubleshooting

### "No frame available yet" en force_keyframe

Las camaras aun no estan publicando o los topics no coinciden.
Verificar:
```bash
ros2 topic list | grep image
ros2 topic hz /primary_camera/color/image_raw
```

### VRAM insuficiente

- Cambiar embedder a `hist` en `counter_default.yaml`
- Usar TensorRT (`.engine`) en vez de `.pt` para YOLO
- Reducir `detection_fps` en la config

### Counter no produce resultados

- Verificar que hay detecciones (el Photographer necesita ver objetos)
- Activar debug: `debug_save_keyframes: true` y revisar `/tmp/cursor_debug/`
- Verificar que `allowed_names` en `counter_default.yaml` incluye las clases detectadas

### Build falla por mensajes

```bash
# Limpiar build previo y re-buildear
rm -rf build/ install/ log/
colcon build --packages-select cursor_vision
```

---

## Re-build rapido (despues de cambios en Python)

Si usaste `--symlink-install`, solo necesitas re-buildear cuando cambias:
- `CMakeLists.txt`
- `package.xml`
- Archivos `.msg`

Para cambios en Python (nodes, lib, core):
```bash
# No necesitas re-build, solo re-sourcear:
source install/setup.bash
```

Para cambios en mensajes o cmake:
```bash
colcon build --packages-select cursor_vision
source install/setup.bash
```

---

## Tests

```bash
cd cencosud-cursor/src/cursor_vision
python3 -m pytest tests/ -v
```

> Nota: Algunos tests de integracion requieren torch y scipy instalados.
> Los tests unitarios del TaskController y conversiones de bbox funcionan
> sin dependencias GPU (usan mocks).
