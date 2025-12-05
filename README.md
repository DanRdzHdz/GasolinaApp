# 🚀 Aplicación de Optimización de Inventarios

## Descripción
Aplicación web para optimizar políticas de inventario (s, S, Precio) usando:
- Red Neuronal para predecir demanda
- Algoritmo Genético con convergencia automática
- Interfaz interactiva con Streamlit

## Características
- ✅ Subir archivos Excel/CSV con datos propios
- ✅ Elegir objetivo: Ganancia, Flujo o Balance
- ✅ Ajustar parámetros de la red y el AG
- ✅ Convergencia automática (sin generaciones fijas)
- ✅ Gráficas interactivas
- ✅ Descargar resultados

## Instalación

```bash
# 1. Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la aplicación
streamlit run app_inventarios.py
```

## Formato de Datos
El archivo debe tener estas columnas:
- `dia`: número de día (1, 2, 3, ...)
- `dia_semana`: día de la semana (1-7)
- `precio`: precio del combustible
- `demanda`: demanda en litros

Ejemplo:
```
dia,dia_semana,precio,demanda
1,1,23.82,14160.45
2,2,23.92,11431.49
3,3,23.89,10007.07
...
```

## Uso
1. Abrir la aplicación en el navegador (http://localhost:8501)
2. Subir archivo de datos (o usar ejemplo)
3. Seleccionar objetivo de optimización
4. Ajustar parámetros si es necesario
5. Clic en "EJECUTAR OPTIMIZACIÓN"
6. Ver resultados y descargar

## Parámetros del AG
- **Tolerancia**: Cambio mínimo para considerar mejora (default: 0.001)
- **Paciencia**: Generaciones sin mejora antes de parar (default: 30)

## Autor
Generado con Claude AI
