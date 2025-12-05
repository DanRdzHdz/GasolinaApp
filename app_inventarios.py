"""
APLICACIÓN DE OPTIMIZACIÓN DE INVENTARIOS - ENSEMBLE
=====================================================
- Ensemble de redes neuronales (más estable)
- Usuario elige: maximizar ganancia, flujo o balance
- Subir Excel/CSV con datos
- Algoritmo genético con convergencia automática
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Optimizador de Inventarios",
    page_icon="⛽",
    layout="wide"
)

st.title("⛽ Optimizador de Política de Inventarios")
st.markdown("*Versión Ensemble v3.0 - Parámetros Fase 3 corregidos*")
st.markdown("---")

# ============================================================
# FUNCIONES
# ============================================================

@st.cache_data
def cargar_datos(archivo):
    """Carga datos desde archivo subido"""
    if archivo.name.endswith('.csv'):
        df = pd.read_csv(archivo)
    else:
        df = pd.read_excel(archivo)
    return df

def preparar_datos(df):
    """Prepara los datos para entrenamiento"""
    df = df.copy()
    df['demanda_t_minus_1'] = df['demanda'].shift(1)
    df['demanda_t_plus_1'] = df['demanda'].shift(-1)
    df_clean = df.dropna().reset_index(drop=True)
    
    X = df_clean[['dia_semana', 'demanda_t_minus_1', 'precio']].values
    Y = df_clean['demanda_t_plus_1'].values
    
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    return {
        'X_train': X[:train_end],
        'Y_train': Y[:train_end],
        'X_val': X[train_end:val_end],
        'Y_val': Y[train_end:val_end],
        'X_test': X[val_end:],
        'Y_test': Y[val_end:],
        'df_clean': df_clean
    }


class EnsembleRedes:
    """Ensemble de redes neuronales - promedia predicciones"""
    
    def __init__(self, n_redes=5, arquitectura=(8, 8), activacion='relu'):
        self.n_redes = n_redes
        self.arquitectura = arquitectura
        self.activacion = activacion
        self.redes = []
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.semillas_usadas = []
        self.mse_individual = []
    
    def entrenar(self, X_train, Y_train, X_val, Y_val, progress_callback=None):
        """Entrena el ensemble de redes"""
        
        # Ajustar scalers
        self.scaler_X.fit(X_train)
        self.scaler_Y.fit(Y_train.reshape(-1, 1))
        
        X_train_scaled = self.scaler_X.transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        Y_train_scaled = self.scaler_Y.transform(Y_train.reshape(-1, 1)).ravel()
        Y_val_scaled = self.scaler_Y.transform(Y_val.reshape(-1, 1)).ravel()
        
        # Generar semillas aleatorias únicas
        rng = np.random.default_rng(int(time.time()))
        self.semillas_usadas = rng.integers(0, 10000, size=self.n_redes).tolist()
        
        for i, semilla in enumerate(self.semillas_usadas):
            mlp = MLPRegressor(
                hidden_layer_sizes=self.arquitectura,
                activation=self.activacion,
                solver='adam',
                max_iter=3000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=50,
                random_state=semilla,
                verbose=False
            )
            mlp.fit(X_train_scaled, Y_train_scaled)
            self.redes.append(mlp)
            
            # Calcular MSE individual
            Y_pred_scaled = mlp.predict(X_val_scaled)
            mse = mean_squared_error(Y_val_scaled, Y_pred_scaled)
            self.mse_individual.append(mse)
            
            if progress_callback:
                progress_callback(i + 1, self.n_redes, semilla, mse)
        
        # Calcular MSE del ensemble
        Y_pred_ensemble = self._predecir_scaled(X_val_scaled)
        mse_ensemble = mean_squared_error(Y_val_scaled, Y_pred_ensemble)
        
        return mse_ensemble
    
    def _predecir_scaled(self, X_scaled):
        """Predice promediando todas las redes (datos ya escalados)"""
        predicciones = np.array([red.predict(X_scaled) for red in self.redes])
        return np.mean(predicciones, axis=0)
    
    def predecir(self, dia_semana, demanda_anterior, precio):
        """Predice demanda promediando todas las redes"""
        X_nuevo = np.array([[dia_semana, demanda_anterior, precio]])
        X_scaled = self.scaler_X.transform(X_nuevo)
        
        predicciones = []
        for red in self.redes:
            Y_pred_scaled = red.predict(X_scaled)
            Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled.reshape(-1, 1)).ravel()[0]
            predicciones.append(Y_pred)
        
        return np.mean(predicciones)
    
    def predecir_con_detalle(self, dia_semana, demanda_anterior, precio):
        """Predice y muestra el detalle de cada red"""
        X_nuevo = np.array([[dia_semana, demanda_anterior, precio]])
        X_scaled = self.scaler_X.transform(X_nuevo)
        
        predicciones = []
        for red in self.redes:
            Y_pred_scaled = red.predict(X_scaled)
            Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled.reshape(-1, 1)).ravel()[0]
            predicciones.append(Y_pred)
        
        return {
            'promedio': np.mean(predicciones),
            'std': np.std(predicciones),
            'min': np.min(predicciones),
            'max': np.max(predicciones),
            'individuales': predicciones
        }


def simular_politica(ensemble, s, S, precio_venta, 
                     dias=30, costo_compra=23.60, h=0.002,
                     inventario_inicial=10000, dia_semana_inicial=2, 
                     demanda_inicial=10000, capacidad_tanque=15000):
    """
    Simula la política de inventarios usando el ensemble
    
    Parámetros económicos (Fase 3):
    - costo_compra: $23.60 MXN/L (costo mayorista)
    - h: 0.002 MXN/L/día (costo almacenamiento)
    - capacidad_tanque: 15,000 L
    
    Cálculos:
    - Ganancia diaria = ventas * (precio_venta - costo_compra)
    - Flujo diario = ingresos - costo_compras - costo_inventario
    - Promedio = suma(30 días) / 30
    """
    inventario = inventario_inicial
    dia_semana = dia_semana_inicial
    demanda_anterior = demanda_inicial
    
    ganancias = []
    flujos = []
    historico = []
    
    for dia in range(1, dias + 1):
        demanda = ensemble.predecir(dia_semana, demanda_anterior, precio_venta)
        demanda = max(0, demanda)
        
        ventas = min(demanda, inventario)
        inventario -= ventas
        
        compras = 0
        if inventario <= s:
            # Ordenar hasta S, pero sin exceder capacidad del tanque
            compras = min(S - inventario, capacidad_tanque - inventario)
            inventario += compras
        
        # CÁLCULO DE GANANCIA:
        # ganancia = ventas * margen = ventas * (precio - costo)
        ganancia = ventas * (precio_venta - costo_compra)
        
        # CÁLCULO DE FLUJO DE EFECTIVO:
        # flujo = ingresos - egresos
        ingresos = ventas * precio_venta
        costo_compras = compras * costo_compra
        costo_inventario = h * inventario
        flujo = ingresos - costo_compras - costo_inventario
        
        ganancias.append(ganancia)
        flujos.append(flujo)
        
        historico.append({
            'dia': dia,
            'dia_semana': dia_semana,
            'demanda': demanda,
            'ventas': ventas,
            'compras': compras,
            'inventario': inventario,
            'ganancia': ganancia,
            'flujo': flujo
        })
        
        demanda_anterior = demanda
        dia_semana = (dia_semana % 7) + 1
    
    # PROMEDIOS: suma de 30 días / 30
    return np.mean(ganancias), np.mean(flujos), historico


class AlgoritmoGenetico:
    """AG con parada por convergencia"""
    
    def __init__(self, ensemble, w1=0.5, w2=0.5,
                 tam_poblacion=40, max_generaciones=500,
                 tolerancia=0.001, paciencia=30,
                 prob_mutacion=0.1, prob_cruce=0.8,
                 costo_compra=23.60, h=0.002, semilla_ga=None):
        
        self.ensemble = ensemble
        self.w1 = w1
        self.w2 = w2
        self.tam_poblacion = tam_poblacion
        self.max_generaciones = max_generaciones
        self.tolerancia = tolerancia
        self.paciencia = paciencia
        self.prob_mutacion = prob_mutacion
        self.prob_cruce = prob_cruce
        self.costo_compra = costo_compra
        self.h = h
        
        if semilla_ga is not None:
            np.random.seed(semilla_ga)
        
        self.limites = {
            's': (2000, 12000),      # Punto de reorden: 2,000 - 12,000 L
            'S': (8000, 15000),      # Nivel reposición: 8,000 - 15,000 L
            'precio': (23.00, 23.99) # Precio venta: $23.00 - $23.99 MXN/L
        }
        
        self.historial = []
        
    def crear_individuo(self):
        return {
            's': np.random.uniform(*self.limites['s']),
            'S': np.random.uniform(*self.limites['S']),
            'precio': np.random.uniform(*self.limites['precio'])
        }
    
    def evaluar(self, individuo):
        s = individuo['s']
        S = individuo['S']
        # Asegurar que S > s
        if S <= s:
            S = min(s + 1000, 15000)
        precio = individuo['precio']
        
        ganancia, flujo, _ = simular_politica(
            self.ensemble, s, S, precio,
            costo_compra=self.costo_compra, h=self.h
        )
        
        F = self.w1 * (-ganancia) + self.w2 * (-flujo)
        return F, ganancia, flujo
    
    def seleccion_torneo(self, poblacion, fitness, k=3):
        indices = np.random.choice(len(poblacion), k, replace=False)
        mejor_idx = indices[np.argmin([fitness[i] for i in indices])]
        return poblacion[mejor_idx].copy()
    
    def cruce(self, padre1, padre2):
        if np.random.random() < self.prob_cruce:
            alpha = np.random.random()
            return {
                's': alpha * padre1['s'] + (1-alpha) * padre2['s'],
                'S': alpha * padre1['S'] + (1-alpha) * padre2['S'],
                'precio': alpha * padre1['precio'] + (1-alpha) * padre2['precio']
            }
        return padre1.copy()
    
    def mutacion(self, individuo):
        for key in ['s', 'S', 'precio']:
            if np.random.random() < self.prob_mutacion:
                rango = self.limites[key][1] - self.limites[key][0]
                delta = np.random.normal(0, rango * 0.1)
                individuo[key] += delta
                individuo[key] = np.clip(individuo[key], *self.limites[key])
        return individuo
    
    def optimizar(self, progress_bar=None, status_text=None):
        """Ejecuta el AG con criterio de convergencia"""
        poblacion = [self.crear_individuo() for _ in range(self.tam_poblacion)]
        
        mejor_global = None
        mejor_fitness_global = float('inf')
        mejor_ganancia_global = 0
        mejor_flujo_global = 0
        
        sin_mejora = 0
        fitness_anterior = float('inf')
        
        for gen in range(self.max_generaciones):
            evaluaciones = [self.evaluar(ind) for ind in poblacion]
            fitness = [e[0] for e in evaluaciones]
            ganancias = [e[1] for e in evaluaciones]
            flujos = [e[2] for e in evaluaciones]
            
            mejor_idx = np.argmin(fitness)
            
            if fitness[mejor_idx] < mejor_fitness_global:
                mejor_fitness_global = fitness[mejor_idx]
                mejor_global = poblacion[mejor_idx].copy()
                mejor_ganancia_global = ganancias[mejor_idx]
                mejor_flujo_global = flujos[mejor_idx]
            
            self.historial.append({
                'generacion': gen,
                'mejor_fitness': mejor_fitness_global,
                'mejor_ganancia': mejor_ganancia_global,
                'mejor_flujo': mejor_flujo_global
            })
            
            if progress_bar:
                progress_bar.progress(min((gen + 1) / self.max_generaciones, 1.0))
            if status_text:
                status_text.text(f"Gen {gen+1} | Ganancia: ${mejor_ganancia_global:,.0f} | Flujo: ${mejor_flujo_global:,.0f}")
            
            cambio_relativo = abs(fitness_anterior - mejor_fitness_global) / (abs(fitness_anterior) + 1e-10)
            
            if cambio_relativo < self.tolerancia:
                sin_mejora += 1
            else:
                sin_mejora = 0
            
            fitness_anterior = mejor_fitness_global
            
            if sin_mejora >= self.paciencia:
                if status_text:
                    status_text.text(f"✓ Convergencia en generación {gen+1}")
                break
            
            nueva_poblacion = [poblacion[mejor_idx].copy()]
            
            while len(nueva_poblacion) < self.tam_poblacion:
                padre1 = self.seleccion_torneo(poblacion, fitness)
                padre2 = self.seleccion_torneo(poblacion, fitness)
                hijo = self.cruce(padre1, padre2)
                hijo = self.mutacion(hijo)
                nueva_poblacion.append(hijo)
            
            poblacion = nueva_poblacion
        
        # Asegurar S > s
        if mejor_global['S'] <= mejor_global['s']:
            mejor_global['S'] = min(mejor_global['s'] + 1000, 15000)
        
        return mejor_global, mejor_ganancia_global, mejor_flujo_global, gen + 1


# ============================================================
# SIDEBAR - CONFIGURACIÓN
# ============================================================
st.sidebar.header("📁 1. Cargar Datos")

archivo = st.sidebar.file_uploader(
    "Subir archivo (CSV o Excel)",
    type=['csv', 'xlsx', 'xls'],
    help="El archivo debe tener columnas: dia, dia_semana, precio, demanda"
)

if archivo is None:
    st.sidebar.info("Usando datos de ejemplo (365 días)")
    usar_ejemplo = True
else:
    usar_ejemplo = False

st.sidebar.markdown("---")
st.sidebar.header("🎯 2. Objetivo")

objetivo = st.sidebar.radio(
    "¿Qué deseas optimizar?",
    options=[
        "💰 Maximizar Ganancia",
        "💵 Maximizar Flujo",
        "⚖️ Balance (ambos)"
    ],
    index=2
)

if objetivo == "⚖️ Balance (ambos)":
    balance = st.sidebar.slider(
        "Ajustar balance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="0 = Solo flujo, 1 = Solo ganancia"
    )
    w1, w2 = balance, 1 - balance
elif objetivo == "💰 Maximizar Ganancia":
    w1, w2 = 1.0, 0.0
else:
    w1, w2 = 0.0, 1.0

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 3. Parámetros")

with st.sidebar.expander("🧠 Ensemble de Redes", expanded=True):
    arquitectura_str = st.text_input("Arquitectura (capas)", value="8, 8")
    arquitectura = tuple(map(int, arquitectura_str.replace(" ", "").split(",")))
    activacion = st.selectbox("Activación", ['relu', 'tanh', 'identity'], index=0)
    n_redes = st.slider("Número de redes en ensemble", 3, 10, 5, 
                        help="Más redes = más estable pero más lento")

with st.sidebar.expander("🧬 Algoritmo Genético"):
    tam_poblacion = st.slider("Tamaño población", 20, 100, 40)
    max_generaciones = st.slider("Máx generaciones", 50, 1000, 300)
    tolerancia = st.number_input("Tolerancia convergencia", value=0.001, format="%.4f")
    paciencia = st.slider("Paciencia (gens sin mejora)", 10, 100, 30)
    semilla_ga = st.number_input("Semilla del GA", value=42, min_value=0)

with st.sidebar.expander("📦 Simulación"):
    dias_sim = st.slider("Días a simular", 7, 90, 30)
    costo_compra = st.number_input("Costo mayorista ($/L)", value=23.60, help="Costo de compra del combustible")
    h = st.number_input("Costo almacenamiento ($/L/día)", value=0.002, format="%.4f")

# ============================================================
# CONTENIDO PRINCIPAL
# ============================================================

# Cargar datos
if usar_ejemplo:
    np.random.seed(42)
    dias = 365
    
    # Precio con variación amplia
    precio_base = 23.5
    precio = precio_base + np.random.uniform(-1.5, 1.5, dias)
    
    # Demanda BASE alta
    demanda_base = 12000 + np.random.normal(0, 500, dias) + 1500 * np.sin(np.arange(dias) * 2 * np.pi / 7)
    
    # ELASTICIDAD MUY FUERTE: por cada $1 arriba, demanda baja 5000 litros
    # Esto fuerza un precio óptimo intermedio, no en el extremo
    elasticidad = -5000  # litros por dólar
    efecto_precio = elasticidad * (precio - precio_base)
    
    demanda = demanda_base + efecto_precio
    demanda = np.clip(demanda, 2000, 20000)
    
    df = pd.DataFrame({
        'dia': range(1, dias + 1),
        'dia_semana': [(i % 7) + 1 for i in range(dias)],
        'precio': precio,
        'demanda': demanda
    })
else:
    df = cargar_datos(archivo)

# Mostrar datos
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Datos Cargados")
    st.write(f"**Filas:** {len(df)}")
    st.write(f"**Columnas:** {list(df.columns)}")
    st.dataframe(df.head(10), height=300)

with col2:
    st.subheader("📈 Vista Previa")
    fig = px.line(df, x='dia', y='demanda', title='Demanda Histórica')
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Botón para ejecutar
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    ejecutar = st.button("🚀 EJECUTAR OPTIMIZACIÓN", type="primary", use_container_width=True)

if ejecutar:
    st.markdown("---")
    
    # Preparar datos
    datos = preparar_datos(df)
    
    # ============================================================
    # PASO 1: ENTRENAR ENSEMBLE
    # ============================================================
    st.subheader("🧠 Paso 1: Entrenamiento del Ensemble")
    st.write(f"Entrenando {n_redes} redes neuronales con semillas aleatorias...")
    
    progress_ensemble = st.progress(0)
    status_ensemble = st.empty()
    
    ensemble = EnsembleRedes(
        n_redes=n_redes,
        arquitectura=arquitectura,
        activacion=activacion
    )
    
    resultados_redes = []
    
    def callback_ensemble(actual, total, semilla, mse):
        progress_ensemble.progress(actual / total)
        status_ensemble.text(f"Red {actual}/{total} (semilla {semilla}): MSE={mse:.4f}")
        resultados_redes.append({'Red': actual, 'Semilla': semilla, 'MSE': mse})
    
    mse_ensemble = ensemble.entrenar(
        datos['X_train'], datos['Y_train'],
        datos['X_val'], datos['Y_val'],
        progress_callback=callback_ensemble
    )
    
    progress_ensemble.progress(1.0)
    
    # Mostrar resultados del ensemble
    col_e1, col_e2 = st.columns([1, 2])
    
    with col_e1:
        st.success(f"✓ Ensemble entrenado ({n_redes} redes)")
        st.write(f"**MSE Ensemble:** {mse_ensemble:.4f}")
        st.write(f"**MSE Promedio Individual:** {np.mean(ensemble.mse_individual):.4f}")
        st.write(f"**Semillas usadas:** {ensemble.semillas_usadas}")
    
    with col_e2:
        df_redes = pd.DataFrame(resultados_redes)
        fig_redes = px.bar(df_redes, x='Red', y='MSE', 
                          title='MSE de cada Red en el Ensemble',
                          text='Semilla')
        fig_redes.add_hline(y=mse_ensemble, line_dash="dash", line_color="red",
                           annotation_text=f"Ensemble: {mse_ensemble:.4f}")
        fig_redes.update_layout(height=250)
        st.plotly_chart(fig_redes, use_container_width=True)
    
    # ============================================================
    # PASO 2: ALGORITMO GENÉTICO
    # ============================================================
    st.markdown("---")
    st.subheader("🧬 Paso 2: Optimización con Algoritmo Genético")
    st.write(f"**Objetivo:** w1={w1:.2f} (ganancia), w2={w2:.2f} (flujo)")
    st.write(f"**Convergencia:** Sin mejora > {tolerancia} por {paciencia} generaciones")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    ga = AlgoritmoGenetico(
        ensemble,
        w1=w1, w2=w2,
        tam_poblacion=tam_poblacion,
        max_generaciones=max_generaciones,
        tolerancia=tolerancia,
        paciencia=paciencia,
        costo_compra=costo_compra,
        h=h,
        semilla_ga=semilla_ga
    )
    
    mejor, ganancia, flujo, generaciones_usadas = ga.optimizar(progress_bar, status_text)
    
    progress_bar.progress(1.0)
    
    # ============================================================
    # RESULTADOS
    # ============================================================
    st.markdown("---")
    st.subheader("🏆 Resultados de la Optimización")
    
    # DEBUG INFO
    st.info(f"🔍 **Debug:** Usando {n_redes} redes en ensemble. Semillas: {ensemble.semillas_usadas[:3]}...")
    
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    
    with col_r1:
        st.metric("s (Punto Reorden)", f"{mejor['s']:,.0f} L")
    with col_r2:
        st.metric("S (Nivel Reposición)", f"{mejor['S']:,.0f} L")
    with col_r3:
        st.metric("Precio Óptimo", f"${mejor['precio']:.2f}")
    with col_r4:
        st.metric("Generaciones", f"{generaciones_usadas}")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("💰 Ganancia Promedio/Día", f"${ganancia:,.2f}")
    with col_m2:
        st.metric("💵 Flujo Promedio/Día", f"${flujo:,.2f}")
    
    # DEBUG: Mostrar también totales
    st.caption(f"📊 Debug: Ganancia total {dias_sim} días = ${ganancia * dias_sim:,.0f} | Flujo total = ${flujo * dias_sim:,.0f}")
    
    # ============================================================
    # GRÁFICAS
    # ============================================================
    st.markdown("---")
    st.subheader("📊 Análisis de Resultados")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Evolución del AG", "📦 Simulación", "📋 Datos", "🔧 Reproducibilidad"])
    
    with tab1:
        df_hist = pd.DataFrame(ga.historial)
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Evolución de Ganancia", "Evolución de Flujo"))
        
        fig.add_trace(
            go.Scatter(x=df_hist['generacion'], y=df_hist['mejor_ganancia'], 
                      mode='lines', name='Ganancia', line=dict(color='green', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_hist['generacion'], y=df_hist['mejor_flujo'],
                      mode='lines', name='Flujo', line=dict(color='blue', width=2)),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        _, _, historico = simular_politica(
            ensemble,
            mejor['s'], mejor['S'], mejor['precio'],
            dias=dias_sim, costo_compra=costo_compra, h=h
        )
        df_sim = pd.DataFrame(historico)
        
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=("Inventario", "Demanda vs Ventas", 
                                          "Ganancia Diaria", "Flujo Diario"))
        
        fig.add_trace(
            go.Scatter(x=df_sim['dia'], y=df_sim['inventario'], mode='lines+markers', 
                      name='Inventario', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_hline(y=mejor['s'], line_dash="dash", line_color="red", row=1, col=1, 
                     annotation_text=f"s={mejor['s']:,.0f}")
        fig.add_hline(y=mejor['S'], line_dash="dash", line_color="green", row=1, col=1,
                     annotation_text=f"S={mejor['S']:,.0f}")
        
        fig.add_trace(
            go.Scatter(x=df_sim['dia'], y=df_sim['demanda'], mode='lines', 
                      name='Demanda', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=df_sim['dia'], y=df_sim['ventas'], mode='lines', 
                      name='Ventas', line=dict(color='green')),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=df_sim['dia'], y=df_sim['ganancia'], name='Ganancia', marker_color='green'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=df_sim['dia'], y=df_sim['flujo'], name='Flujo', marker_color='blue'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.write("**Histórico de Simulación:**")
        st.dataframe(df_sim, height=400)
        
        csv = df_sim.to_csv(index=False)
        st.download_button(
            label="📥 Descargar CSV",
            data=csv,
            file_name="simulacion_optima.csv",
            mime="text/csv"
        )
    
    with tab4:
        st.write("### 🔧 Información para Reproducibilidad")
        st.write("Usa estos parámetros para obtener resultados similares:")
        
        codigo = f"""
# Configuración ENSEMBLE

# Redes Neuronales
n_redes = {n_redes}
arquitectura = {arquitectura}
activacion = '{activacion}'
semillas_redes = {ensemble.semillas_usadas}

# Algoritmo Genético  
w1, w2 = {w1}, {w2}
tam_poblacion = {tam_poblacion}
max_generaciones = {max_generaciones}
tolerancia = {tolerancia}
paciencia = {paciencia}
semilla_ga = {semilla_ga}

# Resultados obtenidos:
# s = {mejor['s']:,.2f} litros
# S = {mejor['S']:,.2f} litros
# precio = ${mejor['precio']:.2f}
# ganancia_promedio = ${ganancia:,.2f}
# flujo_promedio = ${flujo:,.2f}
# mse_ensemble = {mse_ensemble:.4f}
"""
        st.code(codigo, language='python')
        
        st.download_button(
            label="📥 Descargar configuración",
            data=codigo,
            file_name="configuracion_ensemble.py",
            mime="text/plain"
        )
        
        # Ejemplo de predicción con detalle
        st.markdown("---")
        st.write("### 🔍 Ejemplo de Predicción del Ensemble")
        detalle = ensemble.predecir_con_detalle(1, 10000, 23.5)
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.write("**Input:** día_semana=1, demanda_ant=10,000, precio=$23.50")
            st.write(f"**Predicción (promedio):** {detalle['promedio']:,.0f} litros")
            st.write(f"**Desv. estándar:** {detalle['std']:,.0f} litros")
            st.write(f"**Rango:** [{detalle['min']:,.0f}, {detalle['max']:,.0f}]")
        
        with col_d2:
            df_pred = pd.DataFrame({
                'Red': [f"Red {i+1}" for i in range(len(detalle['individuales']))],
                'Predicción': detalle['individuales']
            })
            fig_pred = px.bar(df_pred, x='Red', y='Predicción', title='Predicción de cada red')
            fig_pred.add_hline(y=detalle['promedio'], line_dash="dash", line_color="red",
                              annotation_text=f"Promedio: {detalle['promedio']:,.0f}")
            fig_pred.update_layout(height=250)
            st.plotly_chart(fig_pred, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*Desarrollado con Streamlit + scikit-learn | Versión Ensemble v2.1*")
