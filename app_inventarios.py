"""
APLICACIÓN DE OPTIMIZACIÓN DE INVENTARIOS
==========================================
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

@st.cache_resource
def entrenar_red(X_train, Y_train, arquitectura, activacion):
    """Entrena la red neuronal"""
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).ravel()
    
    mlp = MLPRegressor(
        hidden_layer_sizes=arquitectura,
        activation=activacion,
        solver='adam',
        max_iter=3000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=50,
        verbose=False
    )
    mlp.fit(X_train_scaled, Y_train_scaled)
    
    return mlp, scaler_X, scaler_Y

def predecir_demanda(mlp, scaler_X, scaler_Y, dia_semana, demanda_anterior, precio):
    """Predice demanda usando la red neuronal"""
    X_nuevo = np.array([[dia_semana, demanda_anterior, precio]])
    X_scaled = scaler_X.transform(X_nuevo)
    Y_pred_scaled = mlp.predict(X_scaled)
    return scaler_Y.inverse_transform(Y_pred_scaled.reshape(-1, 1)).ravel()[0]

def simular_politica(mlp, scaler_X, scaler_Y, s, S, precio_venta, 
                     dias=30, costo_compra=20.0, h=0.002,
                     inventario_inicial=10000, dia_semana_inicial=2, 
                     demanda_inicial=10000):
    """Simula la política de inventarios"""
    inventario = inventario_inicial
    dia_semana = dia_semana_inicial
    demanda_anterior = demanda_inicial
    
    ganancias = []
    flujos = []
    historico = []
    
    for dia in range(1, dias + 1):
        demanda = predecir_demanda(mlp, scaler_X, scaler_Y, dia_semana, demanda_anterior, precio_venta)
        demanda = max(0, demanda)
        
        ventas = min(demanda, inventario)
        inventario -= ventas
        
        compras = 0
        if inventario <= s:
            compras = S - inventario
            inventario = S
        
        ingresos = ventas * precio_venta
        costo_compras = compras * costo_compra
        costo_inventario = h * inventario
        
        ganancia = ventas * (precio_venta - costo_compra)
        ganancias.append(ganancia)
        
        flujo = ingresos - costo_compras - costo_inventario
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
    
    return np.mean(ganancias), np.mean(flujos), historico


class AlgoritmoGeneticoConvergencia:
    """AG con parada por convergencia"""
    
    def __init__(self, mlp, scaler_X, scaler_Y, w1=0.5, w2=0.5,
                 tam_poblacion=40, max_generaciones=500,
                 tolerancia=0.001, paciencia=30,
                 prob_mutacion=0.1, prob_cruce=0.8,
                 costo_compra=20.0, h=0.002):
        
        self.mlp = mlp
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
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
        
        self.limites = {
            's': (1000, 8000),
            'S': (10000, 25000),
            'precio': (22.0, 26.0)
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
        S = max(s + 1000, individuo['S'])
        precio = individuo['precio']
        
        ganancia, flujo, _ = simular_politica(
            self.mlp, self.scaler_X, self.scaler_Y,
            s, S, precio,
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
            # Evaluar población
            evaluaciones = [self.evaluar(ind) for ind in poblacion]
            fitness = [e[0] for e in evaluaciones]
            ganancias = [e[1] for e in evaluaciones]
            flujos = [e[2] for e in evaluaciones]
            
            # Mejor de la generación
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
            
            # Actualizar progreso
            if progress_bar:
                progress_bar.progress(min((gen + 1) / self.max_generaciones, 1.0))
            if status_text:
                status_text.text(f"Generación {gen+1} | Ganancia: ${mejor_ganancia_global:,.0f} | Flujo: ${mejor_flujo_global:,.0f}")
            
            # CRITERIO DE CONVERGENCIA
            cambio_relativo = abs(fitness_anterior - mejor_fitness_global) / (abs(fitness_anterior) + 1e-10)
            
            if cambio_relativo < self.tolerancia:
                sin_mejora += 1
            else:
                sin_mejora = 0
            
            fitness_anterior = mejor_fitness_global
            
            # ¿Convergió?
            if sin_mejora >= self.paciencia:
                if status_text:
                    status_text.text(f"✓ Convergencia en generación {gen+1}")
                break
            
            # Nueva población
            nueva_poblacion = [poblacion[mejor_idx].copy()]
            
            while len(nueva_poblacion) < self.tam_poblacion:
                padre1 = self.seleccion_torneo(poblacion, fitness)
                padre2 = self.seleccion_torneo(poblacion, fitness)
                hijo = self.cruce(padre1, padre2)
                hijo = self.mutacion(hijo)
                nueva_poblacion.append(hijo)
            
            poblacion = nueva_poblacion
        
        mejor_global['S'] = max(mejor_global['s'] + 1000, mejor_global['S'])
        
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

# Datos de ejemplo si no hay archivo
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

# Slider para balance personalizado
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

with st.sidebar.expander("Red Neuronal"):
    arquitectura_str = st.text_input("Arquitectura (capas)", value="8, 8")
    arquitectura = tuple(map(int, arquitectura_str.replace(" ", "").split(",")))
    activacion = st.selectbox("Activación", ['relu', 'tanh', 'identity'], index=0)

with st.sidebar.expander("Algoritmo Genético"):
    tam_poblacion = st.slider("Tamaño población", 20, 100, 40)
    max_generaciones = st.slider("Máx generaciones", 50, 1000, 300)
    tolerancia = st.number_input("Tolerancia convergencia", value=0.001, format="%.4f")
    paciencia = st.slider("Paciencia (generaciones sin mejora)", 10, 100, 30)

with st.sidebar.expander("Simulación"):
    dias_sim = st.slider("Días a simular", 7, 90, 30)
    costo_compra = st.number_input("Costo de compra ($/L)", value=20.0)
    h = st.number_input("Costo almacenamiento ($/L/día)", value=0.002, format="%.4f")

# ============================================================
# CONTENIDO PRINCIPAL
# ============================================================

# Cargar datos
if usar_ejemplo:
    # Generar datos de ejemplo
    np.random.seed(42)
    dias = 365
    df = pd.DataFrame({
        'dia': range(1, dias + 1),
        'dia_semana': [(i % 7) + 1 for i in range(dias)],
        'precio': 23.5 + np.cumsum(np.random.normal(0, 0.05, dias)),
        'demanda': 10000 + np.random.normal(0, 2000, dias) + 2000 * np.sin(np.arange(dias) * 2 * np.pi / 7)
    })
    df['demanda'] = df['demanda'].clip(3000, 18000)
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
    df['demanda_t_minus_1'] = df['demanda'].shift(1)
    df['demanda_t_plus_1'] = df['demanda'].shift(-1)
    df_clean = df.dropna().reset_index(drop=True)
    
    X = df_clean[['dia_semana', 'demanda_t_minus_1', 'precio']].values
    Y = df_clean['demanda_t_plus_1'].values
    
    n = len(X)
    train_end = int(n * 0.70)
    X_train, Y_train = X[:train_end], Y[:train_end]
    
    # Entrenar red
    with st.spinner("🧠 Entrenando red neuronal..."):
        mlp, scaler_X, scaler_Y = entrenar_red(X_train, Y_train, arquitectura, activacion)
    
    st.success(f"✓ Red neuronal entrenada: {arquitectura}, {activacion}")
    
    # Ejecutar AG
    st.subheader("🧬 Optimización con Algoritmo Genético")
    st.write(f"**Objetivo:** w1={w1:.2f} (ganancia), w2={w2:.2f} (flujo)")
    st.write(f"**Criterio de parada:** Sin mejora > {tolerancia} por {paciencia} generaciones")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    ga = AlgoritmoGeneticoConvergencia(
        mlp, scaler_X, scaler_Y,
        w1=w1, w2=w2,
        tam_poblacion=tam_poblacion,
        max_generaciones=max_generaciones,
        tolerancia=tolerancia,
        paciencia=paciencia,
        costo_compra=costo_compra,
        h=h
    )
    
    mejor, ganancia, flujo, generaciones_usadas = ga.optimizar(progress_bar, status_text)
    
    progress_bar.progress(1.0)
    
    # Resultados
    st.markdown("---")
    st.subheader("🏆 Resultados de la Optimización")
    
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
        st.metric("💰 Ganancia Promedio", f"${ganancia:,.2f}")
    with col_m2:
        st.metric("💵 Flujo Promedio", f"${flujo:,.2f}")
    
    # Gráficas
    st.markdown("---")
    st.subheader("📊 Análisis de Resultados")
    
    tab1, tab2, tab3 = st.tabs(["📈 Evolución del AG", "📦 Simulación", "📋 Datos"])
    
    with tab1:
        df_hist = pd.DataFrame(ga.historial)
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Evolución de Ganancia", "Evolución de Flujo"))
        
        fig.add_trace(
            go.Scatter(x=df_hist['generacion'], y=df_hist['mejor_ganancia'], 
                      mode='lines', name='Ganancia', line=dict(color='green')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_hist['generacion'], y=df_hist['mejor_flujo'],
                      mode='lines', name='Flujo', line=dict(color='blue')),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Simular con política óptima
        _, _, historico = simular_politica(
            mlp, scaler_X, scaler_Y,
            mejor['s'], mejor['S'], mejor['precio'],
            dias=dias_sim, costo_compra=costo_compra, h=h
        )
        df_sim = pd.DataFrame(historico)
        
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=("Inventario", "Demanda vs Ventas", 
                                          "Ganancia Diaria", "Flujo Diario"))
        
        # Inventario
        fig.add_trace(
            go.Scatter(x=df_sim['dia'], y=df_sim['inventario'], mode='lines+markers', name='Inventario'),
            row=1, col=1
        )
        fig.add_hline(y=mejor['s'], line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=mejor['S'], line_dash="dash", line_color="green", row=1, col=1)
        
        # Demanda vs Ventas
        fig.add_trace(
            go.Scatter(x=df_sim['dia'], y=df_sim['demanda'], mode='lines', name='Demanda'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=df_sim['dia'], y=df_sim['ventas'], mode='lines', name='Ventas'),
            row=1, col=2
        )
        
        # Ganancia
        fig.add_trace(
            go.Bar(x=df_sim['dia'], y=df_sim['ganancia'], name='Ganancia', marker_color='green'),
            row=2, col=1
        )
        
        # Flujo
        fig.add_trace(
            go.Bar(x=df_sim['dia'], y=df_sim['flujo'], name='Flujo', marker_color='blue'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.write("**Histórico de Simulación:**")
        st.dataframe(df_sim, height=400)
        
        # Descargar resultados
        csv = df_sim.to_csv(index=False)
        st.download_button(
            label="📥 Descargar CSV",
            data=csv,
            file_name="simulacion_optima.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("*Desarrollado con Streamlit + scikit-learn*")
