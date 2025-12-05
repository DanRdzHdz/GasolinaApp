"""
OPTIMIZACIÓN DE POLÍTICA DE INVENTARIOS - VERSIÓN ENSEMBLE
===========================================================
- Ensemble de redes neuronales (más estable y honesto)
- Promedia predicciones de múltiples redes
- Algoritmo Genético con convergencia automática
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN - MODIFICA AQUÍ
# ============================================================

# Archivo de datos
ARCHIVO_DATOS = 'datos_fase3.csv'

# Objetivo: ajustar pesos
W1 = 0.5  # Peso ganancia
W2 = 0.5  # Peso flujo

# Red Neuronal - ENSEMBLE
ARQUITECTURA = (8, 8)
ACTIVACION = 'relu'
N_REDES_ENSEMBLE = 5  # Cuántas redes en el ensemble (3-5 recomendado)

# Algoritmo Genético
TAM_POBLACION = 40
MAX_GENERACIONES = 300
TOLERANCIA = 0.001
PACIENCIA = 30
SEMILLA_GA = 42

# Simulación
DIAS_SIMULACION = 30
COSTO_COMPRA = 20.0
H = 0.002

# ============================================================
# FUNCIONES
# ============================================================

def cargar_y_preparar_datos(archivo):
    """Carga y prepara los datos"""
    df = pd.read_csv(archivo)
    
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
        'df': df,
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
    
    def entrenar(self, X_train, Y_train, X_val, Y_val):
        """Entrena el ensemble de redes"""
        print(f"\n{'='*60}")
        print(f"ENTRENANDO ENSEMBLE ({self.n_redes} redes)")
        print(f"{'='*60}")
        
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
        
        print(f"Semillas: {self.semillas_usadas}\n")
        
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
            
            print(f"  Red {i+1}/{self.n_redes} (semilla {semilla}): MSE={mse:.4f}")
        
        # Calcular MSE del ensemble
        Y_pred_ensemble = self._predecir_scaled(X_val_scaled)
        mse_ensemble = mean_squared_error(Y_val_scaled, Y_pred_ensemble)
        
        print(f"\n*** MSE ENSEMBLE (promedio): {mse_ensemble:.4f} ***")
        print(f"    vs MSE individual promedio: {np.mean(self.mse_individual):.4f}")
        print(f"    vs MSE mejor individual: {np.min(self.mse_individual):.4f}")
        
        return mse_ensemble
    
    def _predecir_scaled(self, X_scaled):
        """Predice promediando todas las redes (datos ya escalados)"""
        predicciones = np.array([red.predict(X_scaled) for red in self.redes])
        return np.mean(predicciones, axis=0)
    
    def predecir(self, dia_semana, demanda_anterior, precio):
        """Predice demanda promediando todas las redes"""
        X_nuevo = np.array([[dia_semana, demanda_anterior, precio]])
        X_scaled = self.scaler_X.transform(X_nuevo)
        
        # Obtener predicción de cada red
        predicciones = []
        for red in self.redes:
            Y_pred_scaled = red.predict(X_scaled)
            Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled.reshape(-1, 1)).ravel()[0]
            predicciones.append(Y_pred)
        
        # Devolver promedio
        return np.mean(predicciones)
    
    def predecir_con_detalle(self, dia_semana, demanda_anterior, precio):
        """Predice y muestra el detalle de cada red"""
        X_nuevo = np.array([[dia_semana, demanda_anterior, precio]])
        X_scaled = self.scaler_X.transform(X_nuevo)
        
        predicciones = []
        for i, red in enumerate(self.redes):
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
                     dias=30, costo_compra=20.0, h=0.002,
                     inventario_inicial=10000, dia_semana_inicial=2, 
                     demanda_inicial=10000):
    """Simula la política de inventarios usando el ensemble"""
    inventario = inventario_inicial
    dia_semana = dia_semana_inicial
    demanda_anterior = demanda_inicial
    
    ganancias = []
    flujos = []
    historico = []
    
    for dia in range(1, dias + 1):
        # Predicción usando ensemble (promedio de todas las redes)
        demanda = ensemble.predecir(dia_semana, demanda_anterior, precio_venta)
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


class AlgoritmoGenetico:
    """Algoritmo Genético con convergencia automática"""
    
    def __init__(self, ensemble, w1=0.5, w2=0.5,
                 tam_poblacion=40, max_generaciones=300,
                 tolerancia=0.001, paciencia=30,
                 prob_mutacion=0.1, prob_cruce=0.8,
                 costo_compra=20.0, h=0.002, semilla=None):
        
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
        
        if semilla is not None:
            np.random.seed(semilla)
        
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
    
    def optimizar(self):
        """Ejecuta el AG con criterio de convergencia"""
        print(f"\n{'='*60}")
        print("EJECUTANDO ALGORITMO GENÉTICO")
        print(f"{'='*60}")
        print(f"Objetivo: w1={self.w1} (ganancia), w2={self.w2} (flujo)")
        print(f"Convergencia: tolerancia={self.tolerancia}, paciencia={self.paciencia}")
        print()
        
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
            
            if (gen + 1) % 20 == 0:
                print(f"  Gen {gen+1:3d}: Ganancia=${mejor_ganancia_global:,.0f}, Flujo=${mejor_flujo_global:,.0f}")
            
            cambio_relativo = abs(fitness_anterior - mejor_fitness_global) / (abs(fitness_anterior) + 1e-10)
            
            if cambio_relativo < self.tolerancia:
                sin_mejora += 1
            else:
                sin_mejora = 0
            
            fitness_anterior = mejor_fitness_global
            
            if sin_mejora >= self.paciencia:
                print(f"\n  ✓ CONVERGENCIA en generación {gen+1}")
                break
            
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


def graficar_resultados(historial_ga, historico_sim, mejor, ensemble):
    """Genera todas las gráficas"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Evolución del GA - Ganancia
    ax1 = fig.add_subplot(2, 3, 1)
    df_hist = pd.DataFrame(historial_ga)
    ax1.plot(df_hist['generacion'], df_hist['mejor_ganancia'], 'g-', linewidth=2)
    ax1.set_xlabel('Generación')
    ax1.set_ylabel('Ganancia ($)')
    ax1.set_title('Evolución de Ganancia (AG)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Evolución del GA - Flujo
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(df_hist['generacion'], df_hist['mejor_flujo'], 'b-', linewidth=2)
    ax2.set_xlabel('Generación')
    ax2.set_ylabel('Flujo ($)')
    ax2.set_title('Evolución de Flujo (AG)')
    ax2.grid(True, alpha=0.3)
    
    # 3. MSE de cada red del ensemble
    ax3 = fig.add_subplot(2, 3, 3)
    colores = ['green' if mse == min(ensemble.mse_individual) else 'steelblue' 
               for mse in ensemble.mse_individual]
    bars = ax3.bar(range(len(ensemble.mse_individual)), ensemble.mse_individual, color=colores)
    ax3.axhline(y=np.mean(ensemble.mse_individual), color='red', linestyle='--', 
                label=f'Promedio: {np.mean(ensemble.mse_individual):.4f}')
    ax3.set_xlabel('Red')
    ax3.set_ylabel('MSE')
    ax3.set_title(f'MSE de cada Red en Ensemble ({len(ensemble.redes)} redes)')
    ax3.set_xticks(range(len(ensemble.mse_individual)))
    ax3.set_xticklabels([f'Red {i+1}\n(s={s})' for i, s in enumerate(ensemble.semillas_usadas)], fontsize=8)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Simulación - Inventario
    df_sim = pd.DataFrame(historico_sim)
    
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(df_sim['dia'], df_sim['inventario'], 'b-', linewidth=2, marker='o', markersize=4)
    ax4.axhline(y=mejor['s'], color='red', linestyle='--', linewidth=2, label=f"s={mejor['s']:,.0f}")
    ax4.axhline(y=mejor['S'], color='green', linestyle='--', linewidth=2, label=f"S={mejor['S']:,.0f}")
    ax4.set_xlabel('Día')
    ax4.set_ylabel('Inventario (L)')
    ax4.set_title('Nivel de Inventario')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Demanda vs Ventas
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(df_sim['dia'], df_sim['demanda'], 'b-', linewidth=2, label='Demanda', marker='o', markersize=4)
    ax5.plot(df_sim['dia'], df_sim['ventas'], 'g--', linewidth=2, label='Ventas', marker='s', markersize=4)
    ax5.set_xlabel('Día')
    ax5.set_ylabel('Litros')
    ax5.set_title('Demanda vs Ventas')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Ganancia y Flujo diario
    ax6 = fig.add_subplot(2, 3, 6)
    x = np.arange(len(df_sim))
    width = 0.35
    ax6.bar(x - width/2, df_sim['ganancia'], width, label='Ganancia', color='green', alpha=0.7)
    ax6.bar(x + width/2, df_sim['flujo'], width, label='Flujo', color='blue', alpha=0.7)
    ax6.set_xlabel('Día')
    ax6.set_ylabel('$')
    ax6.set_title('Ganancia y Flujo Diario')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('resultados_ensemble.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nGráfica guardada: resultados_ensemble.png")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":
    
    print("="*60)
    print("OPTIMIZACIÓN DE INVENTARIOS - VERSIÓN ENSEMBLE")
    print("="*60)
    print(f"\nConfiguración:")
    print(f"  - Archivo: {ARCHIVO_DATOS}")
    print(f"  - Objetivo: w1={W1} (ganancia), w2={W2} (flujo)")
    print(f"  - Arquitectura: {ARQUITECTURA}, {ACTIVACION}")
    print(f"  - Redes en ensemble: {N_REDES_ENSEMBLE}")
    
    # 1. Cargar datos
    print(f"\n{'='*60}")
    print("CARGANDO DATOS")
    print(f"{'='*60}")
    
    datos = cargar_y_preparar_datos(ARCHIVO_DATOS)
    print(f"  Datos: {len(datos['df'])} filas")
    print(f"  Train: {len(datos['X_train'])}, Val: {len(datos['X_val'])}, Test: {len(datos['X_test'])}")
    
    # 2. Crear y entrenar ensemble
    ensemble = EnsembleRedes(
        n_redes=N_REDES_ENSEMBLE,
        arquitectura=ARQUITECTURA,
        activacion=ACTIVACION
    )
    
    mse_ensemble = ensemble.entrenar(
        datos['X_train'], datos['Y_train'],
        datos['X_val'], datos['Y_val']
    )
    
    # 3. Ejecutar Algoritmo Genético
    ga = AlgoritmoGenetico(
        ensemble,
        w1=W1, w2=W2,
        tam_poblacion=TAM_POBLACION,
        max_generaciones=MAX_GENERACIONES,
        tolerancia=TOLERANCIA,
        paciencia=PACIENCIA,
        costo_compra=COSTO_COMPRA,
        h=H,
        semilla=SEMILLA_GA
    )
    
    mejor, ganancia, flujo, generaciones = ga.optimizar()
    
    # 4. Simular con política óptima
    _, _, historico_sim = simular_politica(
        ensemble,
        mejor['s'], mejor['S'], mejor['precio'],
        dias=DIAS_SIMULACION,
        costo_compra=COSTO_COMPRA,
        h=H
    )
    
    # 5. Mostrar resultados
    print(f"\n{'='*60}")
    print("RESULTADOS FINALES")
    print(f"{'='*60}")
    print(f"\n*** POLÍTICA ÓPTIMA ***")
    print(f"  s (punto reorden):    {mejor['s']:,.2f} litros")
    print(f"  S (nivel reposición): {mejor['S']:,.2f} litros")
    print(f"  Precio:               ${mejor['precio']:.2f}")
    print(f"\n*** MÉTRICAS ***")
    print(f"  Ganancia promedio:    ${ganancia:,.2f}")
    print(f"  Flujo promedio:       ${flujo:,.2f}")
    print(f"\n*** ENSEMBLE INFO ***")
    print(f"  Redes usadas:         {N_REDES_ENSEMBLE}")
    print(f"  Semillas:             {ensemble.semillas_usadas}")
    print(f"  MSE ensemble:         {mse_ensemble:.4f}")
    print(f"  Generaciones usadas:  {generaciones}")
    
    # 6. Ejemplo de predicción con detalle
    print(f"\n*** EJEMPLO DE PREDICCIÓN ***")
    detalle = ensemble.predecir_con_detalle(1, 10000, 23.5)
    print(f"  Input: dia_semana=1, demanda_ant=10000, precio=23.5")
    print(f"  Predicciones individuales: {[f'{p:.0f}' for p in detalle['individuales']]}")
    print(f"  Promedio (usado): {detalle['promedio']:,.0f}")
    print(f"  Desv. estándar:   {detalle['std']:,.0f}")
    print(f"  Rango:            [{detalle['min']:,.0f}, {detalle['max']:,.0f}]")
    
    # 7. Graficar
    print(f"\n{'='*60}")
    print("GENERANDO GRÁFICAS...")
    print(f"{'='*60}")
    
    graficar_resultados(ga.historial, historico_sim, mejor, ensemble)
    
    # 8. Guardar resultados
    df_sim = pd.DataFrame(historico_sim)
    df_sim.to_csv('simulacion_ensemble.csv', index=False)
    print("Simulación guardada: simulacion_ensemble.csv")
    
    # Guardar configuración
    config = f"""# Configuración ENSEMBLE para reproducir resultados
# ================================================

# Ensemble de Redes
N_REDES = {N_REDES_ENSEMBLE}
ARQUITECTURA = {ARQUITECTURA}
ACTIVACION = '{ACTIVACION}'
SEMILLAS_REDES = {ensemble.semillas_usadas}

# Algoritmo Genético
W1, W2 = {W1}, {W2}
TAM_POBLACION = {TAM_POBLACION}
TOLERANCIA = {TOLERANCIA}
PACIENCIA = {PACIENCIA}
SEMILLA_GA = {SEMILLA_GA}

# Resultados
# s = {mejor['s']:,.2f}
# S = {mejor['S']:,.2f}
# precio = {mejor['precio']:.2f}
# ganancia_promedio = {ganancia:,.2f}
# flujo_promedio = {flujo:,.2f}
# mse_ensemble = {mse_ensemble:.4f}
"""
    
    with open('configuracion_ensemble.txt', 'w') as f:
        f.write(config)
    print("Configuración guardada: configuracion_ensemble.txt")
    
    print(f"\n{'='*60}")
    print("¡OPTIMIZACIÓN COMPLETADA!")
    print(f"{'='*60}")
