# Predicción de la suscripción a depósitos a plazo

Proyecto integrador de **Aprendizaje Automático y Minería de Datos** orientado a predecir si un cliente contratará un depósito a plazo antes de realizar una llamada de marketing.

[![Abrir en Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ritterb64/proyectomodulo5/blob/main/Proyecto_Bank_Marketing_Colab_Grupo1.ipynb)
[![Dataset UCI](https://img.shields.io/badge/Dataset-UCI%20Bank%20Marketing-2F80ED)](https://archive.ics.uci.edu/dataset/222/bank+marketing)

## Descripción del proyecto

Las campañas telefónicas pueden consumir una cantidad importante de tiempo y recursos cuando los clientes se contactan sin una selección previa. Este proyecto analiza información demográfica, financiera y de campañas anteriores para identificar clientes con mayor probabilidad de contratar un depósito a plazo.

El problema se aborda como una **clasificación binaria supervisada**:

- `yes` (`1`): el cliente contrató el depósito.
- `no` (`0`): el cliente no contrató el depósito.

La variable objetivo es **`y`**.

### Pregunta de investigación

> ¿En qué medida las características demográficas, financieras y/o el historial de campañas permiten predecir si un cliente de la institución bancaria contratará un depósito a plazo, sin utilizar información que solamente se conoce después de finalizar la llamada?

## Objetivos

### Objetivo general

Desarrollar y comparar modelos de clasificación que permitan anticipar si un cliente podría contratar un depósito a plazo.

### Objetivos específicos

- Revisar la estructura y la calidad de los datos.
- Explorar las variables demográficas, financieras y de campaña.
- Preparar las variables numéricas y categóricas para el modelado.
- Entrenar y comparar una regresión logística y árboles de decisión.
- Evaluar los modelos mediante accuracy, precision, recall, F1 y matrices de confusión.
- Analizar el umbral de decisión, el desbalance de clases y el sobreajuste.
- Identificar limitaciones, riesgos éticos y posibles fugas de información.

## Dataset

Se utilizó el archivo `bank-full.csv` del conjunto público [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), disponible en el UCI Machine Learning Repository.

| Característica | Descripción |
|---|---|
| Institución | Entidad bancaria portuguesa |
| Periodo | Mayo de 2008 a noviembre de 2010 |
| Registros | 45.211 |
| Columnas originales | 17 |
| Variable objetivo | `y` (`yes` / `no`) |
| No contrató | 39.922 registros (88,3 %) |
| Sí contrató | 5.289 registros (11,7 %) |
| Licencia del dataset | CC BY 4.0 |

El conjunto presenta un desbalance importante: solamente el 11,7 % de los clientes contrató. Por este motivo, la selección de modelos no se basa únicamente en el accuracy; el **F1 de la clase positiva** se utiliza como criterio principal.

La duración promedio de las llamadas es de **258,16 segundos**, aproximadamente **4 minutos con 18 segundos**.

## Calidad y preparación de los datos

La revisión inicial produjo los siguientes resultados:

- No existen valores `NaN`.
- No existen textos vacíos.
- No existen filas duplicadas exactas.
- Algunas variables categóricas contienen el valor `unknown`.

`unknown` se conserva como una categoría independiente porque representa información no disponible. No se reemplaza por la moda ni se trata como un valor nulo, ya que hacerlo introduciría una suposición no sustentada por los datos. Por la misma razón, el pipeline no aplica imputación.

También se creó `previously_contacted` para distinguir si el cliente había sido contactado anteriormente. Cuando no existe historial previo, el dataset lo representa mediante `pdays = -1`, `previous = 0` y, normalmente, `poutcome = unknown`.

### Prevención de fuga de información

La variable `duration` indica la duración de la llamada, pero solo se conoce después de que el contacto terminó. Como el propósito es priorizar clientes **antes de llamar**, esta variable se excluye del modelo operativo para evitar fuga de información.

## Metodología

1. Definición del problema y de la variable objetivo.
2. Carga y revisión de calidad del dataset.
3. Análisis exploratorio y estadística descriptiva.
4. Conversión de `y` a formato binario.
5. Exclusión de `duration` y creación de `previously_contacted`.
6. División estratificada en entrenamiento, validación y prueba.
7. Transformación de variables mediante pipelines.
8. Entrenamiento de regresiones logísticas y árboles de decisión.
9. Selección de configuraciones con el conjunto de validación.
10. Evaluación independiente con el conjunto de prueba.

### División de los datos

| Conjunto | Registros | Proporción | Clase `yes` |
|---|---:|---:|---:|
| Entrenamiento | 27.126 | 60 % | 11,7 % |
| Validación | 9.042 | 20 % | 11,7 % |
| Prueba | 9.043 | 20 % | 11,7 % |

La división se realizó con `random_state=42` y estratificación. El conjunto de entrenamiento permite aprender los parámetros; el de validación permite elegir el umbral y comparar configuraciones; el de prueba se reserva para la evaluación final.

### Preprocesamiento

- **Regresión logística:** estandarización de variables numéricas con `StandardScaler`.
- **Árboles:** variables numéricas en su escala original.
- **Variables categóricas:** conversión mediante `OneHotEncoder(handle_unknown="ignore")`.
- **Aplicación por tipo:** `ColumnTransformer` y `Pipeline` de scikit-learn.

## Modelos evaluados

- Regresión logística con umbral estándar de 0,50.
- Regresión logística con umbral seleccionado en validación.
- Regresión logística con `class_weight="balanced"`.
- Árbol de decisión sin balanceo.
- Árbol de decisión con `class_weight="balanced"`.

Para el árbol se evaluaron profundidades **2, 6, 8, 10 y 12**. Aunque la profundidad 12 alcanzó un F1 de validación ligeramente superior, se eligió la profundidad 8 como compromiso entre desempeño, sencillez y menor riesgo de sobreajuste.

En la regresión logística se probaron umbrales desde **0,05 hasta 0,90**, con incrementos de 0,01. El umbral final de **0,18** fue seleccionado exclusivamente por obtener el mayor F1 en validación.

## Resultados

| Modelo | Configuración | F1 validación | Precision prueba | Recall prueba | F1 prueba |
|---|---|---:|---:|---:|---:|
| **Regresión logística** | **Umbral 0,18** | **0,457** | **0,416** | **0,466** | **0,440** |
| Árbol de decisión | Profundidad 8 + balanceo | 0,437 | 0,341 | 0,547 | 0,420 |
| Regresión logística | `class_weight="balanced"` | 0,375 | 0,266 | 0,629 | 0,373 |
| Árbol de decisión | Profundidad 8 | 0,330 | 0,625 | 0,205 | 0,309 |
| Regresión logística | Umbral 0,50 | 0,294 | 0,664 | 0,174 | 0,276 |

### Modelo seleccionado

El modelo final es la **regresión logística con umbral validado de 0,18**, porque obtuvo el mayor F1 en validación. En el conjunto de prueba alcanzó:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,861 |
| Precision | 0,416 |
| Recall | 0,466 |
| F1 | 0,440 |

Su matriz de confusión en prueba fue:

| Valor real / Predicción | No contrató | Sí contrató |
|---|---:|---:|
| No contrató | 7.293 | 692 |
| Sí contrató | 565 | 493 |

En términos prácticos, el modelo identifica aproximadamente a **47 de cada 100 clientes** que realmente contratarían. De cada 100 clientes señalados como posibles contratantes, aproximadamente **42 contratarían**. El umbral menor aumenta la detección de clientes interesados, aunque también genera más falsos positivos y posibles llamadas innecesarias.

El árbol balanceado obtuvo un recall superior (0,547), pero un F1 menor (0,420). Podría considerarse si el objetivo operativo priorizara encontrar más clientes interesados, incluso aceptando una mayor cantidad de contactos sin contratación.

## Variables destacadas por el árbol

El árbol balanceado de profundidad 8 se utilizó como apoyo interpretativo. Sus cinco variables transformadas con mayor importancia fueron:

| Variable | Importancia |
|---|---:|
| `poutcome_success` | 0,2867 |
| `contact_unknown` | 0,1597 |
| `housing_yes` | 0,0784 |
| `day` | 0,0713 |
| `balance` | 0,0501 |

Estas importancias describen cómo el árbol separó los registros; no demuestran relaciones causales.

## Experimento con `duration`

Como comprobación metodológica se entrenó una regresión logística equivalente incluyendo `duration` y manteniendo el umbral 0,50:

| Escenario | F1 prueba |
|---|---:|
| Modelo operativo sin `duration` | 0,276 |
| Modelo de referencia con `duration` | 0,453 |

La mejora muestra cuánto puede elevarse el resultado al usar información posterior a la llamada. Esta versión no se selecciona porque no puede utilizarse para decidir previamente a quién contactar.

## Estructura recomendada del repositorio

```text
proyectomodulo5/
├── README.md
├── Proyecto_Bank_Marketing_Colab_Grupo1.ipynb
├── Informe_Tecnico_Bank_Marketing_Grupo1_Final.pdf
├── bank-full.csv
└── requirements.txt
```

## Ejecución en Google Colab

1. Abrir el [notebook en Google Colab](https://colab.research.google.com/github/ritterb64/proyectomodulo5/blob/main/Proyecto_Bank_Marketing_Colab_Grupo1.ipynb).
2. Descargar `bank-full.csv` desde UCI o desde el repositorio.
3. Subir el archivo a la carpeta principal de la sesión de Colab con el nombre exacto `bank-full.csv`.
4. Verificar que la ruta sea `/content/bank-full.csv`.
5. Seleccionar **Entorno de ejecución > Ejecutar todas**.

> Los archivos almacenados en `/content` se eliminan cuando termina la sesión de Colab. Si se reinicia el entorno, el CSV debe cargarse nuevamente.

## Ejecución local

```bash
git clone https://github.com/ritterb64/proyectomodulo5.git
cd proyectomodulo5
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Proyecto_Bank_Marketing_Colab_Grupo1.ipynb
```

En Windows, la activación del entorno virtual se realiza con:

```powershell
.venv\Scripts\activate
```

Para ejecutar el notebook localmente, debe cambiarse la ruta de carga del CSV por:

```python
df = pd.read_csv("bank-full.csv", sep=";")
```

### Dependencias principales

- Python 3.10 o superior.
- NumPy.
- pandas.
- Matplotlib.
- seaborn.
- scikit-learn.
- Jupyter Notebook o Google Colab.

## Limitaciones y consideraciones éticas

- Los datos pertenecen a una sola institución portuguesa y al periodo 2008–2010.
- El comportamiento de clientes actuales o de otros países puede ser diferente.
- La clase positiva está desbalanceada.
- El umbral 0,18 debe recalibrarse con datos del contexto real de uso.
- Variables como edad, ocupación, educación y estado civil pueden generar segmentaciones injustas.
- Un falso positivo puede producir una llamada innecesaria y un falso negativo puede omitir a un cliente interesado.
- El modelo debe apoyar la decisión humana, no excluir automáticamente a una persona.
- Antes de una implementación real se requiere validación con datos recientes, revisión de sesgos y monitoreo periódico.

## Documentación

- [Notebook del proyecto](./Proyecto_Bank_Marketing_Colab_Grupo1.ipynb)
- [Informe técnico](./Informe_Tecnico_Bank_Marketing_Grupo1_Final.pdf)
- [Dataset en UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [DOI del dataset](https://doi.org/10.24432/C5K306)

## Equipo

- Anthony Polo.
- Ritter Briones.
- Mariana Mora.

**Programa:** Maestría en Gestión y Analítica de Datos  
**Institución:** Universidad San Gregorio de Portoviejo  
**Docente:** Mg. Adriana Collaguazo Jaramillo

## Referencias

- Moro, S., Rita, P. y Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5K306>
- UCI Machine Learning Repository. *Bank Marketing*. <https://archive.ics.uci.edu/dataset/222/bank+marketing>

## Uso académico

Este repositorio fue desarrollado con fines académicos. El dataset conserva su licencia **Creative Commons Attribution 4.0 (CC BY 4.0)**. Cualquier uso del código o de los resultados debe reconocer a sus autores y a la fuente original de los datos.
