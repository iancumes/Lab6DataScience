# Laboratorio 6 — Análisis de redes sociales en YouTube

**Universidad del Valle de Guatemala · CC3084 Data Science · Semestre II, 2026**

Análisis completo de las **actividades 1 a 10** del laboratorio sobre dos conjuntos de datos de
YouTube: `youtube_videos.csv` (293 videos × 20 variables) y `youtube_comments.csv`
(406 comentarios × 17 variables).

## Entregables

| Entregable | Ubicación |
|---|---|
| Informe con resultados, visualizaciones e interpretación | [`informe/Laboratorio6_Informe.docx`](informe/Laboratorio6_Informe.docx) |
| Script reproducible (actividades 1 a 10) | [`scripts/lab6_analisis.py`](scripts/lab6_analisis.py) |
| Notebooks acumulativos, ya ejecutados | [`notebooks/01…`](notebooks/01_carga_calidad_y_preprocesamiento.ipynb) · [`02…`](notebooks/02_exploratorio_y_red_bipartita.ipynb) · [`03…`](notebooks/03_laboratorio6_completo.ipynb) |
| Tablas, figuras y grafos generados | [`outputs/`](outputs/) |
| Datos procesados | [`data/processed/`](data/processed/) |
| Avance entregado el 3 de septiembre (actividades 1 a 4) | [`notebooks/avance/`](notebooks/avance/) |

## Instalación

Requiere **Python 3.11 o superior**.

```bash
python -m venv .venv
source .venv/bin/activate          # en Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

`es_core_news_sm` se usa para tokenización, stopwords y lematización en español. La primera
ejecución descarga además el modelo de sentimiento `robertuito-sentiment-analysis` (~450 MB) desde
Hugging Face; si no hay conexión, el análisis continúa con un respaldo léxico en español y lo
documenta en la variable `modelo_sentimiento`.

Para trabajar con el notebook conviene registrar un kernel propio:

```bash
python -m ipykernel install --user --name lab6 --display-name "Python (Lab 6)"
```

## Ejecución

```bash
python scripts/lab6_analisis.py
```

Ese único comando ejecuta las actividades 1 a 10 y regenera **todo** el material derivado:
`outputs/tables/` (71 CSV), `outputs/figures/` (13 PNG), `outputs/graphs/` (4 GraphML),
`data/processed/` y `outputs/resultados.json` con las 122 métricas que alimentan el informe.
Tarda unos dos minutos, la mayor parte en el modelo de sentimiento.

```bash
python scripts/generar_notebook.py --ejecutar   # los tres notebooks, a partir del mismo código
python scripts/generar_notebook.py --ejecutar --solo 2   # sólo un hito
python scripts/generar_informe.py               # informe .docx a partir de resultados.json
```

### Los tres notebooks

Son **acumulativos y autocontenidos**, igual que los del avance: cada hito conserva íntegro el
anterior, vuelve a leer los CSV originales y puede ejecutarse sin haber corrido los previos.

| Notebook | Actividades | Contenido añadido |
|---|---|---|
| `01_carga_calidad_y_preprocesamiento.ipynb` | 1 y 2 | Carga, unidad de observación, integración, diagnóstico de calidad, normalización, conversión de conteos y limpieza de texto. |
| `02_exploratorio_y_red_bipartita.ipynb` | + 3 y 4 | Descriptivos, concentración, popularidad vs participación, preguntas 3.5 y 3.6, y construcción de la red bipartita. |
| `03_laboratorio6_completo.ipynb` | + 5 a 10 | Proyecciones, topología, comunidades, centralidad, sentimiento, conclusiones y exportación de `resultados.json`. **Es la entrega final.** |

Sólo el hito 3 escribe `outputs/resultados.json`, que es lo que consume el informe.

`scripts/lab6_analisis.py` está escrito en formato *percent* (`# %%`), de modo que el mismo archivo
se ejecuta como script y se convierte en notebooks. **Los notebooks se generan, nunca se editan a
mano**: así las versiones del análisis no pueden desincronizarse. Del mismo modo, ninguna cifra del
informe está escrita a mano; todas se leen de `outputs/resultados.json`.

El análisis es determinista: todas las semillas están fijadas en 42 (numpy, Louvain, diseños de
fuerzas e intermediación).

## Estructura del repositorio

```
├── youtube_videos.csv, youtube_comments.csv   Datos originales, nunca modificados
├── scripts/
│   ├── lab6_analisis.py                       Análisis completo, actividades 1 a 10
│   ├── generar_notebook.py                    Deriva los tres notebooks del script
│   ├── generar_informe.py                     Construye el informe .docx
│   └── avance/generar_notebooks_avance.py     Generador de la entrega parcial
├── notebooks/
│   ├── 01_carga_calidad_y_preprocesamiento.ipynb   Hito 1: actividades 1 y 2
│   ├── 02_exploratorio_y_red_bipartita.ipynb       Hito 2: + actividades 3 y 4
│   ├── 03_laboratorio6_completo.ipynb              Entrega final: + actividades 5 a 10
│   └── avance/                                Actividades 1 a 4 (entrega del 3 de septiembre)
├── informe/Laboratorio6_Informe.docx          Informe final
├── outputs/{tables,figures,graphs}            Material reproducible
└── data/processed/                            Comentarios y videos limpios
```

## Decisiones metodológicas principales

- **Los identificadores son la llave; los nombres son sólo etiquetas.** `video_id`, `channel_id`,
  `comment_id` y `author_channel_id` construyen la red. `channel_name`, `author_name` y los
  *handles* nunca sustituyen a un ID: pueden repetirse o cambiar en el tiempo.
- **Una arista significa una sola cosa:** ese autor publicó al menos un comentario principal en ese
  video. Su peso es el número de comentarios. No implica amistad, conversación ni aprobación.
- **`reply_count` nunca genera aristas entre usuarios.** Los datos indican cuántas respuestas
  recibió un comentario, pero no quién las escribió.
- **Un video sin comentarios no está aislado: no tiene datos.** 274 de los 293 videos carecen de
  comentarios recolectados; se conservan en la red como falta de cobertura, no como silencio social.
- **Ningún comentario se elimina en la limpieza.** Cada uno es una arista; descartar los que quedan
  sin contenido léxico distorsionaría la estructura. Se marcan con `apto_para_texto`.
- **Se conservan dos versiones del texto.** `texto_original` alimenta el sentimiento (necesita
  negación, puntuación y emojis); `texto_limpio`, lematizado, alimenta frecuencias y temas.
- **Un `like_count_text` en blanco significa cero.** YouTube oculta el contador cuando vale 0. La
  imputación de esos 189 registros queda marcada en `like_count_imputado`.
- **Las comunidades se detectan sobre la red bipartita, no sobre la proyección autor–autor**, cuyas
  camarillas convierten 343 observaciones en 10 732 aristas artificiales.
- **La transitividad clásica vale 0 por construcción** en una red bipartita; se reporta el
  *clustering* bipartito de Latapy y el coeficiente de redundancia.
- **Sentimiento con `pysentimiento` (RoBERTuito)**, entrenado en español y con texto de redes
  sociales, aplicado sobre el texto original y con umbral de 10 comentarios para comparar grupos.

## Resultados principales

- La red bipartita completa tiene **625 nodos y 343 aristas** cuyo peso suma los 406 comentarios.
- El **97.3 %** de los autores comentó en un solo video; sólo **9 de 332** aparecen en más de uno.
- La subred observada se fragmenta en **10 componentes**; la mayor reúne el 81.5 % de los nodos.
- **7 autores** son puntos de articulación: eliminarlos fragmenta el núcleo de la red.
- Louvain ponderado detecta **17 comunidades** (modularidad 0.777) que coinciden casi exactamente
  con videos individuales, no con grupos de personas.
- El **61.3 %** de los comentarios es negativo, con diferencias significativas entre canales
  (Kruskal–Wallis, p < 0.001).
