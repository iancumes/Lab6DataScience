# Laboratorio 6 — Análisis de redes sociales en YouTube

Avance correspondiente a las actividades 1–4 del laboratorio de CC3084 Data Science.

## Contenido

- `notebooks/01_carga_integracion_y_calidad.ipynb`: carga, comprensión, integración y primera parte del diagnóstico de calidad.
- `notebooks/02_preprocesamiento_y_eda.ipynb`: conserva el hito anterior y añade limpieza de texto y análisis exploratorio.
- `notebooks/03_avance_actividades_1_a_4.ipynb`: versión canónica del avance; conserva lo anterior y añade la red bipartita autor–video.
- `youtube_videos.csv` y `youtube_comments.csv`: datos originales. No son modificados por el análisis.
- `data/processed`: copias procesadas generadas por los notebooks 02 y 03.
- `outputs/tables`: diagnósticos y tablas reproducibles.
- `outputs/figures`: figuras exportadas por los notebooks.
- `outputs/graphs`: red exportada en GraphML para inspección opcional en Gephi.

Los notebooks son acumulativos y autocontenidos: cada uno vuelve a leer los CSV originales y puede ejecutarse sin haber corrido los anteriores. El notebook 03 es la entrega completa del avance.

## Instalación

Se recomienda Python 3.11 o superior y un entorno virtual.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download es_core_news_sm
python -m ipykernel install --user --name lab6 --display-name "Python (Lab 6)"
```

El modelo `es_core_news_sm` se usa para tokenización, stopwords y lematización en español. Los notebooks apuntan al kernel `Python (Lab 6)` registrado en el último comando. Después de instalarlo, reinicie Jupyter si ya estaba abierto.

## Ejecución

Desde la raíz del repositorio:

```powershell
python -m jupyter lab
```

Para validar un notebook de principio a fin:

```powershell
python -m nbconvert --execute --to notebook --inplace notebooks/03_avance_actividades_1_a_4.ipynb --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=lab6
```

También puede ejecutar los tres hitos en orden:

```powershell
python -m nbconvert --execute --to notebook --inplace notebooks/01_carga_integracion_y_calidad.ipynb --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=lab6
python -m nbconvert --execute --to notebook --inplace notebooks/02_preprocesamiento_y_eda.ipynb --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=lab6
python -m nbconvert --execute --to notebook --inplace notebooks/03_avance_actividades_1_a_4.ipynb --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=lab6
```

## Decisiones metodológicas principales

- `video_id`, `channel_id`, `comment_id` y `author_channel_id` son los identificadores; los nombres y handles se conservan solo como etiquetas.
- Un `like_count_text` vacío se interpreta como cero likes mostrados por YouTube. El texto crudo se conserva para auditoría.
- Los comentarios duplicados no se eliminan si tienen IDs distintos; se marcan para revisión.
- Una arista indica que un autor publicó al menos un comentario principal en un video. Su peso es el número de comentarios de ese autor en ese video.
- `reply_count` no crea aristas entre autores porque los datos no identifican quién respondió.
- Los videos sin comentarios recolectados se etiquetan como falta de cobertura y no como aislamiento social demostrado.
- Comunidades y sentimiento se tratan solo de forma preliminar; los análisis formales corresponden a actividades posteriores.

## Hitos sugeridos para commits

1. `feat: cargar, integrar y diagnosticar datos de YouTube`
2. `feat: completar limpieza textual y análisis exploratorio`
3. `feat: construir red bipartita y cerrar avance 1-4`
