# Laboratorio 6 — Análisis de redes sociales en YouTube

**Universidad del Valle de Guatemala · CC3084 Data Science · Semestre II, 2026**

Análisis sobre `youtube_videos.csv` (293 videos × 20 variables) y `youtube_comments.csv`
(406 comentarios × 17 variables).

## Estado

El avance (actividades 1 a 4) se reconstruyó como un **pipeline reproducible único**,
`scripts/lab6_analisis.py`, sobre el que se desarrollan las actividades 5 a 10. Los notebooks
originales del avance se conservan en `notebooks/avance/`.

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

## Ejecución

```bash
python scripts/lab6_analisis.py
```

Ese único comando regenera todo el material derivado: `outputs/tables/`, `outputs/figures/`,
`outputs/graphs/`, `data/processed/` y `outputs/resultados.json` con las métricas del análisis.

El script está escrito en formato *percent* (`# %%`), de modo que el mismo archivo se ejecuta como
script y se puede convertir a notebook. El análisis es determinista: todas las semillas están
fijadas en 42 y las estructuras derivadas de conjuntos se recorren en orden canónico.

## Decisiones metodológicas principales

- **Los identificadores son la llave; los nombres son sólo etiquetas.** `video_id`, `channel_id`,
  `comment_id` y `author_channel_id` construyen la red; `channel_name`, `author_name` y los
  *handles* nunca sustituyen a un ID.
- **Una arista significa una sola cosa:** ese autor publicó al menos un comentario principal en ese
  video. Su peso es el número de comentarios. No implica amistad, conversación ni aprobación.
- **`reply_count` nunca genera aristas entre usuarios.** Los datos indican cuántas respuestas
  recibió un comentario, pero no quién las escribió.
- **Un video sin comentarios no está aislado: no tiene datos.** 274 de los 293 videos carecen de
  comentarios recolectados; se conservan en la red como falta de cobertura.
- **Ningún comentario se elimina en la limpieza.** Cada uno es una arista; los que quedan sin
  contenido léxico se marcan con `apto_para_texto`.
- **Se conservan dos versiones del texto.** `texto_original` alimenta el sentimiento (necesita
  negación, puntuación y emojis); `texto_limpio`, lematizado, alimenta frecuencias y temas.
- **Un `like_count_text` en blanco significa cero.** YouTube oculta el contador cuando vale 0; la
  imputación queda marcada en `like_count_imputado`.
