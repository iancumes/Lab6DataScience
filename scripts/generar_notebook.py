"""Genera los tres notebooks del laboratorio a partir de scripts/lab6_analisis.py.

El script está escrito en formato *percent* (`# %%`) y es la única fuente de verdad. De él se
derivan tres notebooks **acumulativos**, siguiendo el mismo patrón que los notebooks del avance:
cada hito conserva íntegro el anterior y añade las actividades siguientes, de modo que los tres son
autocontenidos y pueden ejecutarse por separado.

    notebooks/01_carga_calidad_y_preprocesamiento.ipynb   actividades 1 y 2
    notebooks/02_exploratorio_y_red_bipartita.ipynb       + actividades 3 y 4
    notebooks/03_laboratorio6_completo.ipynb              + actividades 5 a 10 (entrega final)

Los notebooks se generan, nunca se editan a mano: así el código del script y el de los notebooks no
pueden desincronizarse. Las cifras de los resúmenes se leen de outputs/resultados.json.

Uso:
    python scripts/generar_notebook.py
    python scripts/generar_notebook.py --ejecutar          # ejecuta los tres
    python scripts/generar_notebook.py --ejecutar --solo 3 # ejecuta sólo el hito 3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
ORIGEN = ROOT / "scripts" / "lab6_analisis.py"
DESTINO = ROOT / "notebooks"

SEPARADOR = re.compile(r"^# %%(?P<markdown> \[markdown\])?\s*$")

# Encabezado de la primera celda de cada hito posterior: marca dónde se corta el script.
CORTES = {
    2: "## 3. Análisis exploratorio",
    3: "## 5. Proyecciones de la red",
}


def dividir_en_celdas(texto: str) -> list[tuple[str, str]]:
    celdas: list[tuple[str, str]] = []
    tipo, buffer = None, []
    for linea in texto.splitlines():
        coincidencia = SEPARADOR.match(linea)
        if coincidencia:
            if tipo is not None:
                celdas.append((tipo, "\n".join(buffer)))
            tipo = "markdown" if coincidencia.group("markdown") else "code"
            buffer = []
        elif tipo is not None:
            buffer.append(linea)
    if tipo is not None:
        celdas.append((tipo, "\n".join(buffer)))
    return celdas


def limpiar_markdown(texto: str) -> str:
    """Quita el '# ' inicial que convierte el texto en comentario de Python."""
    return "\n".join(linea[2:] if linea.startswith("# ") else ("" if linea.strip() == "#" else linea)
                     for linea in texto.splitlines()).strip()


def limpiar_codigo(texto: str) -> str:
    """En el notebook no hacen falta el backend sin ventana ni el rodeo de __file__."""
    return texto.strip("\n").replace(
        'matplotlib.use("Agg") if __name__ == "__main__" else None\n', "")


def celdas_del_script() -> list[nbf.NotebookNode]:
    celdas = []
    for tipo, contenido in dividir_en_celdas(ORIGEN.read_text(encoding="utf-8")):
        if tipo == "markdown":
            texto = limpiar_markdown(contenido)
            if texto:
                celdas.append(nbf.v4.new_markdown_cell(texto))
        else:
            codigo = limpiar_codigo(contenido)
            if codigo.strip():
                celdas.append(nbf.v4.new_code_cell(codigo))
    return celdas


def indice_de_corte(celdas: list[nbf.NotebookNode], encabezado: str) -> int:
    for indice, celda in enumerate(celdas):
        if celda.cell_type == "markdown" and celda.source.startswith(encabezado):
            return indice
    raise ValueError(f"No se encontró el encabezado de corte: {encabezado!r}")


def metricas() -> dict:
    ruta = ROOT / "outputs" / "resultados.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def resumenes(R: dict) -> dict[int, dict]:
    """Título y tl;dr de cada hito, con las cifras leídas del análisis."""
    def n(clave, defecto="—"):
        return R.get(clave, defecto)

    return {
        1: {
            "archivo": "01_carga_calidad_y_preprocesamiento.ipynb",
            "titulo": "Laboratorio 6 — Hito 1: carga, integración, calidad y preprocesamiento",
            "actividades": "Cubre las actividades 1 y 2 del enunciado.",
            "tldr": [
                f"Se cargaron {n('n_videos')} videos y {n('n_comentarios')} comentarios sin modificar "
                "los archivos originales.",
                f"Las llaves primarias son completas y únicas; los {n('comentarios_asociados')} "
                "comentarios se asocian con un video sin pérdida ni expansión de filas.",
                f"El riesgo principal no es la integridad de la unión sino la cobertura: sólo "
                f"{n('videos_con_comentarios')} de {n('n_videos')} videos "
                f"({n('cobertura_videos_pct')} %) tienen comentarios recolectados.",
                f"El diagnóstico detecta {n('likes_en_blanco')} conteos de «me gusta» en blanco, "
                f"{n('handles_codificados')} handles con codificación porcentual de URL y dos "
                "variables inutilizables (viewer_rating, is_pinned).",
                f"La limpieza de texto no elimina ningún registro y reduce el volumen de tokens en "
                f"{n('reduccion_tokens_pct')} %.",
            ],
        },
        2: {
            "archivo": "02_exploratorio_y_red_bipartita.ipynb",
            "titulo": "Laboratorio 6 — Hito 2: análisis exploratorio y red bipartita",
            "actividades": "Conserva íntegro el hito 1 y añade las actividades 3 y 4.",
            "tldr": [
                "Este notebook repite el hito 1 completo y sobre él construye el análisis "
                "exploratorio y la red bipartita autor–video.",
                f"La participación está muy concentrada: un solo video reúne el "
                f"{n('video_top_pct')} % de los comentarios y un solo canal el "
                f"{n('canal_top_pct')} % (Gini de comentarios por video con cobertura: "
                f"{n('gini_videos_cobertura')}).",
                f"Sólo {n('autores_multivideo_n')} de {n('autores_unicos')} autores comentan en más "
                f"de un video y {n('autores_multicanal_n')} cruzan canales distintos.",
                f"La red completa contiene {n('red_nodos_total')} nodos y {n('red_aristas_total')} "
                f"aristas cuyo peso suma {n('peso_total')}, es decir, el total de comentarios.",
                f"Los {n('videos_sin_comentarios')} videos sin comentarios se conservan en la red "
                "como falta de cobertura, no como aislamiento social demostrado.",
            ],
        },
        3: {
            "archivo": "03_laboratorio6_completo.ipynb",
            "titulo": "Laboratorio 6 — Entrega final: actividades 1 a 10",
            "actividades": "Versión canónica: conserva los hitos 1 y 2 y añade las actividades 5 a 10.",
            "tldr": [
                "Notebook canónico de la entrega. Reproduce todo el análisis y exporta las tablas, "
                "figuras, grafos y métricas que alimentan el informe.",
                f"La subred observada se fragmenta en {n('n_componentes_obs')} componentes; la mayor "
                f"reúne el {n('pct_componente_mayor')} % de los nodos.",
                f"El {n('pct_autores_grado1')} % de los autores tiene grado 1: casi nadie comenta en "
                "más de un video, de modo que la red agrupa audiencias, no conversaciones.",
                f"Louvain ponderado detecta {n('n_comunidades')} comunidades "
                f"(modularidad {n('modularidad')}) que coinciden casi exactamente con videos "
                "individuales.",
                f"{n('n_autores_puente')} autores son puntos de articulación: eliminarlos fragmenta "
                "el núcleo de la red pese a que aportan entre 2 y 4 comentarios cada uno.",
                f"El {n('pct_negativo_global')} % de los comentarios es negativo, con diferencias "
                "significativas entre canales.",
            ],
        },
    }


CIERRES = {
    1: """## Cierre del hito 1

La integración es íntegra y las llaves son utilizables: ningún comentario queda huérfano y la unión
no expande filas. El riesgo real está en la cobertura, no en la calidad de las llaves.

La limpieza conserva los dos textos exigidos —`texto_original` para auditoría y sentimiento,
`texto_limpio` para frecuencias y temas— y no elimina ningún registro, porque cada comentario es
una arista de la red que se construye en el hito 2.

El análisis exploratorio, la red bipartita y las actividades 5 a 10 continúan en
`02_exploratorio_y_red_bipartita.ipynb` y `03_laboratorio6_completo.ipynb`.""",
    2: """## Cierre del hito 2

La participación observada está fuertemente concentrada en pocos videos y canales, y la cobertura
de comentarios condiciona todas las comparaciones. Popularidad y participación describen fenómenos
distintos: la correlación entre visualizaciones y comentarios sólo aparece al restringirse a los
videos con cobertura.

La red bipartita queda construida y validada, con la tabla de nodos, la tabla de aristas y el
significado exacto de una arista declarado antes de interpretarla. Las proyecciones, la topología,
las comunidades, la centralidad y el sentimiento se desarrollan en
`03_laboratorio6_completo.ipynb`.""",
}


def construir(hito: int, celdas: list[nbf.NotebookNode], resumen: dict) -> nbf.NotebookNode:
    cabecera = [
        nbf.v4.new_markdown_cell(
            f"# {resumen['titulo']}\n\n"
            "**Universidad del Valle de Guatemala — CC3084 Data Science — Semestre II, 2026**\n\n"
            f"{resumen['actividades']}"
        ),
        nbf.v4.new_markdown_cell(
            "## tl;dr\n\n" + "\n".join(f"- {linea}" for linea in resumen["tldr"])
        ),
    ]
    cuerpo = list(celdas)
    if hito in CIERRES:
        cuerpo.append(nbf.v4.new_markdown_cell(CIERRES[hito]))

    notebook = nbf.v4.new_notebook(cells=cabecera + cuerpo)
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python (Lab 6)", "language": "python", "name": "lab6"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    })
    return notebook


def main() -> None:
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--ejecutar", action="store_true",
                            help="ejecuta los notebooks y guarda las salidas")
    analizador.add_argument("--solo", type=int, choices=(1, 2, 3), default=None,
                            help="genera (y ejecuta) sólo el hito indicado")
    argumentos = analizador.parse_args()

    todas = celdas_del_script()
    # La primera celda es el título global del script; cada notebook pone el suyo.
    cuerpo = todas[1:]
    limites = {
        1: indice_de_corte(cuerpo, CORTES[2]),
        2: indice_de_corte(cuerpo, CORTES[3]),
        3: len(cuerpo),
    }

    DESTINO.mkdir(parents=True, exist_ok=True)
    descripciones = resumenes(metricas())
    hitos = [argumentos.solo] if argumentos.solo else [1, 2, 3]

    for hito in hitos:
        resumen = descripciones[hito]
        notebook = construir(hito, cuerpo[: limites[hito]], resumen)
        if argumentos.ejecutar:
            from nbclient import NotebookClient
            NotebookClient(notebook, timeout=1800, kernel_name="python3",
                           resources={"metadata": {"path": str(ROOT)}}).execute()
        ruta = DESTINO / resumen["archivo"]
        nbf.write(notebook, ruta)
        print(f"Hito {hito}: {ruta.relative_to(ROOT)} ({len(notebook.cells)} celdas)"
              + (" — ejecutado" if argumentos.ejecutar else ""))


if __name__ == "__main__":
    main()
