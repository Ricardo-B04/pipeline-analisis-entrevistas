# Pipeline de análisis de entrevistas

Este proyecto hace **análisis cualitativo asistido por IA** de entrevistas. Lee una transcripción `.txt`, identifica turnos por encabezados como `Altavoz 2  00:34`, asigna roles, divide las respuestas en unidades semánticas y produce CSV y SQLite con una fila por unidad.

El repositorio contiene el código, las pruebas y esta guía. Las transcripciones, los archivos de audio, las salidas CSV/SQLite, los respaldos, los checkpoints y las credenciales se excluyen con `.gitignore`.

## Ejecutar

Requiere Python 3.9 o posterior y una clave de la API de OpenAI para las etapas de IA. No hay paquetes adicionales.

```bash
python3 -B pipeline_entrevistas.py "/ruta/entrevista.txt" --preview
export OPENAI_API_KEY="..."
python3 -B pipeline_entrevistas.py "/ruta/entrevista.txt" --output "/ruta/unidades.csv"
```

Si los roles ya se conocen, se pueden fijar y ahorrar esa llamada:

```bash
python3 -B pipeline_entrevistas.py "/ruta/entrevista.txt" \
  --roles '{"1":"Entrevistador","2":"Entrevistado"}' \
  --output "/ruta/unidades.csv"
```

Para cambiar modelos use `--role-model`, `--segment-model` y `--enrich-model`. Los valores iniciales son `gpt-6-luna`, `gpt-6.1-sol` y `gpt-6-luna`. Use `--encoding` si la detección automática no reproduce bien los acentos. El archivo adjunto se lee correctamente como `mac_roman`.

## Salida y control de calidad

El CSV y la tabla `unidades` de SQLite contienen `entrevista_id`, `unidad_id`, interlocutor, rol, turnos y tiempos de origen, texto literal, métricas cuantitativas, tono, emoción, tema, tres ideas, palabras clave, revisión manual y modelos utilizados. Tono tiene tres opciones; emoción usa un catálogo de once emociones y `Ninguna clara`. Las tres columnas de ideas pueden quedar vacías cuando el fragmento no contiene tres ideas reales. La opción `--sqlite` permite consolidar varias entrevistas en una sola base.

Las métricas se calculan con Python sobre el contenido de `texto`, sin llamadas a IA:

- `num_caracteres`: caracteres Unicode, incluyendo espacios y puntuación.
- `num_caracteres_sin_espacios`: caracteres excluyendo espacios, tabuladores y saltos de línea.
- `num_palabras`: secuencias de letras o números; excluye signos aislados. Los términos unidos por guion o apóstrofo cuentan como una palabra.
- `num_palabras_unicas`: palabras distintas sin diferenciar mayúsculas y minúsculas; conserva los acentos.

La columna anterior `resumen_descriptivo` se elimina del esquema y de la petición al modelo. Esto reduce los tokens generados; las llamadas de enriquecimiento siguen obteniendo las categorías, el tema y las ideas. Al escribir sobre una base SQLite del esquema anterior, el script crea un respaldo `*.antes-metricas-*.sqlite` y migra todas sus entrevistas, calculando sus métricas localmente.

Python recorta las unidades del texto original con anclas que propone el modelo. Si un ancla no es literal o es ambigua, se reintenta una vez y se detiene la corrida; no se exporta un CSV incompleto. Un archivo `.checkpoint.json` junto al CSV permite reanudar sin repetir los bloques terminados. Se elimina al completar la exportación. La columna `revision_manual` marca preguntas en voz del entrevistado, posibles errores de atribución o preguntas citadas que conviene revisar.

El texto se envía a la API durante el análisis. `store: false` evita guardar las respuestas como estado reutilizable de la API; revise la política de datos de su organización antes de procesar entrevistas sensibles. Conserve el `.txt` original para auditar las filas. Un CSV no reemplaza una revisión cualitativa humana.

## Pruebas

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python3 -m unittest -v test_pipeline_entrevistas.py
```
