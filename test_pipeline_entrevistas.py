import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pipeline_entrevistas as p


class PipelineTests(unittest.TestCase):
    def test_metricas_con_acentos_signos_y_palabras_repetidas(self):
        text = "¡Sí, sí!  2026\n"
        self.assertEqual(p.text_metrics(text), {
            "num_caracteres": 15,
            "num_caracteres_sin_espacios": 11,
            "num_palabras": 3,
            "num_palabras_unicas": 2,
        })
        self.assertEqual(p.text_metrics("... —"), {
            "num_caracteres": 5, "num_caracteres_sin_espacios": 4,
            "num_palabras": 0, "num_palabras_unicas": 0,
        })

    def test_checkpoint_enriquecido_actualiza_metricas_sin_api(self):
        rows = [{"texto": "Fue útil.", "tono_emocional": "Positivo", "resumen_descriptivo": "Resumen previo"}]
        with patch.object(p, "call_model") as model:
            p.enrich(rows, "x", 12, lambda: None)
            model.assert_not_called()
        self.assertNotIn("resumen_descriptivo", rows[0])
        self.assertEqual(rows[0]["num_palabras"], 2)
        self.assertNotIn("resumen_descriptivo", p.ANNOTATION["properties"])

    def test_migracion_sqlite_conserva_otras_entrevistas(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.sqlite"
            with sqlite3.connect(str(path)) as connection:
                connection.execute('CREATE TABLE unidades (entrevista_id TEXT, unidad_id INTEGER, texto TEXT, resumen_descriptivo TEXT, PRIMARY KEY (entrevista_id, unidad_id))')
                connection.execute('INSERT INTO unidades VALUES (?, ?, ?, ?)', ("E1", 1, "Fue útil.", "Resumen"))
            p.write_sqlite(path, [{"entrevista_id": "E2", "unidad_id": 1, "texto": "Falta personal."}])
            with sqlite3.connect(str(path)) as connection:
                names = {column[1] for column in connection.execute('PRAGMA table_info(unidades)')}
                values = connection.execute('SELECT entrevista_id, num_palabras, typeof(num_palabras) FROM unidades ORDER BY entrevista_id').fetchall()
            self.assertNotIn("resumen_descriptivo", names)
            self.assertEqual(values, [("E1", 2, "integer"), ("E2", 2, "integer")])
            self.assertEqual(len(list(Path(directory).glob('*.antes-metricas-*.sqlite'))), 1)

    def test_transcripcion_macroman_y_turnos(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entrevista.txt"
            path.write_bytes("Título\rAltavoz 1  00:00\r¿Qué pasó?\rAltavoz 2  00:02\rFue útil.\r".encode("mac_roman"))
            content, encoding = p.read_transcript(path)
            turns = p.parse_turns(content)
        self.assertEqual(encoding, "mac_roman")
        self.assertEqual([(t.speaker, t.time, t.text) for t in turns], [("1", "00:00", "¿Qué pasó?"), ("2", "00:02", "Fue útil.")])

    def test_segmentacion_literal_y_cobertura(self):
        source = "La iniciativa creció. Necesitamos más personal."
        units = p.locate_units(source, ["Necesitamos más personal."])
        self.assertEqual([u[0] for u in units], ["La iniciativa creció.", "Necesitamos más personal."])
        with self.assertRaises(ValueError):
            p.locate_units(source, ["inventado"])

    def test_flujo_simulado_sin_api(self):
        turns = p.parse_turns("Altavoz 1  00:00\n¿Qué pasó?\nAltavoz 2  00:03\nCrecimos. Falta personal.\nAltavoz 1  00:10\n¿Por qué?\nAltavoz 2  00:12\nLa demanda aumentó.")
        roles = p.classify_roles(turns, "x", {"1": "Entrevistador", "2": "Entrevistado"})
        blocks = p.blocks_for_speaker(turns, roles, 500)
        self.assertEqual(len(blocks), 2)
        with patch.object(p, "call_model", side_effect=[{"inicios": ["Falta personal."]}, {"inicios": []}]):
            units = [u for block in blocks for u in p.segment_block(block, "x")]
        self.assertEqual([u["texto"] for u in units], ["Crecimos.", "Falta personal.", "La demanda aumentó."])
        self.assertEqual([u["tiempo_inicio"] for u in units], ["00:03", "00:03", "00:12"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "datos.csv"
            rows = [{"unidad_id": index, **unit} for index, unit in enumerate(units, 1)]
            p.write_csv(output, rows)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(len(saved), 3)
            self.assertEqual(saved[-1]["texto"], "La demanda aumentó.")
            database = Path(directory) / "datos.sqlite"
            for row in rows:
                row["entrevista_id"] = "E1"
            p.write_sqlite(database, rows)
            p.write_sqlite(database, rows)
            with sqlite3.connect(str(database)) as connection:
                count = connection.execute("SELECT COUNT(*) FROM unidades WHERE entrevista_id = 'E1'").fetchone()[0]
            self.assertEqual(count, 3)


if __name__ == "__main__":
    unittest.main()
