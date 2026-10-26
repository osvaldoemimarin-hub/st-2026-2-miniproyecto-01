# datos

| Campo | Valor |
|---|---|
| Archivo | `sacrificio_porcino.csv` |
| Qué mide | Cabezas de ganado porcino sacrificadas por mes en Colombia, total nacional. El archivo trae también los machos, las hembras y los vacunos sacrificados, que el guion no usa. |
| Fuente | DANE, Encuesta de Sacrificio de Ganado (ESAG), series históricas, cuadro 3 (`series-hist-ESAG-IItrim2026.xls`, actualizado el 10 de agosto de 2026): <https://www.dane.gov.co/index.php/estadisticas-por-tema/agropecuario/encuesta-de-sacrificio-de-ganado> |
| Licencia | Información pública del DANE. Los términos de uso del DANE permiten usarla y reproducirla citando la fuente (Ley 1712 de 2014, de transparencia y acceso a la información pública). |
| Frecuencia | mensual |
| Periodo | 2008-10 a 2026-06 (213 observaciones; los meses de 2026 son preliminares) |
| Columna de fecha | `mes` (formato `AAAA-MM`) |
| Columna que se pronostica | `porcinos` |

La serie no tiene meses vacíos. Lo que se decidió con sus datos atípicos
está en `docs/informe.md`, sección 1.
