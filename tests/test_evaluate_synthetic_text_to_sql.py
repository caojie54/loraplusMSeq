import unittest

from evaluate_synthetic_text_to_sql import (
    evaluate_sql,
    evaluate_sql_metrics,
    normalize_sql,
)


MULTI_STATEMENT_CASES = [
    ("INSERT INTO t VALUES (1); UPDATE t SET v = 2;", "INSERT INTO t VALUES (1);"),
    ("SELECT * FROM a; SELECT * FROM b;", "SELECT * FROM a;"),
    ("DELETE FROM a; DELETE FROM b;", "DELETE FROM a;"),
    ("CREATE TABLE a (id INT); CREATE TABLE b (id INT);", "CREATE TABLE a (id INT);"),
    ("ALTER TABLE a ADD v INT; UPDATE a SET v = 1;", "ALTER TABLE a ADD v INT;"),
    ("DROP TABLE a; DROP TABLE b;", "DROP TABLE a;"),
    ("REPLACE INTO a VALUES (1); REPLACE INTO b VALUES (2);", "REPLACE INTO a VALUES (1);"),
    ("TRUNCATE TABLE a; TRUNCATE TABLE b;", "TRUNCATE TABLE a;"),
    ("WITH x AS (SELECT 1) SELECT * FROM x; SELECT 2;", "WITH x AS (SELECT 1) SELECT * FROM x;"),
    ("INSERT INTO a VALUES ('x;y'); UPDATE a SET v = 'z';", "INSERT INTO a VALUES ('x;y');"),
    ('INSERT INTO a VALUES ("x;y"); UPDATE a SET v = "z";', 'INSERT INTO a VALUES ("x;y");'),
    ("SELECT 'a;b' AS value; SELECT 'c;d' AS value;", "SELECT 'a;b' AS value;"),
    ("UPDATE a SET v = 'one;two'; DELETE FROM a WHERE id = 2;", "UPDATE a SET v = 'one;two';"),
    ("INSERT INTO a SELECT * FROM b; INSERT INTO c SELECT * FROM d;", "INSERT INTO a SELECT * FROM b;"),
    ("CREATE VIEW a AS SELECT 1; SELECT * FROM a;", "CREATE VIEW a AS SELECT 1;"),
    ("UPDATE a SET v = 1; SELECT v FROM a;", "UPDATE a SET v = 1;"),
    ("DELETE FROM a WHERE id = 1; INSERT INTO audit VALUES (1);", "DELETE FROM a WHERE id = 1;"),
    ("SELECT COUNT(*) FROM a; SELECT COUNT(*) FROM b; SELECT COUNT(*) FROM c;", "SELECT COUNT(*) FROM a;"),
    ("INSERT INTO a VALUES (1); UPDATE a SET v = 2; SELECT * FROM a;", "INSERT INTO a VALUES (1);"),
    ("CREATE TABLE a (v TEXT); INSERT INTO a VALUES ('semi;colon'); SELECT * FROM a;", "CREATE TABLE a (v TEXT);"),
]


class SqlExactMatchTest(unittest.TestCase):
    def test_full_multi_statement_predictions_match(self):
        for full_sql, _ in MULTI_STATEMENT_CASES:
            with self.subTest(sql=full_sql):
                score, checked = evaluate_sql(
                    [{"response": f"```sql\n{full_sql}\n```", "answer": full_sql}]
                )
                self.assertEqual(score, 1.0)
                self.assertTrue(checked[0]["flag"])

    def test_first_statement_only_never_matches(self):
        self.assertEqual(len(MULTI_STATEMENT_CASES), 20)
        for full_sql, first_statement in MULTI_STATEMENT_CASES:
            with self.subTest(sql=full_sql):
                score, checked = evaluate_sql(
                    [{"response": first_statement, "answer": full_sql}]
                )
                self.assertEqual(score, 0.0)
                self.assertFalse(checked[0]["flag"])

    def test_normalization_is_quote_aware(self):
        sql = "  Answer:  SELECT  'Alpha  Beta', \"Mixed Case\"  FROM T;  "
        self.assertEqual(
            normalize_sql(sql),
            "select 'Alpha  Beta', \"Mixed Case\" from t",
        )
        self.assertNotEqual(normalize_sql("SELECT 'ABC'"), normalize_sql("SELECT 'abc'"))

    def test_semicolon_inside_literal_is_not_a_split_point(self):
        sql = "SELECT 'alpha;beta' AS value;"
        self.assertEqual(normalize_sql(sql), "select 'alpha;beta' as value")

    def test_exact_match_is_primary_across_every_row(self):
        rows = [
            {"response": "select 1", "answer": "SELECT 1", "sql_context": "bad context"},
            {"response": "select missing", "answer": "SELECT missing"},
            {"response": "not sql", "answer": "SELECT 1"},
            {"response": "```sql\nselect 1;\n```", "answer": "SELECT 1"},
        ]
        metrics, checked = evaluate_sql_metrics(rows)
        self.assertEqual(metrics["exact_match"], 0.75)
        self.assertEqual(metrics["exact_match_correct"], 3)
        self.assertEqual(metrics["total"], 4)
        self.assertEqual(metrics["execution_accuracy"], 0.5)
        self.assertEqual(metrics["execution_correct"], 1)
        self.assertEqual(metrics["execution_eligible"], 2)
        self.assertEqual(metrics["execution_coverage"], 0.5)
        self.assertEqual(metrics["context_invalid"], 1)
        self.assertEqual(metrics["gold_invalid"], 1)
        self.assertEqual(metrics["prediction_invalid"], 1)
        self.assertEqual(
            [row["execution_status"] for row in checked],
            ["context_invalid", "gold_invalid", "prediction_invalid", "match"],
        )

    def test_no_eligible_rows_uses_null_execution_accuracy(self):
        metrics, _ = evaluate_sql_metrics(
            [{"response": "SELECT 1", "answer": "SELECT missing"}]
        )
        self.assertIsNone(metrics["execution_accuracy"])
        self.assertEqual(metrics["execution_coverage"], 0.0)


class SqlExecutionTest(unittest.TestCase):
    def assert_execution(self, context, gold, prediction, expected):
        metrics, checked = evaluate_sql_metrics(
            [
                {
                    "sql_context": context,
                    "answer": gold,
                    "response": prediction,
                }
            ]
        )
        self.assertEqual(metrics["execution_eligible"], 1)
        self.assertEqual(metrics["execution_correct"], int(expected))
        self.assertEqual(checked[0]["execution_match"], expected)

    def test_unordered_query_comparison_preserves_duplicates(self):
        context = "CREATE TABLE t(v INT); INSERT INTO t VALUES (1), (1), (2);"
        self.assert_execution(context, "SELECT v FROM t", "SELECT DISTINCT v FROM t", False)
        self.assert_execution(
            context,
            "SELECT v FROM t",
            "SELECT v FROM t ORDER BY v DESC",
            True,
        )

    def test_context_is_recovered_from_compact_dataset_input(self):
        context = "CREATE TABLE t(v INT); INSERT INTO t VALUES (1);"
        metrics, checked = evaluate_sql_metrics(
            [
                {
                    "input": "Return every value.\n\n### Context:\n" + context,
                    "output": "SELECT v FROM t",
                    "response": "SELECT v FROM t",
                }
            ]
        )
        self.assertEqual(metrics["execution_eligible"], 1)
        self.assertEqual(metrics["execution_correct"], 1)
        self.assertTrue(checked[0]["execution_match"])

    def test_gold_top_level_order_by_is_honored(self):
        context = "CREATE TABLE t(v INT); INSERT INTO t VALUES (1), (2);"
        self.assert_execution(
            context,
            "SELECT v FROM t ORDER BY v ASC",
            "SELECT v FROM t ORDER BY v DESC",
            False,
        )

    def test_empty_results_compare_width_but_ignore_aliases(self):
        context = "CREATE TABLE t(v INT);"
        self.assert_execution(
            context,
            "SELECT v AS one FROM t",
            "SELECT v AS one, v AS two FROM t",
            False,
        )
        self.assert_execution(
            context,
            "SELECT v AS one FROM t",
            "SELECT v AS another_name FROM t",
            True,
        )

    def test_order_by_inside_subquery_does_not_order_outer_result(self):
        context = "CREATE TABLE t(v INT); INSERT INTO t VALUES (1), (2);"
        self.assert_execution(
            context,
            "SELECT v FROM (SELECT v FROM t ORDER BY v ASC)",
            "SELECT v FROM t ORDER BY v DESC",
            True,
        )

    def test_dml_compares_every_table_not_only_target_table(self):
        context = (
            "CREATE TABLE a(v INT); CREATE TABLE b(v INT);"
            "INSERT INTO a VALUES (1); INSERT INTO b VALUES (9);"
        )
        self.assert_execution(
            context,
            "UPDATE a SET v = 2;",
            "UPDATE a SET v = 2; DELETE FROM b;",
            False,
        )

    def test_ddl_compares_full_schema(self):
        self.assert_execution(
            "",
            "CREATE TABLE extra(v INT);",
            "CREATE VIEW extra AS SELECT 1 AS v;",
            False,
        )

    def test_all_mutating_statements_are_executed(self):
        context = "CREATE TABLE t(v INT); INSERT INTO t VALUES (1);"
        self.assert_execution(
            context,
            "INSERT INTO t VALUES (2); UPDATE t SET v = 3 WHERE v = 1;",
            "INSERT INTO t VALUES (2);",
            False,
        )

    def test_trigger_body_semicolons_are_not_split(self):
        context = (
            "CREATE TABLE t(v INT); CREATE TABLE audit(v INT);"
        )
        script = (
            "CREATE TRIGGER log_insert AFTER INSERT ON t BEGIN "
            "INSERT INTO audit VALUES (NEW.v); "
            "INSERT INTO audit VALUES (NEW.v + 1); END; "
            "INSERT INTO t VALUES (4);"
        )
        self.assert_execution(context, script, script, True)

    def test_prediction_cannot_attach_a_filesystem_database(self):
        metrics, checked = evaluate_sql_metrics(
            [
                {
                    "answer": "SELECT 1",
                    "response": "ATTACH DATABASE '/tmp/not-created.db' AS external",
                }
            ]
        )
        self.assertEqual(metrics["prediction_invalid"], 1)
        self.assertEqual(checked[0]["execution_status"], "prediction_invalid")


if __name__ == "__main__":
    unittest.main()
