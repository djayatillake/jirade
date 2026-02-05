"""Databricks SQL client for metadata-only queries.

This client provides secure access to Databricks for CI operations,
ensuring that only metadata queries are allowed (no raw data exposure).

Supports both OAuth (recommended) and token-based authentication.
"""

import logging
import re
from typing import Any, Literal

from databricks import sql as databricks_sql

logger = logging.getLogger(__name__)

AuthType = Literal["oauth", "token"]


class DatabricksMetadataClient:
    """Client for Databricks SQL with metadata-only query restrictions.

    Security constraints:
    - Only allows metadata queries (DESCRIBE, SHOW, COUNT, aggregations)
    - No SELECT * or raw data queries
    - Results are aggregated, never individual records

    Authentication:
    - OAuth (default): Uses browser-based OAuth or existing Databricks CLI credentials
    - Token: Uses a personal access token (PAT)
    """

    # Whitelist of allowed query patterns (compiled regex)
    ALLOWED_PATTERNS = [
        # Schema/table information
        re.compile(r"^\s*DESCRIBE\s+(TABLE\s+)?(EXTENDED\s+)?[\w.`\"]+\s*$", re.IGNORECASE),
        re.compile(r"^\s*SHOW\s+COLUMNS\s+(IN|FROM)\s+[\w.`\"]+\s*$", re.IGNORECASE),
        re.compile(r"^\s*SHOW\s+TABLES\s*(IN|FROM)?\s*[\w.`\"]*\s*(LIKE\s+['\"].*['\"])?\s*$", re.IGNORECASE),
        re.compile(r"^\s*SHOW\s+SCHEMAS\s*(LIKE\s+['\"].*['\"])?\s*$", re.IGNORECASE),
        # Count queries (row counts)
        re.compile(r"^\s*SELECT\s+COUNT\s*\(\s*\*\s*\)\s+FROM\s+[\w.`\"]+\s*(WHERE\s+.*)?\s*$", re.IGNORECASE),
        re.compile(r"^\s*SELECT\s+COUNT\s*\(\s*\*\s*\)\s+AS\s+\w+\s+FROM\s+[\w.`\"]+\s*(WHERE\s+.*)?\s*$", re.IGNORECASE),
        # Distinct count queries (cardinality)
        re.compile(r"^\s*SELECT\s+COUNT\s*\(\s*DISTINCT\s+[\w.`\"]+\s*\)\s+(AS\s+\w+\s+)?FROM\s+[\w.`\"]+\s*(WHERE\s+.*)?\s*$", re.IGNORECASE),
        # NULL count queries
        re.compile(r"^\s*SELECT\s+COUNT\s*\(\s*\*\s*\)\s+(AS\s+\w+\s+)?FROM\s+[\w.`\"]+\s+WHERE\s+[\w.`\"]+\s+IS\s+(NOT\s+)?NULL\s*$", re.IGNORECASE),
        # Value distribution (GROUP BY with COUNT)
        re.compile(r"^\s*SELECT\s+[\w.`\"]+\s*,\s*COUNT\s*\(\s*\*\s*\)\s+(AS\s+\w+\s+)?FROM\s+[\w.`\"]+\s+(WHERE\s+.*)?\s*GROUP\s+BY\s+[\w.`\"]+\s*(ORDER\s+BY\s+.*)?(LIMIT\s+\d+)?\s*$", re.IGNORECASE),
        # Min/Max for numeric ranges
        re.compile(r"^\s*SELECT\s+(MIN|MAX)\s*\(\s*[\w.`\"]+\s*\)\s+(AS\s+\w+\s+)?(,\s*(MIN|MAX)\s*\(\s*[\w.`\"]+\s*\)\s+(AS\s+\w+\s+)?)*FROM\s+[\w.`\"]+\s*(WHERE\s+.*)?\s*$", re.IGNORECASE),
        # Drop table/schema (for cleanup)
        re.compile(r"^\s*DROP\s+TABLE\s+(IF\s+EXISTS\s+)?[\w.`\"]+\s*$", re.IGNORECASE),
        re.compile(r"^\s*DROP\s+SCHEMA\s+(IF\s+EXISTS\s+)?[\w.`\"]+\s*(CASCADE|RESTRICT)?\s*$", re.IGNORECASE),
        # Create schema (for CI)
        re.compile(r"^\s*CREATE\s+SCHEMA\s+(IF\s+NOT\s+EXISTS\s+)?[\w.`\"]+\s*$", re.IGNORECASE),
    ]

    def __init__(
        self,
        host: str,
        http_path: str,
        auth_type: AuthType = "oauth",
        token: str | None = None,
        catalog: str | None = None,
    ):
        """Initialize Databricks client.

        Args:
            host: Databricks workspace host URL.
            http_path: SQL warehouse HTTP path.
            auth_type: Authentication type - "oauth" (default) or "token".
            token: Databricks personal access token (required if auth_type="token").
            catalog: Default Unity Catalog to use (optional).
        """
        self.host = host.rstrip("/").replace("https://", "")
        self.http_path = http_path
        self.auth_type = auth_type
        self.token = token
        self.catalog = catalog
        self._connection = None

        if auth_type == "token" and not token:
            raise ValueError("Token is required when auth_type='token'")

    def _get_connection(self):
        """Get or create a Databricks SQL connection."""
        if self._connection is None:
            if self.auth_type == "oauth":
                # Use OAuth - will use browser auth or existing credentials
                logger.info(f"Connecting to Databricks via OAuth: {self.host}")
                self._connection = databricks_sql.connect(
                    server_hostname=self.host,
                    http_path=self.http_path,
                    auth_type="databricks-oauth",
                )
            else:
                # Use token authentication
                logger.info(f"Connecting to Databricks via token: {self.host}")
                self._connection = databricks_sql.connect(
                    server_hostname=self.host,
                    http_path=self.http_path,
                    access_token=self.token,
                )

            # Set default catalog if specified
            if self.catalog:
                cursor = self._connection.cursor()
                cursor.execute(f"USE CATALOG {self.catalog}")
                cursor.close()
                logger.info(f"Set default catalog: {self.catalog}")

        return self._connection

    def close(self) -> None:
        """Close the connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "DatabricksMetadataClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def is_query_allowed(self, query: str) -> bool:
        """Check if a query is allowed based on the whitelist.

        Args:
            query: SQL query to check.

        Returns:
            True if query matches an allowed pattern.
        """
        # Normalize query
        normalized = " ".join(query.split())

        for pattern in self.ALLOWED_PATTERNS:
            if pattern.match(normalized):
                return True

        return False

    def execute_metadata_query(self, query: str) -> list[dict[str, Any]]:
        """Execute a metadata query with security validation.

        Args:
            query: SQL query to execute.

        Returns:
            List of result rows as dicts.

        Raises:
            ValueError: If query is not allowed.
        """
        if not self.is_query_allowed(query):
            raise ValueError(
                f"Query not allowed. Only metadata queries (DESCRIBE, SHOW, COUNT, GROUP BY aggregations) are permitted. "
                f"Query: {query[:100]}..."
            )

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()

            return [dict(zip(columns, row)) for row in rows]
        finally:
            cursor.close()

    def execute_unsafe_query(self, query: str) -> list[dict[str, Any]]:
        """Execute a query without security validation.

        WARNING: This should only be used for dbt build commands
        that run in isolated CI schemas. Never use for user-facing queries.

        Args:
            query: SQL query to execute.

        Returns:
            List of result rows as dicts.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(query)
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
            return []
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    # High-level metadata methods
    # -------------------------------------------------------------------------

    def get_table_schema(self, table_name: str) -> list[dict[str, Any]]:
        """Get schema information for a table.

        Args:
            table_name: Fully qualified table name (catalog.schema.table).

        Returns:
            List of column info dicts with name, type, nullable, etc.
        """
        return self.execute_metadata_query(f"DESCRIBE TABLE {table_name}")

    def get_row_count(self, table_name: str, where_clause: str | None = None) -> int:
        """Get row count for a table.

        Args:
            table_name: Fully qualified table name.
            where_clause: Optional WHERE clause (without WHERE keyword).

        Returns:
            Row count.
        """
        query = f"SELECT COUNT(*) FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"

        result = self.execute_metadata_query(query)
        return result[0][list(result[0].keys())[0]] if result else 0

    def get_null_count(self, table_name: str, column_name: str) -> int:
        """Get count of NULL values in a column.

        Args:
            table_name: Fully qualified table name.
            column_name: Column name.

        Returns:
            Count of NULL values.
        """
        result = self.execute_metadata_query(
            f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} IS NULL"
        )
        return result[0][list(result[0].keys())[0]] if result else 0

    def get_distinct_count(self, table_name: str, column_name: str) -> int:
        """Get count of distinct values in a column.

        Args:
            table_name: Fully qualified table name.
            column_name: Column name.

        Returns:
            Count of distinct values.
        """
        result = self.execute_metadata_query(
            f"SELECT COUNT(DISTINCT {column_name}) FROM {table_name}"
        )
        return result[0][list(result[0].keys())[0]] if result else 0

    def get_value_distribution(
        self,
        table_name: str,
        column_name: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get value distribution for a column (for low-cardinality columns).

        Args:
            table_name: Fully qualified table name.
            column_name: Column name.
            limit: Maximum number of distinct values to return.

        Returns:
            List of {value, count} dicts.
        """
        result = self.execute_metadata_query(
            f"SELECT {column_name}, COUNT(*) AS cnt FROM {table_name} "
            f"GROUP BY {column_name} ORDER BY cnt DESC LIMIT {limit}"
        )
        return result

    def get_table_metadata(self, table_name: str) -> dict[str, Any]:
        """Get comprehensive metadata for a table.

        Returns schema, row count, and NULL counts for each column.

        Args:
            table_name: Fully qualified table name.

        Returns:
            Dict with schema, row_count, and column_stats.
        """
        schema = self.get_table_schema(table_name)
        row_count = self.get_row_count(table_name)

        column_stats = []
        for col in schema:
            col_name = col.get("col_name", col.get("column_name", ""))
            if not col_name or col_name.startswith("#"):
                continue

            try:
                null_count = self.get_null_count(table_name, col_name)
                distinct_count = self.get_distinct_count(table_name, col_name)
                column_stats.append({
                    "column": col_name,
                    "type": col.get("data_type", col.get("type", "")),
                    "null_count": null_count,
                    "null_pct": round(null_count * 100.0 / row_count, 2) if row_count > 0 else 0,
                    "distinct_count": distinct_count,
                    "uniqueness": round(distinct_count * 100.0 / row_count, 2) if row_count > 0 else 0,
                })
            except Exception as e:
                logger.warning(f"Failed to get stats for column {col_name}: {e}")
                column_stats.append({
                    "column": col_name,
                    "type": col.get("data_type", col.get("type", "")),
                    "error": str(e),
                })

        return {
            "table": table_name,
            "row_count": row_count,
            "schema": schema,
            "column_stats": column_stats,
        }

    # -------------------------------------------------------------------------
    # CI schema management
    # -------------------------------------------------------------------------

    def create_ci_schema(self, schema_name: str) -> None:
        """Create a CI schema.

        Args:
            schema_name: Schema name to create.
        """
        self.execute_metadata_query(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
        logger.info(f"Created CI schema: {schema_name}")

    def drop_ci_schema(self, schema_name: str) -> None:
        """Drop a CI schema and all its tables.

        Args:
            schema_name: Schema name to drop.
        """
        self.execute_metadata_query(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
        logger.info(f"Dropped CI schema: {schema_name}")

    def list_tables_in_schema(self, schema_name: str) -> list[str]:
        """List all tables in a schema.

        Args:
            schema_name: Schema name.

        Returns:
            List of table names.
        """
        result = self.execute_metadata_query(f"SHOW TABLES IN {schema_name}")
        return [row.get("tableName", row.get("table_name", "")) for row in result]

    def drop_table(self, table_name: str) -> None:
        """Drop a table.

        Args:
            table_name: Fully qualified table name.
        """
        self.execute_metadata_query(f"DROP TABLE IF EXISTS {table_name}")
        logger.info(f"Dropped table: {table_name}")

    # -------------------------------------------------------------------------
    # Comparison methods
    # -------------------------------------------------------------------------

    def compare_tables(
        self,
        base_table: str,
        ci_table: str,
    ) -> dict[str, Any]:
        """Compare metadata between base (prod) and CI tables.

        Args:
            base_table: Production table name.
            ci_table: CI table name.

        Returns:
            Comparison results with schema diffs, row count diffs, etc.
        """
        results = {
            "base_table": base_table,
            "ci_table": ci_table,
            "row_count": {},
            "schema_changes": [],
            "null_changes": [],
            "has_diff": False,
        }

        # Row count comparison
        try:
            base_count = self.get_row_count(base_table)
            ci_count = self.get_row_count(ci_table)

            results["row_count"] = {
                "base": base_count,
                "ci": ci_count,
                "diff": ci_count - base_count,
                "pct_change": round((ci_count - base_count) * 100.0 / base_count, 2) if base_count > 0 else None,
            }

            if results["row_count"]["diff"] != 0:
                results["has_diff"] = True
        except Exception as e:
            results["row_count_error"] = str(e)

        # Schema comparison
        try:
            base_schema = self.get_table_schema(base_table)
            ci_schema = self.get_table_schema(ci_table)

            base_cols = {
                row.get("col_name", row.get("column_name", "")): row.get("data_type", row.get("type", ""))
                for row in base_schema
                if not row.get("col_name", row.get("column_name", "")).startswith("#")
            }
            ci_cols = {
                row.get("col_name", row.get("column_name", "")): row.get("data_type", row.get("type", ""))
                for row in ci_schema
                if not row.get("col_name", row.get("column_name", "")).startswith("#")
            }

            for col, dtype in ci_cols.items():
                if col not in base_cols:
                    results["schema_changes"].append({
                        "column": col,
                        "change": "ADDED",
                        "type": dtype,
                    })
                elif base_cols[col] != dtype:
                    results["schema_changes"].append({
                        "column": col,
                        "change": "TYPE_CHANGED",
                        "base_type": base_cols[col],
                        "ci_type": dtype,
                    })

            for col in base_cols:
                if col not in ci_cols:
                    results["schema_changes"].append({
                        "column": col,
                        "change": "REMOVED",
                        "type": base_cols[col],
                    })

            if results["schema_changes"]:
                results["has_diff"] = True
        except Exception as e:
            results["schema_error"] = str(e)

        # NULL count comparison for common columns
        try:
            common_cols = set(base_cols.keys()) & set(ci_cols.keys())
            for col in list(common_cols)[:10]:  # Limit to first 10 columns for performance
                try:
                    base_nulls = self.get_null_count(base_table, col)
                    ci_nulls = self.get_null_count(ci_table, col)

                    if base_nulls != ci_nulls:
                        results["null_changes"].append({
                            "column": col,
                            "base_nulls": base_nulls,
                            "ci_nulls": ci_nulls,
                            "diff": ci_nulls - base_nulls,
                        })
                except Exception:
                    pass

            if results["null_changes"]:
                results["has_diff"] = True
        except Exception as e:
            results["null_count_error"] = str(e)

        return results

    def get_new_table_metadata(self, table_name: str) -> dict[str, Any]:
        """Get metadata for a new table (not in prod).

        Returns schema, row count, NULL stats, and uniqueness metrics.

        Args:
            table_name: Table name.

        Returns:
            Metadata dict with schema and stats.
        """
        return self.get_table_metadata(table_name)
