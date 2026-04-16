"""
ERP database — schema, seed data, and helper functions.
Uses PostgreSQL (Supabase) via psycopg2 with a thin sqlite3-compatible wrapper.
"""

import os
import psycopg2
import psycopg2.extras

# Set DATABASE_URL as an environment variable to override the default.
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres.pmoemhlanxuohteuycqv:7zMC0jNFJf4422Dl@aws-0-eu-west-1.pooler.supabase.com:6543/postgres"
)

CREDIT_LIMIT = 200_000.0

# Canonical customer list — single source of truth for seed data
# (customer_no, name, discount_pct, auto_approve)
_CUSTOMER_SEED: list[tuple[str, str, float, int]] = [
    ("32127188-4", "Navy Seal",   0.20, 1),
    ("33127198-4", "Bundeswehr",  0.20, 1),
    ("87232099-1", "Royal Navy",  0.10, 0),
    ("87233099-1", "Flyvevåbnet", 0.05, 0),
]

# Figure permissions per customer (from approved product matrix)
_PERMISSION_SEED: dict[str, dict[str, int]] = {
    "32127188-4": {"A1234H15": 1, "B1375A23": 1, "B2378F81": 0, "B8555G23": 1, "B1375A22": 1},
    "33127198-4": {"A1234H15": 1, "B1375A23": 1, "B2378F81": 0, "B8555G23": 0, "B1375A22": 0},
    "87232099-1": {"A1234H15": 1, "B1375A23": 1, "B2378F81": 1, "B8555G23": 1, "B1375A22": 1},
    "87233099-1": {"A1234H15": 1, "B1375A23": 0, "B2378F81": 0, "B8555G23": 0, "B1375A22": 1},
}

# Derived constants (kept for backward compatibility)
AUTO_APPROVE_CUSTOMERS: set[str] = {name for _, name, _, auto in _CUSTOMER_SEED if auto}
CUSTOMER_DISCOUNTS: dict[str, float] = {name: disc for _, name, disc, _ in _CUSTOMER_SEED}

# Component catalogue: colour → (varenummer, unit_price)
COMPONENTS: dict[str, tuple[str, float]] = {
    "red":    ("H14-17",  4_000.0),
    "blue":   ("B43-13",  3_000.0),
    "green":  ("H15-17",  6_000.0),
    "yellow": ("A248-2",  4_000.0),
    "white":  ("A247-1", 26_000.0),
    "orange": ("S992-44", 6_000.0),
    "purple": ("B43-12",  6_000.0),
}

# List prices per figure (kr.)
FIGURE_LIST_PRICES: dict[str, float] = {
    "B8555G23": 65_000.0,
    "B2378F81": 64_000.0,
    "B1375A23": 45_000.0,
    "A1234H15": 57_000.0,
    "B1375A22": 61_000.0,
}

# Component seed: one row per unique component per figure
# (varenummer, quantity-used, unit_price)
_COMPONENT_SEED: dict[str, list[tuple[str, int, float]]] = {
    "B8555G23": [  # Blue×2, Yellow×2, Green×1, White×1
        ("B43-13",  2,  3_000.0),
        ("A248-2",  2,  4_000.0),
        ("H15-17",  1,  6_000.0),
        ("A247-1",  1, 26_000.0),
    ],
    "B2378F81": [  # White×1, Purple×1, Red×1, Green×1, Blue×1
        ("A247-1",  1, 26_000.0),
        ("B43-12",  1,  6_000.0),
        ("H14-17",  1,  4_000.0),
        ("H15-17",  1,  6_000.0),
        ("B43-13",  1,  3_000.0),
    ],
    "B1375A23": [  # Orange×2, Purple×2, Yellow×1
        ("S992-44", 2,  6_000.0),
        ("B43-12",  2,  6_000.0),
        ("A248-2",  1,  4_000.0),
    ],
    "A1234H15": [  # Red×2, White×1, Yellow×1
        ("H14-17",  2,  4_000.0),
        ("A247-1",  1, 26_000.0),
        ("A248-2",  1,  4_000.0),
    ],
    "B1375A22": [  # Red×1, Green×2, White×1
        ("H14-17",  1,  4_000.0),
        ("H15-17",  2,  6_000.0),
        ("A247-1",  1, 26_000.0),
    ],
}

STATUS_LABELS: dict[str, str] = {
    "ny_ordre":            "Ny ordre",
    "oekonomi_check":      "Økonomi check",
    "pending_kapital":     "Afventer kapital",
    "kapital_ok":          "Kapital OK",
    "pending_kunde":       "Afventer kunde",
    "kunde_godkendt":      "Kunde godkendt",
    "indkoeb_afventer":    "Indkøb afventer",
    "klodser_hentet":      "Klodser hentet",
    "klar_til_produktion": "Klar til produktion",
    "i_produktion":        "I produktion",
    "klar_til_qc":         "Klar til QC",
    "paa_lager":           "På lager",
    "klar_til_afhentning": "Klar til afhentning",
    "afhentet":            "Afhentet",
    "faktureret":          "Faktureret",
    "afvist":              "Afvist",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id           SERIAL PRIMARY KEY,
    customer_no  TEXT UNIQUE NOT NULL,
    name         TEXT NOT NULL,
    discount_pct REAL DEFAULT 0.0,
    auto_approve INTEGER DEFAULT 0,
    created_at   TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS customer_permissions (
    customer_no  TEXT NOT NULL,
    figure_id    TEXT NOT NULL,
    approved     INTEGER DEFAULT 1,
    PRIMARY KEY (customer_no, figure_id)
);

CREATE TABLE IF NOT EXISTS customer_orders (
    id                      SERIAL PRIMARY KEY,
    customer_order_no       TEXT UNIQUE NOT NULL,
    customer                TEXT NOT NULL,
    expected_delivery_at    TIMESTAMP,
    payment_request_sent    INTEGER DEFAULT 0,
    production_notified     INTEGER DEFAULT 0,
    prod_pickup_requested   INTEGER DEFAULT 0,
    prod_delivery_confirmed INTEGER DEFAULT 0,
    salg_pickup_requested   INTEGER DEFAULT 0,
    salg_delivery_confirmed INTEGER DEFAULT 0,
    holdeplads              TEXT,
    produced_at             TIMESTAMP,
    qc_result               TEXT,
    created_at              TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS orders (
    id                      SERIAL PRIMARY KEY,
    order_id                TEXT UNIQUE NOT NULL,
    customer_order_no       TEXT,
    customer                TEXT NOT NULL,
    figure_id               TEXT NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'ny_ordre',
    holdeplads              TEXT,
    quantity                INTEGER DEFAULT 1,
    estimated_cost          REAL,
    actual_cost             REAL,
    selling_price           REAL,
    customer_rating         INTEGER,
    qc_result               TEXT,
    prod_pickup_requested   INTEGER DEFAULT 0,
    prod_delivery_confirmed INTEGER DEFAULT 0,
    salg_pickup_requested   INTEGER DEFAULT 0,
    salg_delivery_confirmed INTEGER DEFAULT 0,
    payment_request_sent    INTEGER DEFAULT 0,
    production_notified     INTEGER DEFAULT 0,
    produced_at             TIMESTAMP,
    created_at              TIMESTAMP DEFAULT NOW(),
    updated_at              TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS invoices (
    id           SERIAL PRIMARY KEY,
    order_id     TEXT NOT NULL,
    customer     TEXT,
    amount       REAL,
    invoice_date DATE,
    created_at   TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS payments (
    id          SERIAL PRIMARY KEY,
    order_id    TEXT NOT NULL,
    supplier    TEXT,
    amount      REAL,
    invoice_ref TEXT,
    status      TEXT DEFAULT 'ikke_betalt',
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS component_prices (
    figure_id  TEXT NOT NULL,
    component  TEXT NOT NULL,
    quantity   INTEGER,
    unit_price REAL
);

CREATE TABLE IF NOT EXISTS order_events (
    id         SERIAL PRIMARY KEY,
    order_id   TEXT NOT NULL,
    status     TEXT NOT NULL,
    department TEXT,
    note       TEXT,
    timestamp  TIMESTAMP NOT NULL
);
"""


# ---------------------------------------------------------------------------
# sqlite3-compatible wrapper so app.py needs no changes
# ---------------------------------------------------------------------------

class _Row:
    """Row object supporting both row["name"] and row[0] access."""

    def __init__(self, keys: list, values: tuple):
        self._data = dict(zip(keys, values))
        self._values = list(values)
        self._keys = keys

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        return self._data[key]

    def __iter__(self):
        return iter(self._values)

    def keys(self):
        return self._keys

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __repr__(self):
        return repr(self._data)


class _Cursor:
    """Wraps a psycopg2 cursor with a sqlite3-compatible interface.
    Converts ? placeholders to %s automatically."""

    def __init__(self, cur):
        self._cur = cur

    @staticmethod
    def _adapt(sql: str) -> str:
        return sql.replace("?", "%s")

    def execute(self, sql: str, params=()):
        self._cur.execute(self._adapt(sql), params)
        return self

    def executemany(self, sql: str, seq):
        psycopg2.extras.execute_batch(self._cur, self._adapt(sql), seq)
        return self

    def _wrap(self, row):
        if row is None:
            return None
        keys = [d[0] for d in self._cur.description]
        return _Row(keys, row)

    def fetchone(self):
        return self._wrap(self._cur.fetchone())

    def fetchall(self):
        if self._cur.description is None:
            return []
        keys = [d[0] for d in self._cur.description]
        return [_Row(keys, r) for r in self._cur.fetchall()]

    def __iter__(self):
        if self._cur.description is None:
            return
        keys = [d[0] for d in self._cur.description]
        for row in self._cur:
            yield _Row(keys, row)


class _Connection:
    """Wraps a psycopg2 connection with a sqlite3-compatible interface."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql: str, params=()) -> _Cursor:
        cur = self._conn.cursor()
        c = _Cursor(cur)
        c.execute(sql, params)
        return c

    def executemany(self, sql: str, seq) -> _Cursor:
        cur = self._conn.cursor()
        c = _Cursor(cur)
        c.executemany(sql, seq)
        return c

    def executescript(self, sql: str):
        """Run multiple semicolon-separated SQL statements."""
        cur = self._conn.cursor()
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(stmt)
        return _Cursor(cur)

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._conn.rollback()
        else:
            self._conn.commit()
        self._conn.close()


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def get_connection() -> _Connection:
    conn = psycopg2.connect(DATABASE_URL)
    return _Connection(conn)


def _add_column_if_missing(conn: _Connection,
                           table: str, column: str, col_type: str) -> None:
    row = conn.execute(
        "SELECT column_name FROM information_schema.columns"
        " WHERE table_name = ? AND column_name = ?",
        (table, column),
    ).fetchone()
    if row is None:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def init_db(conn: _Connection) -> None:
    conn.executescript(_SCHEMA)

    # Migrations for existing databases
    _add_column_if_missing(conn, "orders", "selling_price",       "REAL")
    _add_column_if_missing(conn, "orders", "quantity",            "INTEGER DEFAULT 1")
    _add_column_if_missing(conn, "orders", "customer_rating",     "INTEGER")
    _add_column_if_missing(conn, "orders", "customer_order_no",   "TEXT")
    for col, typ in [
        ("payment_request_sent",    "INTEGER DEFAULT 0"),
        ("production_notified",     "INTEGER DEFAULT 0"),
        ("prod_pickup_requested",   "INTEGER DEFAULT 0"),
        ("prod_delivery_confirmed", "INTEGER DEFAULT 0"),
        ("salg_pickup_requested",   "INTEGER DEFAULT 0"),
        ("salg_delivery_confirmed", "INTEGER DEFAULT 0"),
        ("holdeplads",              "TEXT"),
        ("produced_at",             "TIMESTAMP"),
        ("qc_result",               "TEXT"),
    ]:
        _add_column_if_missing(conn, "customer_orders", col, typ)

    # Reseed component prices if the catalogue has changed
    existing = {
        r[0]
        for r in conn.execute("SELECT DISTINCT component FROM component_prices").fetchall()
    }
    expected = {comp for rows in _COMPONENT_SEED.values() for comp, _, _ in rows}
    if not expected.issubset(existing):
        conn.execute("DELETE FROM component_prices")
        for fig_id, rows in _COMPONENT_SEED.items():
            conn.executemany(
                "INSERT INTO component_prices (figure_id, component, quantity, unit_price)"
                " VALUES (?, ?, ?, ?)",
                [(fig_id, comp, qty, price) for comp, qty, price in rows],
            )

    # Seed customers + permissions if table is empty
    if conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 0:
        conn.executemany(
            "INSERT INTO customers (customer_no, name, discount_pct, auto_approve)"
            " VALUES (?, ?, ?, ?) ON CONFLICT (customer_no) DO NOTHING",
            _CUSTOMER_SEED,
        )
        for customer_no, perms in _PERMISSION_SEED.items():
            conn.executemany(
                "INSERT INTO customer_permissions (customer_no, figure_id, approved)"
                " VALUES (?, ?, ?) ON CONFLICT (customer_no, figure_id) DO NOTHING",
                [(customer_no, fig, approved) for fig, approved in perms.items()],
            )

    conn.commit()


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------

def figure_cost(conn: _Connection, figure_id: str) -> float:
    """Total component cost for one figure (used for credit-limit checking)."""
    rows = conn.execute(
        "SELECT quantity, unit_price FROM component_prices WHERE figure_id = ?",
        (figure_id,),
    ).fetchall()
    return sum(r["quantity"] * r["unit_price"] for r in rows)


def figure_selling_price(figure_id: str, customer: str) -> float:
    """List price minus the customer's discount."""
    list_price = FIGURE_LIST_PRICES.get(figure_id, 0.0)
    discount   = CUSTOMER_DISCOUNTS.get(customer, 0.0)
    return round(list_price * (1.0 - discount), 2)


def active_orders_cost(conn: _Connection) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(estimated_cost), 0) FROM orders"
        " WHERE status NOT IN ('afvist', 'faktureret') AND estimated_cost IS NOT NULL"
    ).fetchone()
    return row[0] or 0.0


def available_credit(conn: _Connection) -> float:
    return CREDIT_LIMIT - active_orders_cost(conn)


def profit_liquidity(conn: _Connection) -> float:
    """Overskudslikviditet = faktureret omsætning − betalte leverandøromkostninger."""
    revenue = conn.execute(
        "SELECT COALESCE(SUM(amount),0) FROM invoices"
    ).fetchone()[0] or 0.0
    costs = conn.execute(
        "SELECT COALESCE(SUM(amount),0) FROM payments WHERE status='betalt'"
    ).fetchone()[0] or 0.0
    return max(0.0, revenue - costs)


def total_available(conn: _Connection) -> float:
    """Samlet rådighedsbeløb = kassekredit til rådighed + overskudslikviditet."""
    return available_credit(conn) + profit_liquidity(conn)


def next_order_id(conn: _Connection) -> str:
    row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM orders").fetchone()
    return f"ORD-{row[0] + 1:04d}"


def next_customer_order_no(conn: _Connection) -> str:
    row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM customer_orders").fetchone()
    return f"KO-{row[0] + 1:04d}"


def assign_holdeplads(conn: _Connection) -> str:
    used = {
        r[0]
        for r in conn.execute(
            "SELECT holdeplads FROM orders"
            " WHERE holdeplads IS NOT NULL AND status NOT IN ('afvist','faktureret','afhentet')"
        ).fetchall()
    }
    for letter in "ABCD":
        for num in range(1, 10):
            slot = f"{letter}{num}"
            if slot not in used:
                return slot
    return "X1"


def get_customers(conn: _Connection) -> list:
    return conn.execute("SELECT * FROM customers ORDER BY name").fetchall()


def get_customer_by_name(conn: _Connection, name: str):
    return conn.execute("SELECT * FROM customers WHERE name=?", (name,)).fetchone()


def is_figure_allowed(conn: _Connection, customer_no: str, figure_id: str) -> bool:
    """Return True if the customer is permitted to order this figure.
    If no permission row exists, default to allowed (backward compat)."""
    row = conn.execute(
        "SELECT approved FROM customer_permissions WHERE customer_no=? AND figure_id=?",
        (customer_no, figure_id),
    ).fetchone()
    return row is None or bool(row["approved"])


def get_permissions_map(conn: _Connection) -> dict[str, dict[str, bool]]:
    """Return {customer_no: {figure_id: approved}} for all customers."""
    result: dict[str, dict[str, bool]] = {}
    for row in conn.execute("SELECT customer_no, figure_id, approved FROM customer_permissions"):
        result.setdefault(row["customer_no"], {})[row["figure_id"]] = bool(row["approved"])
    return result
