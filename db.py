"""
ERP database — schema, seed data, and helper functions.
"""

import os
import sqlite3

DATABASE = os.path.join(os.path.dirname(__file__), "erp.db")

CREDIT_LIMIT = 200_000.0

AUTO_APPROVE_CUSTOMERS = {"Navy Seal", "Bundeswehr", "Royal Navy", "Flyvevåbnet"}

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

# Discount rates per customer (0.20 = 20 %)
CUSTOMER_DISCOUNTS: dict[str, float] = {
    "Navy Seal":   0.20,
    "Bundeswehr":  0.20,
    "Royal Navy":  0.10,
    "Flyvevåbnet": 0.05,
}

# Component seed: one row per unique component per figure
# (varenummer, quantity-used, unit_price)
_COMPONENT_SEED: dict[str, list[tuple[str, int, float]]] = {
    "B8555G23": [  # Blue×2, Yellow×2, Green×1, White×1
        ("B43-13",  2,  3_000.0),   # blå
        ("A248-2",  2,  4_000.0),   # gul
        ("H15-17",  1,  6_000.0),   # grøn
        ("A247-1",  1, 26_000.0),   # hvid
    ],
    "B2378F81": [  # White×1, Purple×1, Red×1, Green×1, Blue×1
        ("A247-1",  1, 26_000.0),   # hvid
        ("B43-12",  1,  6_000.0),   # lilla
        ("H14-17",  1,  4_000.0),   # rød
        ("H15-17",  1,  6_000.0),   # grøn
        ("B43-13",  1,  3_000.0),   # blå
    ],
    "B1375A23": [  # Orange×2, Purple×2, Yellow×1
        ("S992-44", 2,  6_000.0),   # orange
        ("B43-12",  2,  6_000.0),   # lilla
        ("A248-2",  1,  4_000.0),   # gul
    ],
    "A1234H15": [  # Red×2, White×1, Yellow×1
        ("H14-17",  2,  4_000.0),   # rød
        ("A247-1",  1, 26_000.0),   # hvid
        ("A248-2",  1,  4_000.0),   # gul
    ],
    "B1375A22": [  # Red×1, Green×2, White×1
        ("H14-17",  1,  4_000.0),   # rød
        ("H15-17",  2,  6_000.0),   # grøn
        ("A247-1",  1, 26_000.0),   # hvid
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
CREATE TABLE IF NOT EXISTS customer_orders (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_order_no    TEXT UNIQUE NOT NULL,
    customer             TEXT NOT NULL,
    expected_delivery_at DATETIME,
    created_at           DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
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
    produced_at             DATETIME,
    customer_rating         INTEGER,
    created_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at              DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS invoices (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id     TEXT NOT NULL,
    customer     TEXT,
    amount       REAL,
    invoice_date DATE,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS payments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id     TEXT NOT NULL,
    supplier     TEXT,
    amount       REAL,
    invoice_ref  TEXT,
    status       TEXT DEFAULT 'ikke_betalt',
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS component_prices (
    figure_id   TEXT NOT NULL,
    component   TEXT NOT NULL,
    quantity    INTEGER,
    unit_price  REAL
);

CREATE TABLE IF NOT EXISTS order_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    TEXT NOT NULL,
    status      TEXT NOT NULL,
    department  TEXT,
    note        TEXT,
    timestamp   DATETIME NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def _add_column_if_missing(conn: sqlite3.Connection,
                           table: str, column: str, col_type: str) -> None:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    # Migrations for existing databases
    _add_column_if_missing(conn, "orders", "selling_price",       "REAL")
    _add_column_if_missing(conn, "orders", "quantity",            "INTEGER DEFAULT 1")
    _add_column_if_missing(conn, "orders", "customer_rating",     "INTEGER")
    _add_column_if_missing(conn, "orders", "customer_order_no",   "TEXT")

    # Reseed component prices if the catalogue has changed (check for new varenumre)
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
    conn.commit()


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------

def figure_cost(conn: sqlite3.Connection, figure_id: str) -> float:
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


def active_orders_cost(conn: sqlite3.Connection) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(estimated_cost), 0) FROM orders"
        " WHERE status NOT IN ('afvist', 'faktureret') AND estimated_cost IS NOT NULL"
    ).fetchone()
    return row[0] or 0.0


def available_credit(conn: sqlite3.Connection) -> float:
    return CREDIT_LIMIT - active_orders_cost(conn)


def profit_liquidity(conn: sqlite3.Connection) -> float:
    """Overskudslikviditet = faktureret omsætning − betalte leverandøromkostninger."""
    revenue = conn.execute(
        "SELECT COALESCE(SUM(amount),0) FROM invoices"
    ).fetchone()[0] or 0.0
    costs = conn.execute(
        "SELECT COALESCE(SUM(amount),0) FROM payments WHERE status='betalt'"
    ).fetchone()[0] or 0.0
    return max(0.0, revenue - costs)


def total_available(conn: sqlite3.Connection) -> float:
    """Samlet rådighedsbeløb = kassekredit til rådighed + overskudslikviditet."""
    return available_credit(conn) + profit_liquidity(conn)


def next_order_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM orders").fetchone()
    return f"ORD-{row[0] + 1:04d}"


def next_customer_order_no(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM customer_orders").fetchone()
    return f"KO-{row[0] + 1:04d}"


def assign_holdeplads(conn: sqlite3.Connection) -> str:
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
