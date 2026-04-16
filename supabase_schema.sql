-- ============================================================
--  ERP-Games — Supabase / PostgreSQL schema + seed data
--  Kør dette script i Supabase SQL Editor
-- ============================================================

-- ────────────────────────────────────────────────────────────
--  TABELLER
-- ────────────────────────────────────────────────────────────

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


-- ────────────────────────────────────────────────────────────
--  SEED: Kunder
-- ────────────────────────────────────────────────────────────

INSERT INTO customers (customer_no, name, discount_pct, auto_approve) VALUES
    ('32127188-4', 'Navy Seal',   0.20, 1),
    ('33127198-4', 'Bundeswehr',  0.20, 1),
    ('87232099-1', 'Royal Navy',  0.10, 0),
    ('87233099-1', 'Flyvevåbnet', 0.05, 0)
ON CONFLICT (customer_no) DO NOTHING;


-- ────────────────────────────────────────────────────────────
--  SEED: Figurrettigheder per kunde
-- ────────────────────────────────────────────────────────────

INSERT INTO customer_permissions (customer_no, figure_id, approved) VALUES
    ('32127188-4', 'A1234H15', 1),
    ('32127188-4', 'B1375A23', 1),
    ('32127188-4', 'B2378F81', 0),
    ('32127188-4', 'B8555G23', 1),
    ('32127188-4', 'B1375A22', 1),
    ('33127198-4', 'A1234H15', 1),
    ('33127198-4', 'B1375A23', 1),
    ('33127198-4', 'B2378F81', 0),
    ('33127198-4', 'B8555G23', 0),
    ('33127198-4', 'B1375A22', 0),
    ('87232099-1', 'A1234H15', 1),
    ('87232099-1', 'B1375A23', 1),
    ('87232099-1', 'B2378F81', 1),
    ('87232099-1', 'B8555G23', 1),
    ('87232099-1', 'B1375A22', 1),
    ('87233099-1', 'A1234H15', 1),
    ('87233099-1', 'B1375A23', 0),
    ('87233099-1', 'B2378F81', 0),
    ('87233099-1', 'B8555G23', 0),
    ('87233099-1', 'B1375A22', 1)
ON CONFLICT (customer_no, figure_id) DO NOTHING;


-- ────────────────────────────────────────────────────────────
--  SEED: Komponentpriser per figur
-- ────────────────────────────────────────────────────────────

DELETE FROM component_prices;

INSERT INTO component_prices (figure_id, component, quantity, unit_price) VALUES
    -- B8555G23: Blå×2, Gul×2, Grøn×1, Hvid×1
    ('B8555G23', 'B43-13',  2,  3000.0),
    ('B8555G23', 'A248-2',  2,  4000.0),
    ('B8555G23', 'H15-17',  1,  6000.0),
    ('B8555G23', 'A247-1',  1, 26000.0),
    -- B2378F81: Hvid×1, Lilla×1, Rød×1, Grøn×1, Blå×1
    ('B2378F81', 'A247-1',  1, 26000.0),
    ('B2378F81', 'B43-12',  1,  6000.0),
    ('B2378F81', 'H14-17',  1,  4000.0),
    ('B2378F81', 'H15-17',  1,  6000.0),
    ('B2378F81', 'B43-13',  1,  3000.0),
    -- B1375A23: Orange×2, Lilla×2, Gul×1
    ('B1375A23', 'S992-44', 2,  6000.0),
    ('B1375A23', 'B43-12',  2,  6000.0),
    ('B1375A23', 'A248-2',  1,  4000.0),
    -- A1234H15: Rød×2, Hvid×1, Gul×1
    ('A1234H15', 'H14-17',  2,  4000.0),
    ('A1234H15', 'A247-1',  1, 26000.0),
    ('A1234H15', 'A248-2',  1,  4000.0),
    -- B1375A22: Rød×1, Grøn×2, Hvid×1
    ('B1375A22', 'H14-17',  1,  4000.0),
    ('B1375A22', 'H15-17',  2,  6000.0),
    ('B1375A22', 'A247-1',  1, 26000.0);
