import os
import tempfile
import threading
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

_TZ = ZoneInfo("Europe/Copenhagen")

import cv2
from flask import (Flask, Response, flash, g, jsonify,
                   redirect, render_template, request, session, url_for)

import db as _db
from cv.pipeline import build_palette, detect_layers, load_palette, save_palette
from data.references import REFERENCE_SEQUENCES

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 1))

# Folder where one reference image per figure is stored
REFS_DIR = os.path.join(os.path.dirname(__file__), "static", "refs")
os.makedirs(REFS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _ref_path(figure_id: str) -> str | None:
    """Return the path to the reference image for figure_id, or None."""
    for ext in ALLOWED_EXTENSIONS:
        p = os.path.join(REFS_DIR, figure_id + ext)
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Thread-safe camera
# ---------------------------------------------------------------------------

class Camera:
    def __init__(self, index: int = 0):
        self._index = index
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._open()

    def _open(self):
        self._cap = cv2.VideoCapture(self._index)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def read(self) -> cv2.typing.MatLike | None:
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                self._open()
            ret, frame = self._cap.read()
            return frame if ret else None

    def release(self):
        with self._lock:
            if self._cap and self._cap.isOpened():
                self._cap.release()
            self._cap = None


camera = Camera(CAMERA_INDEX)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db() -> object:
    if "db" not in g:
        g.db = _db.get_connection()
    return g.db


@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db:
        db.close()


@app.before_request
def ensure_db():
    _db.init_db(get_db())


# ---------------------------------------------------------------------------
# Template globals
# ---------------------------------------------------------------------------

@app.context_processor
def inject_globals():
    db = get_db()
    salg_badge = db.execute(
        "SELECT COUNT(*) FROM orders WHERE status IN ('pending_kunde','klar_til_afhentning')"
    ).fetchone()[0]

    oekonomi_badge = db.execute(
        "SELECT (SELECT COUNT(*) FROM orders WHERE status='pending_kapital')"
        " + (SELECT COUNT(*) FROM payments WHERE status='ikke_betalt')"
    ).fetchone()[0]

    logistik_badge = db.execute(
        "SELECT COUNT(*) FROM orders WHERE status IN"
        " ('klodser_hentet','paa_lager')"
        " OR (status='klar_til_produktion' AND prod_pickup_requested=1)"
        " OR (status='klar_til_afhentning' AND salg_pickup_requested=1)"
    ).fetchone()[0]

    indkoeb_badge = db.execute(
        "SELECT COUNT(*) FROM orders WHERE status='indkoeb_afventer'"
    ).fetchone()[0]

    produktion_badge = db.execute(
        "SELECT COUNT(*) FROM orders WHERE status='klar_til_produktion'"
    ).fetchone()[0]

    qc_badge = db.execute(
        "SELECT COUNT(*) FROM orders WHERE status='klar_til_qc'"
    ).fetchone()[0]

    customers = get_db().execute(
        "SELECT customer_no, name FROM customers ORDER BY name"
    ).fetchall()

    return {
        "badges": {
            "salg": salg_badge,
            "oekonomi": oekonomi_badge,
            "logistik": logistik_badge,
            "indkoeb": indkoeb_badge,
            "produktion": produktion_badge,
            "qc": qc_badge,
        },
        "status_labels": _db.STATUS_LABELS,
        "figures": list(REFERENCE_SEQUENCES.keys()),
        "all_customers": customers,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Maps status → department name (used for event log)
_STATUS_DEPT: dict[str, str] = {
    "ny_ordre":            "Salg",
    "oekonomi_check":      "Økonomi",
    "pending_kapital":     "Økonomi",
    "kapital_ok":          "Økonomi",
    "pending_kunde":       "Salg",
    "kunde_godkendt":      "Salg",
    "indkoeb_afventer":    "Indkøb",
    "klodser_hentet":      "Logistik",
    "klar_til_produktion": "Logistik",
    "i_produktion":        "Produktion",
    "klar_til_qc":         "Produktion",
    "paa_lager":           "S&K / Logistik",
    "klar_til_afhentning": "Logistik",
    "afhentet":            "Salg",
    "faktureret":          "Økonomi",
    "afvist":              "—",
}


def _now() -> str:
    return datetime.now(_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _log_event(db, order_id: str, status: str, note: str | None = None):
    """Append one row to order_events."""
    db.execute(
        "INSERT INTO order_events (order_id, status, department, note, timestamp)"
        " VALUES (?, ?, ?, ?, ?)",
        (order_id, status, _STATUS_DEPT.get(status, "—"), note, _now()),
    )


def _update_status(db, order_id: str, status: str, extra: dict | None = None):
    fields = {"status": status, "updated_at": _now()}
    if extra:
        fields.update(extra)
    set_clause = ", ".join(f"{k}=?" for k in fields)
    db.execute(
        f"UPDATE orders SET {set_clause} WHERE order_id=?",
        list(fields.values()) + [order_id],
    )
    _log_event(db, order_id, status)
    db.commit()


def _update_ko_status(db, ko_no: str, status: str, extra: dict | None = None):
    """Move ALL orders in a KO to the same status, and update the KO row if extra has KO fields."""
    fields = {"status": status, "updated_at": _now()}
    # Fields that live on the orders table
    order_extra = {k: v for k, v in (extra or {}).items()
                   if k in ("qc_result", "actual_cost", "produced_at", "holdeplads", "customer_rating")}
    # Fields that live on the customer_orders table
    ko_extra = {k: v for k, v in (extra or {}).items()
                if k in ("holdeplads", "produced_at", "qc_result")}
    if order_extra:
        fields.update(order_extra)
    set_clause = ", ".join(f"{k}=?" for k in fields)
    db.execute(
        f"UPDATE orders SET {set_clause} WHERE customer_order_no=?",
        list(fields.values()) + [ko_no],
    )
    if ko_extra:
        ko_set = ", ".join(f"{k}=?" for k in ko_extra)
        db.execute(
            f"UPDATE customer_orders SET {ko_set} WHERE customer_order_no=?",
            list(ko_extra.values()) + [ko_no],
        )
    # Log event on each order in the KO
    for row in db.execute("SELECT order_id FROM orders WHERE customer_order_no=?", (ko_no,)).fetchall():
        _log_event(db, row["order_id"], status)
    db.commit()


def _ko_no_of(db, order_id: str) -> str | None:
    """Return the customer_order_no for an order_id."""
    row = db.execute("SELECT customer_order_no FROM orders WHERE order_id=?", (order_id,)).fetchone()
    return row["customer_order_no"] if row and row["customer_order_no"] else None


def _compare_sequences(detected: list[str], expected: list[str]) -> dict:
    errors = []
    for i in range(max(len(detected), len(expected))):
        det = detected[i] if i < len(detected) else None
        exp = expected[i] if i < len(expected) else None
        if det != exp:
            errors.append({"layer": i + 1, "expected": exp, "detected": det})
    return {
        "result":   "pass" if not errors else "fail",
        "detected": detected,
        "expected": expected,
        "errors":   errors,
    }


def _mjpeg_frames():
    while True:
        frame = camera.read()
        if frame is None:
            import numpy as np
            placeholder = (np.ones((240, 320, 3), dtype="uint8") * 127)
            ret, buf = cv2.imencode(".jpg", placeholder)
        else:
            ret, buf = cv2.imencode(".jpg", frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes() + b"\r\n"
            )
        time.sleep(1 / 30)


# ===========================================================================
# DASHBOARD
# ===========================================================================

@app.get("/")
def dashboard():
    db = get_db()
    orders = db.execute(
        "SELECT o.*, co.expected_delivery_at FROM orders o"
        " LEFT JOIN customer_orders co ON co.customer_order_no = o.customer_order_no"
        " ORDER BY o.updated_at DESC"
    ).fetchall()
    return render_template("dashboard.html", orders=orders)


@app.get("/api/kpis")
def api_kpis():
    db = get_db()
    today = date.today().isoformat()
    active = db.execute(
        "SELECT COUNT(*) FROM orders WHERE status NOT IN ('afvist','faktureret')"
    ).fetchone()[0]
    pending = db.execute(
        "SELECT COUNT(*) FROM orders WHERE status IN ('pending_kapital','pending_kunde')"
    ).fetchone()[0]
    paa_lager = db.execute(
        "SELECT COUNT(*) FROM orders WHERE status='paa_lager'"
    ).fetchone()[0]
    produced_today = db.execute(
        "SELECT COUNT(*) FROM orders WHERE produced_at LIKE ?",
        (f"{today}%",),
    ).fetchone()[0]
    todays_revenue = db.execute(
        "SELECT COALESCE(SUM(amount),0) FROM invoices WHERE invoice_date=?",
        (today,),
    ).fetchone()[0]
    todays_supplier_costs = db.execute(
        "SELECT COALESCE(SUM(amount), 0) FROM payments"
        " WHERE DATE(created_at) = ?",
        (today,),
    ).fetchone()[0]
    todays_profit = round(todays_revenue - todays_supplier_costs, 0)
    total_revenue = db.execute(
        "SELECT COALESCE(SUM(amount),0) FROM invoices"
    ).fetchone()[0]
    total_supplier_costs = db.execute(
        "SELECT COALESCE(SUM(amount), 0) FROM payments"
    ).fetchone()[0]
    rating_row = db.execute(
        "SELECT ROUND(AVG(customer_rating),1), COUNT(customer_rating)"
        " FROM orders WHERE customer_rating IS NOT NULL"
    ).fetchone()

    # Penalty: for each KO past deadline with unfinished items
    now_str = _now()
    overdue_rows = db.execute(
        "SELECT co.customer_order_no, co.expected_delivery_at, co.customer,"
        " COALESCE(SUM(o.selling_price),0) as total_value"
        " FROM customer_orders co"
        " JOIN orders o ON o.customer_order_no = co.customer_order_no"
        " WHERE co.expected_delivery_at IS NOT NULL"
        "   AND co.expected_delivery_at < ?"
        "   AND o.status NOT IN ('faktureret','afvist')"
        " GROUP BY co.customer_order_no",
        (now_str,),
    ).fetchall()

    total_penalty = 0.0
    overdue_list  = []
    for row in overdue_rows:
        try:
            deadline = datetime.strptime(row["expected_delivery_at"], "%Y-%m-%d %H:%M:%S")
            now_dt   = datetime.now(_TZ).replace(tzinfo=None)
            mins_late = max(0, int((now_dt - deadline).total_seconds() / 60))
        except (ValueError, TypeError):
            mins_late = 0
        penalty = round(row["total_value"] * 0.01 * mins_late, 0)
        total_penalty += penalty
        overdue_list.append({
            "ko_no":     row["customer_order_no"],
            "customer":  row["customer"],
            "mins_late": mins_late,
            "penalty":   penalty,
        })

    total_profit = round(total_revenue - total_supplier_costs - total_penalty, 0)

    return jsonify({
        "active_orders":    active,
        "pending_approval": pending,
        "paa_lager":        paa_lager,
        "produced_today":   produced_today,
        "todays_revenue":   todays_revenue,
        "avg_rating":       rating_row[0],
        "rating_count":     rating_row[1],
        "total_penalty":    total_penalty,
        "overdue_count":    len(overdue_list),
        "overdue":          overdue_list,
        "todays_profit":    todays_profit,
        "total_profit":     total_profit,
    })


# ===========================================================================
# SALG
# ===========================================================================

@app.get("/salg")
def salg():
    db = get_db()
    pending_kunde = db.execute(
        "SELECT o.*, co.customer_order_no as ko_no, co.expected_delivery_at"
        " FROM orders o"
        " LEFT JOIN customer_orders co ON co.customer_order_no = o.customer_order_no"
        " WHERE o.status='pending_kunde' ORDER BY o.created_at"
    ).fetchall()
    # KOs with klar_til_afhentning — one row per KO
    klar_kos = db.execute(
        "SELECT co.* FROM customer_orders co"
        " WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_order_no=co.customer_order_no"
        "               AND o.status='klar_til_afhentning')"
        " ORDER BY co.created_at DESC"
    ).fetchall()
    klar_ko_orders = {
        ko["customer_order_no"]: db.execute(
            "SELECT * FROM orders WHERE customer_order_no=? AND status='klar_til_afhentning'",
            (ko["customer_order_no"],),
        ).fetchall()
        for ko in klar_kos
    }

    # KOs afhentet awaiting invoice — one row per KO
    afhentet_kos = db.execute(
        "SELECT co.* FROM customer_orders co"
        " WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_order_no=co.customer_order_no"
        "               AND o.status='afhentet')"
        "   AND NOT EXISTS (SELECT 1 FROM invoices i"
        "                    JOIN orders o ON i.order_id=o.order_id"
        "                   WHERE o.customer_order_no=co.customer_order_no)"
        " ORDER BY co.created_at DESC"
    ).fetchall()
    afhentet_ko_orders = {
        ko["customer_order_no"]: db.execute(
            "SELECT * FROM orders WHERE customer_order_no=? AND status='afhentet'",
            (ko["customer_order_no"],),
        ).fetchall()
        for ko in afhentet_kos
    }

    # Open KOs for adding more items
    open_kos = db.execute(
        "SELECT co.*, COUNT(o.id) as item_count"
        " FROM customer_orders co"
        " LEFT JOIN orders o ON o.customer_order_no = co.customer_order_no"
        " WHERE co.expected_delivery_at > ? OR co.expected_delivery_at IS NULL"
        " GROUP BY co.customer_order_no"
        " ORDER BY co.created_at DESC LIMIT 20",
        (_now(),),
    ).fetchall()
    perms = _db.get_permissions_map(db)
    return render_template("salg.html",
                           pending_kunde=pending_kunde,
                           klar_kos=klar_kos,
                           klar_ko_orders=klar_ko_orders,
                           afhentet_kos=afhentet_kos,
                           afhentet_ko_orders=afhentet_ko_orders,
                           open_kos=open_kos,
                           perms=perms,
                           now=_now())


def _create_order_line(db, customer_order_no: str, customer: str, figure_id: str,
                       quantity: int = 1) -> str:
    """Create one order line item under a KO. Returns order_id."""
    order_id      = _db.next_order_id(db)
    unit_cost     = _db.figure_cost(db, figure_id)
    total_cost    = unit_cost * quantity
    c_row         = _db.get_customer_by_name(db, customer)
    discount      = c_row["discount_pct"] if c_row else _db.CUSTOMER_DISCOUNTS.get(customer, 0.0)
    list_price    = _db.FIGURE_LIST_PRICES.get(figure_id, 0.0)
    selling_price = round(list_price * (1.0 - discount) * quantity, 2)

    db.execute(
        "INSERT INTO orders"
        " (order_id, customer_order_no, customer, figure_id, status,"
        "  quantity, estimated_cost, selling_price)"
        " VALUES (?, ?, ?, ?, 'oekonomi_check', ?, ?, ?)",
        (order_id, customer_order_no, customer, figure_id, quantity, total_cost, selling_price),
    )
    _log_event(db, order_id, "ny_ordre",
               f"Oprettet under {customer_order_no} for {customer} (antal: {quantity})")
    db.commit()

    # Auto økonomi check — brug kassekredit + overskudslikviditet
    available  = _db.total_available(db)
    new_status = "kapital_ok" if available >= total_cost else "pending_kapital"
    _update_status(db, order_id, new_status)

    if new_status == "kapital_ok":
        c_row   = _db.get_customer_by_name(db, customer)
        is_auto = bool(c_row["auto_approve"]) if c_row else customer in _db.AUTO_APPROVE_CUSTOMERS
        if is_auto:
            _update_status(db, order_id, "indkoeb_afventer")
        else:
            _update_status(db, order_id, "pending_kunde")

    return order_id


@app.post("/salg/ny_kundeordre")
def salg_ny_kundeordre():
    """Create a new customer order (KO) with one or more figures."""
    db = get_db()
    customer      = request.form.get("customer", "").strip()
    delivery_mins = request.form.get("delivery_minutes", "").strip()
    figure_ids  = request.form.getlist("figure_id")
    quantities  = request.form.getlist("quantity")
    figure_ids  = [f for f in figure_ids if f in REFERENCE_SEQUENCES]

    if not customer or not figure_ids:
        flash("Udfyld kundenavn og mindst én figur.", "error")
        return redirect(url_for("salg"))

    # Permission check
    c_row = _db.get_customer_by_name(db, customer)
    if c_row:
        blocked = [f for f in figure_ids if not _db.is_figure_allowed(db, c_row["customer_no"], f)]
        if blocked:
            flash(f"Kunden har ikke rettighed til: {', '.join(blocked)}.", "error")
            return redirect(url_for("salg"))

    # Expected delivery datetime
    expected_at = None
    if delivery_mins:
        try:
            mins = int(delivery_mins)
            from datetime import timedelta
            expected_at = (datetime.now(_TZ) + timedelta(minutes=mins)).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

    ko_no = _db.next_customer_order_no(db)
    db.execute(
        "INSERT INTO customer_orders (customer_order_no, customer, expected_delivery_at)"
        " VALUES (?, ?, ?)",
        (ko_no, customer, expected_at),
    )
    db.commit()

    created_ids = []
    for i, fig in enumerate(figure_ids):
        try:
            qty = max(1, int(quantities[i])) if i < len(quantities) else 1
        except (ValueError, IndexError):
            qty = 1
        oid = _create_order_line(db, ko_no, customer, fig, quantity=qty)
        created_ids.append(oid)

    exp_label = f" — forventet levering om {delivery_mins} min" if delivery_mins else ""
    flash(
        f"Kundeordre {ko_no} oprettet med {len(created_ids)} produkt(er)"
        f" ({', '.join(created_ids)}){exp_label}.",
        "info",
    )
    return redirect(url_for("salg"))


@app.post("/salg/tilfoej_produkt/<ko_no>")
def salg_tilfoej_produkt(ko_no):
    """Add an extra line item to an existing KO."""
    db = get_db()
    figure_id = request.form.get("figure_id", "").strip()
    ko = db.execute(
        "SELECT * FROM customer_orders WHERE customer_order_no=?", (ko_no,)
    ).fetchone()
    if not ko or figure_id not in REFERENCE_SEQUENCES:
        flash("Ugyldig kundeordre eller figur.", "error")
        return redirect(url_for("salg"))

    c_row = _db.get_customer_by_name(db, ko["customer"])
    if c_row and not _db.is_figure_allowed(db, c_row["customer_no"], figure_id):
        flash(f"Kunden har ikke rettighed til {figure_id}.", "error")
        return redirect(url_for("salg"))

    try:
        qty = max(1, int(request.form.get("quantity", 1)))
    except ValueError:
        qty = 1

    created_ids = [_create_order_line(db, ko_no, ko["customer"], figure_id, quantity=qty)]
    flash(f"Produkt tilføjet til {ko_no} ({created_ids[0]}, antal: {qty}).", "info")
    return redirect(url_for("salg"))


@app.post("/salg/godkend_kunde/<order_id>")
def salg_godkend_kunde(order_id):
    db = get_db()
    ko_no = _ko_no_of(db, order_id)
    if ko_no:
        _update_ko_status(db, ko_no, "indkoeb_afventer")
        flash(f"{ko_no} godkendt — sendt direkte til indkøb.", "info")
    else:
        _update_status(db, order_id, "indkoeb_afventer")
        flash(f"{order_id} godkendt — sendt direkte til indkøb.", "info")
    return redirect(url_for("salg"))


@app.post("/salg/afvis_kunde/<order_id>")
def salg_afvis_kunde(order_id):
    db = get_db()
    ko_no = _ko_no_of(db, order_id)
    if ko_no:
        _update_ko_status(db, ko_no, "afvist")
        flash(f"{ko_no} afvist.", "info")
    else:
        _update_status(db, order_id, "afvist")
        flash(f"{order_id} afvist.", "info")
    return redirect(url_for("salg"))


@app.post("/salg/registrer_afhentning/<ko_no>")
def salg_registrer_afhentning(ko_no):
    db = get_db()
    db.execute(
        "UPDATE customer_orders SET salg_pickup_requested=1 WHERE customer_order_no=?", (ko_no,)
    )
    db.commit()
    ko = db.execute("SELECT * FROM customer_orders WHERE customer_order_no=?", (ko_no,)).fetchone()
    if ko and ko["salg_delivery_confirmed"]:
        _update_ko_status(db, ko_no, "afhentet")
        flash(f"{ko_no} afhentet — begge parter har bekræftet.", "info")
    else:
        flash(f"{ko_no}: Afhentet markeret — venter på logistiks bekræftelse.", "info")
    return redirect(url_for("salg"))


@app.post("/salg/opret_faktura/<ko_no>")
def salg_opret_faktura(ko_no):
    db = get_db()
    amount = request.form.get("amount", "0").strip()
    try:
        amount = float(amount)
    except ValueError:
        flash("Ugyldigt beløb.", "error")
        return redirect(url_for("salg"))

    ko = db.execute("SELECT * FROM customer_orders WHERE customer_order_no=?", (ko_no,)).fetchone()
    if not ko:
        flash("Kundeordre ikke fundet.", "error")
        return redirect(url_for("salg"))

    # One invoice for the whole KO
    first = db.execute(
        "SELECT order_id FROM orders WHERE customer_order_no=? LIMIT 1", (ko_no,)
    ).fetchone()
    db.execute(
        "INSERT INTO invoices (order_id, customer, amount, invoice_date)"
        " VALUES (?, ?, ?, ?)",
        (first["order_id"] if first else ko_no, ko["customer"], amount, date.today().isoformat()),
    )
    db.commit()
    _update_ko_status(db, ko_no, "faktureret")
    flash(f"Faktura oprettet for {ko_no}: {amount:,.2f} kr.", "info")
    return redirect(url_for("salg"))


# ===========================================================================
# ØKONOMI
# ===========================================================================

@app.get("/oekonomi")
def oekonomi():
    db = get_db()
    pending_kapital = db.execute(
        "SELECT * FROM orders WHERE status='pending_kapital' ORDER BY created_at"
    ).fetchall()
    payments = db.execute(
        "SELECT * FROM payments ORDER BY created_at DESC"
    ).fetchall()
    invoices = db.execute(
        "SELECT i.*, o.figure_id FROM invoices i"
        " JOIN orders o ON i.order_id=o.order_id"
        " ORDER BY i.created_at DESC"
    ).fetchall()
    today_sum = db.execute(
        "SELECT COALESCE(SUM(amount),0) FROM invoices WHERE invoice_date=?",
        (date.today().isoformat(),),
    ).fetchone()[0]
    total_sum = db.execute(
        "SELECT COALESCE(SUM(amount),0) FROM invoices"
    ).fetchone()[0]
    credit_used    = _db.active_orders_cost(db)
    credit_avail   = _db.available_credit(db)
    profit         = _db.profit_liquidity(db)
    total_avail    = _db.total_available(db)
    total_costs_paid = db.execute(
        "SELECT COALESCE(SUM(amount),0) FROM payments WHERE status='betalt'"
    ).fetchone()[0] or 0.0
    margin_pct = round((profit / total_sum * 100), 1) if total_sum else 0.0
    return render_template("oekonomi.html",
                           pending_kapital=pending_kapital,
                           payments=payments,
                           invoices=invoices,
                           today_sum=today_sum,
                           total_sum=total_sum,
                           credit_used=credit_used,
                           credit_avail=credit_avail,
                           credit_limit=_db.CREDIT_LIMIT,
                           profit=profit,
                           total_avail=total_avail,
                           total_costs_paid=total_costs_paid,
                           margin_pct=margin_pct)


@app.post("/oekonomi/godkend_kapital/<order_id>")
def oekonomi_godkend_kapital(order_id):
    db = get_db()
    row = db.execute(
        "SELECT customer FROM orders WHERE order_id=?", (order_id,)
    ).fetchone()
    if row:
        customer = row["customer"]
        if customer in _db.AUTO_APPROVE_CUSTOMERS:
            _update_status(db, order_id, "indkoeb_afventer")
        else:
            _update_status(db, order_id, "pending_kunde")
    flash(f"{order_id} kapital godkendt.", "info")
    return redirect(url_for("oekonomi"))


@app.post("/oekonomi/afvis_kapital/<order_id>")
def oekonomi_afvis_kapital(order_id):
    _update_status(get_db(), order_id, "afvist")
    flash(f"{order_id} afvist af økonomi.", "info")
    return redirect(url_for("oekonomi"))


@app.post("/oekonomi/betal/<int:payment_id>")
def oekonomi_betal(payment_id):
    db = get_db()
    db.execute(
        "UPDATE payments SET status='betalt' WHERE id=?", (payment_id,)
    )
    db.commit()
    flash("Betaling markeret som betalt.", "info")
    return redirect(url_for("oekonomi"))


# ===========================================================================
# LOGISTIK
# ===========================================================================

def _ko_list(db, status_filter: str, extra_where: str = "") -> list:
    """Return distinct KOs that have at least one order in the given status."""
    return db.execute(
        f"SELECT co.* FROM customer_orders co"
        f" WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_order_no=co.customer_order_no"
        f"               AND o.status=?{' AND ' + extra_where if extra_where else ''})"
        f" ORDER BY co.created_at",
        (status_filter,),
    ).fetchall()


def _ko_products(db, ko_no: str, status: str | None = None) -> list:
    """Return all orders (line items) in a KO, optionally filtered by status."""
    q = "SELECT * FROM orders WHERE customer_order_no=?"
    params = [ko_no]
    if status:
        q += " AND status=?"
        params.append(status)
    return db.execute(q, params).fetchall()


@app.get("/logistik")
def logistik():
    db = get_db()

    # KOs with klodser_hentet orders
    klodser_kos = _ko_list(db, "klodser_hentet")
    ko_orders = {ko["customer_order_no"]: _ko_products(db, ko["customer_order_no"], "klodser_hentet")
                 for ko in klodser_kos}
    components = {}
    for orders in ko_orders.values():
        for o in orders:
            components[o["order_id"]] = db.execute(
                "SELECT component, quantity, unit_price FROM component_prices WHERE figure_id=?",
                (o["figure_id"],),
            ).fetchall()

    # KOs awaiting prod delivery confirmation
    pending_prod_kos = db.execute(
        "SELECT co.* FROM customer_orders co"
        " WHERE co.prod_pickup_requested=1 AND co.prod_delivery_confirmed=0"
        "   AND EXISTS (SELECT 1 FROM orders o WHERE o.customer_order_no=co.customer_order_no"
        "               AND o.status='klar_til_produktion')"
        " ORDER BY co.created_at"
    ).fetchall()

    # KOs on lager
    paa_lager_kos = _ko_list(db, "paa_lager")
    lager_orders  = {ko["customer_order_no"]: _ko_products(db, ko["customer_order_no"], "paa_lager")
                     for ko in paa_lager_kos}

    # KOs awaiting salg delivery confirmation
    pending_salg_kos = db.execute(
        "SELECT co.* FROM customer_orders co"
        " WHERE co.salg_pickup_requested=1 AND co.salg_delivery_confirmed=0"
        "   AND EXISTS (SELECT 1 FROM orders o WHERE o.customer_order_no=co.customer_order_no"
        "               AND o.status='klar_til_afhentning')"
        " ORDER BY co.created_at"
    ).fetchall()

    return render_template("logistik.html",
                           klodser_kos=klodser_kos,
                           ko_orders=ko_orders,
                           components=components,
                           pending_prod_kos=pending_prod_kos,
                           paa_lager_kos=paa_lager_kos,
                           lager_orders=lager_orders,
                           pending_salg_kos=pending_salg_kos)


@app.post("/logistik/send_payment_request/<ko_no>")
def logistik_send_payment_request(ko_no):
    db = get_db()
    supplier    = request.form.get("supplier", "").strip()
    amount      = request.form.get("amount", "0").strip()
    invoice_ref = request.form.get("invoice_ref", "").strip()

    try:
        amount = float(amount)
    except ValueError:
        flash("Ugyldigt beløb.", "error")
        return redirect(url_for("logistik"))

    # One payment record for the whole KO
    first = db.execute(
        "SELECT order_id FROM orders WHERE customer_order_no=? LIMIT 1", (ko_no,)
    ).fetchone()
    db.execute(
        "INSERT INTO payments (order_id, supplier, amount, invoice_ref)"
        " VALUES (?, ?, ?, ?)",
        (first["order_id"] if first else ko_no, supplier, amount, invoice_ref),
    )
    db.execute(
        "UPDATE customer_orders SET payment_request_sent=1 WHERE customer_order_no=?", (ko_no,)
    )
    db.commit()

    ko = db.execute("SELECT * FROM customer_orders WHERE customer_order_no=?", (ko_no,)).fetchone()
    if ko and ko["production_notified"]:
        _update_ko_status(db, ko_no, "klar_til_produktion")
        flash(f"{ko_no} klar til produktion — begge opgaver fuldført.", "info")
    else:
        flash(f"Betalingsanmodning sendt for {ko_no} — afventer produktion-notifikation.", "info")
    return redirect(url_for("logistik"))


@app.post("/logistik/notificer_prod/<ko_no>")
def logistik_notificer_prod(ko_no):
    db = get_db()
    db.execute(
        "UPDATE customer_orders SET production_notified=1 WHERE customer_order_no=?", (ko_no,)
    )
    db.commit()

    ko = db.execute("SELECT * FROM customer_orders WHERE customer_order_no=?", (ko_no,)).fetchone()
    if ko and ko["payment_request_sent"]:
        _update_ko_status(db, ko_no, "klar_til_produktion")
        flash(f"{ko_no} klar til produktion — begge opgaver fuldført.", "info")
    else:
        flash(f"{ko_no}: Produktion notificeret — afventer betalingsanmodning.", "info")
    return redirect(url_for("logistik"))


@app.post("/logistik/bekraeft_prod/<ko_no>")
def logistik_bekraeft_prod(ko_no):
    db = get_db()
    db.execute(
        "UPDATE customer_orders SET prod_delivery_confirmed=1 WHERE customer_order_no=?", (ko_no,)
    )
    db.commit()
    ko = db.execute("SELECT * FROM customer_orders WHERE customer_order_no=?", (ko_no,)).fetchone()
    if ko and ko["prod_pickup_requested"]:
        _update_ko_status(db, ko_no, "i_produktion")
        flash(f"{ko_no} er nu i produktion — begge parter har bekræftet.", "info")
    else:
        flash(f"{ko_no}: Aflevering bekræftet — venter på produktion.", "info")
    return redirect(url_for("logistik"))


@app.post("/logistik/bekraeft_salg/<ko_no>")
def logistik_bekraeft_salg(ko_no):
    db = get_db()
    db.execute(
        "UPDATE customer_orders SET salg_delivery_confirmed=1 WHERE customer_order_no=?", (ko_no,)
    )
    db.commit()
    ko = db.execute("SELECT * FROM customer_orders WHERE customer_order_no=?", (ko_no,)).fetchone()
    if ko and ko["salg_pickup_requested"]:
        _update_ko_status(db, ko_no, "afhentet")
        flash(f"{ko_no} afhentet — begge parter har bekræftet.", "info")
    else:
        flash(f"{ko_no}: Bekræftelse registreret — venter på salg.", "info")
    return redirect(url_for("logistik"))


@app.post("/logistik/notificer_salg/<ko_no>")
def logistik_notificer_salg(ko_no):
    db = get_db()
    holdeplads = request.form.get("holdeplads", "").strip() or None
    if holdeplads:
        db.execute(
            "UPDATE customer_orders SET holdeplads=? WHERE customer_order_no=?",
            (holdeplads, ko_no),
        )
        db.execute(
            "UPDATE orders SET holdeplads=?, updated_at=? WHERE customer_order_no=?",
            (holdeplads, _now(), ko_no),
        )
        db.commit()
    _update_ko_status(db, ko_no, "klar_til_afhentning")
    flash(f"{ko_no} notificeret til salg — klar til afhentning.", "info")
    return redirect(url_for("logistik"))


# ===========================================================================
# INDKØB
# ===========================================================================

@app.get("/indkoeb")
def indkoeb():
    db = get_db()
    # One row per KO that has any order in indkoeb_afventer
    kos = db.execute(
        "SELECT co.* FROM customer_orders co"
        " WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_order_no=co.customer_order_no"
        "               AND o.status='indkoeb_afventer')"
        " ORDER BY co.created_at"
    ).fetchall()
    # All orders per KO + per-order components (quantity × order.quantity)
    ko_orders = {}
    order_components = {}  # {order_id: [{component, quantity, unit_price}]}
    for ko in kos:
        orders = db.execute(
            "SELECT * FROM orders WHERE customer_order_no=? AND status='indkoeb_afventer'",
            (ko["customer_order_no"],),
        ).fetchall()
        ko_orders[ko["customer_order_no"]] = orders
        for o in orders:
            qty_mult = o["quantity"] if o["quantity"] else 1
            rows = db.execute(
                "SELECT component, quantity, unit_price FROM component_prices WHERE figure_id=?",
                (o["figure_id"],),
            ).fetchall()
            order_components[o["order_id"]] = [
                {"component": c["component"],
                 "quantity":  c["quantity"] * qty_mult,
                 "unit_price": c["unit_price"]}
                for c in rows
            ]

    # Reverse map: varenummer → CSS colour name
    comp_colors = {varenr: color for color, (varenr, _) in _db.COMPONENTS.items()}

    return render_template("indkoeb.html", kos=kos, ko_orders=ko_orders,
                           order_components=order_components, comp_colors=comp_colors)


@app.post("/indkoeb/bekraeft/<ko_no>")
def indkoeb_bekraeft(ko_no):
    db = get_db()
    orders = db.execute(
        "SELECT * FROM orders WHERE customer_order_no=? AND status='indkoeb_afventer'", (ko_no,)
    ).fetchall()
    for o in orders:
        qty    = o["quantity"] if o["quantity"] else 1
        actual = _db.figure_cost(db, o["figure_id"]) * qty
        db.execute("UPDATE orders SET actual_cost=? WHERE order_id=?", (actual, o["order_id"]))
    db.commit()
    _update_ko_status(db, ko_no, "klodser_hentet")
    flash(f"Kundeordre {ko_no} — klodser hentet, sendt til logistik.", "info")
    return redirect(url_for("indkoeb"))


# ===========================================================================
# PRODUKTION
# ===========================================================================

@app.get("/produktion")
def produktion():
    db = get_db()
    klar = db.execute(
        "SELECT o.*, co.expected_delivery_at, co.customer_order_no as ko_no"
        " FROM orders o"
        " LEFT JOIN customer_orders co ON co.customer_order_no = o.customer_order_no"
        " WHERE o.status='klar_til_produktion' ORDER BY o.updated_at"
    ).fetchall()
    i_prod = db.execute(
        "SELECT o.*, co.expected_delivery_at, co.customer_order_no as ko_no"
        " FROM orders o"
        " LEFT JOIN customer_orders co ON co.customer_order_no = o.customer_order_no"
        " WHERE o.status='i_produktion' ORDER BY o.updated_at"
    ).fetchall()

    # Group orders by KO so we can show all instructions per KO together
    def group_by_ko(orders):
        groups = {}
        for o in orders:
            key = o["customer_order_no"] or o["order_id"]
            groups.setdefault(key, []).append(o)
        return groups

    return render_template("produktion.html",
                           klar=klar, i_prod=i_prod,
                           klar_by_ko=group_by_ko(klar),
                           i_prod_by_ko=group_by_ko(i_prod),
                           sequences=REFERENCE_SEQUENCES,
                           now=_now())


@app.post("/produktion/hent/<ko_no>")
def produktion_hent(ko_no):
    db = get_db()
    db.execute(
        "UPDATE customer_orders SET prod_pickup_requested=1 WHERE customer_order_no=?", (ko_no,)
    )
    db.commit()
    ko = db.execute("SELECT * FROM customer_orders WHERE customer_order_no=?", (ko_no,)).fetchone()
    if ko and ko["prod_delivery_confirmed"]:
        _update_ko_status(db, ko_no, "i_produktion")
        flash(f"{ko_no} er nu i produktion.", "info")
    else:
        flash(f"{ko_no}: Hentning registreret — venter på logistiks bekræftelse.", "info")
    return redirect(url_for("produktion"))


@app.post("/produktion/faerdig/<ko_no>")
def produktion_faerdig(ko_no):
    db = get_db()
    now = _now()
    db.execute(
        "UPDATE customer_orders SET produced_at=? WHERE customer_order_no=?", (now, ko_no)
    )
    db.commit()
    _update_ko_status(db, ko_no, "klar_til_qc", {"produced_at": now})
    flash(f"{ko_no} klar til QC-inspektion.", "info")
    return redirect(url_for("produktion"))


# ===========================================================================
# QC — Sikkerhed & Kvalitet
# ===========================================================================

@app.get("/qc")
def qc():
    db = get_db()
    orders_klar = db.execute(
        "SELECT * FROM orders WHERE status='klar_til_qc' ORDER BY produced_at"
    ).fetchall()
    last_result = session.pop("qc_result", None)

    # Build reference info for every figure: {figure_id: url_or_None}
    ref_urls = {}
    for fig_id in REFERENCE_SEQUENCES:
        p = _ref_path(fig_id)
        if p:
            rel = os.path.relpath(p, os.path.join(os.path.dirname(__file__), "static"))
            ref_urls[fig_id] = url_for("static", filename=rel.replace("\\", "/"))
        else:
            ref_urls[fig_id] = None

    return render_template("qc.html",
                           orders_klar=orders_klar,
                           last_result=last_result,
                           ref_urls=ref_urls)


@app.get("/qc/video_feed")
def qc_video_feed():
    return Response(_mjpeg_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/qc/inspect/<order_id>")
def qc_inspect(order_id):
    db = get_db()
    row = db.execute(
        "SELECT figure_id FROM orders WHERE order_id=?", (order_id,)
    ).fetchone()
    if not row:
        flash("Ordre ikke fundet.", "error")
        return redirect(url_for("qc"))

    figure_id = row["figure_id"]
    expected  = REFERENCE_SEQUENCES.get(figure_id, [])

    frame = camera.read()
    if frame is None:
        flash("Kamera ikke tilgængeligt.", "error")
        return redirect(url_for("qc"))

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, frame)
        palette_path = os.path.join(REFS_DIR, figure_id + ".json")
        palette = load_palette(palette_path)
        detected = detect_layers(tmp_path, expected_colors=expected, palette=palette)
    finally:
        os.unlink(tmp_path)

    outcome = _compare_sequences(detected, expected)
    outcome["figure_id"] = figure_id
    outcome["order_id"]  = order_id

    # Store result on the order
    db.execute(
        "UPDATE orders SET qc_result=?, updated_at=? WHERE order_id=?",
        (outcome["result"], _now(), order_id),
    )
    db.commit()
    session["qc_result"] = outcome
    return redirect(url_for("qc"))


@app.get("/qc/inspect_manual/<order_id>")
def qc_inspect_manual(order_id):
    db = get_db()
    row = db.execute(
        "SELECT figure_id FROM orders WHERE order_id=?", (order_id,)
    ).fetchone()
    if not row:
        flash("Ordre ikke fundet.", "error")
        return redirect(url_for("qc"))
    figure_id = row["figure_id"]
    expected  = REFERENCE_SEQUENCES.get(figure_id, [])
    all_colors = ["red", "blue", "yellow", "green", "white", "orange", "purple"]
    return render_template("qc_manual.html",
                           order_id=order_id,
                           figure_id=figure_id,
                           expected=expected,
                           all_colors=all_colors)


@app.post("/qc/submit_manual/<order_id>")
def qc_submit_manual(order_id):
    db = get_db()
    row = db.execute(
        "SELECT figure_id FROM orders WHERE order_id=?", (order_id,)
    ).fetchone()
    if not row:
        flash("Ordre ikke fundet.", "error")
        return redirect(url_for("qc"))
    figure_id = row["figure_id"]
    expected  = REFERENCE_SEQUENCES.get(figure_id, [])

    detected = [request.form.get(f"layer_{i}", "") for i in range(len(expected))]

    outcome = _compare_sequences(detected, expected)
    outcome["figure_id"] = figure_id
    outcome["order_id"]  = order_id

    db.execute(
        "UPDATE orders SET qc_result=?, updated_at=? WHERE order_id=?",
        (outcome["result"], _now(), order_id),
    )
    db.commit()
    session["qc_result"] = outcome
    return redirect(url_for("qc"))


def _all_qc_passed(db, ko_no: str) -> bool:
    """True if every order in the KO has qc_result='pass'."""
    total = db.execute(
        "SELECT COUNT(*) FROM orders WHERE customer_order_no=?", (ko_no,)
    ).fetchone()[0]
    passed = db.execute(
        "SELECT COUNT(*) FROM orders WHERE customer_order_no=? AND qc_result='pass'", (ko_no,)
    ).fetchone()[0]
    return total > 0 and total == passed


@app.post("/qc/godkend/<order_id>")
def qc_godkend(order_id):
    db = get_db()
    # Mark this product as passed
    db.execute("UPDATE orders SET qc_result='pass', updated_at=? WHERE order_id=?",
               (_now(), order_id))
    db.commit()
    ko_no = _ko_no_of(db, order_id)
    if ko_no and _all_qc_passed(db, ko_no):
        holdeplads = _db.assign_holdeplads(db)
        _update_ko_status(db, ko_no, "paa_lager", {"holdeplads": holdeplads})
        db.execute("UPDATE customer_orders SET holdeplads=?, qc_result='pass' WHERE customer_order_no=?",
                   (holdeplads, ko_no))
        db.commit()
        flash(f"{ko_no} — alle produkter godkendt, placeret på lager {holdeplads}.", "info")
    else:
        remaining = db.execute(
            "SELECT COUNT(*) FROM orders WHERE customer_order_no=? AND (qc_result IS NULL OR qc_result!='pass')",
            (ko_no,)
        ).fetchone()[0] if ko_no else 0
        flash(f"{order_id} godkendt — {remaining} produkt(er) afventer stadig QC.", "info")
    return redirect(url_for("qc"))


@app.post("/qc/godkend_manuel/<order_id>")
def qc_godkend_manuel(order_id):
    """Manual override — approve despite failed or missing CV inspection."""
    db = get_db()
    db.execute("UPDATE orders SET qc_result='pass', updated_at=? WHERE order_id=?",
               (_now(), order_id))
    db.commit()
    ko_no = _ko_no_of(db, order_id)
    if ko_no and _all_qc_passed(db, ko_no):
        holdeplads = _db.assign_holdeplads(db)
        _update_ko_status(db, ko_no, "paa_lager", {"holdeplads": holdeplads})
        db.execute("UPDATE customer_orders SET holdeplads=?, qc_result='pass' WHERE customer_order_no=?",
                   (holdeplads, ko_no))
        db.commit()
        flash(f"{ko_no} — alle produkter manuelt godkendt, placeret på lager {holdeplads}.", "info")
    else:
        flash(f"{order_id} manuelt godkendt — øvrige produkter afventer QC.", "info")
    return redirect(url_for("qc"))


@app.post("/qc/afvis/<order_id>")
def qc_afvis(order_id):
    db = get_db()
    ko_no = _ko_no_of(db, order_id)
    if ko_no:
        _update_ko_status(db, ko_no, "i_produktion")
        flash(f"{ko_no} fejlede QC — hele kundeordren returneret til produktion.", "info")
    else:
        _update_status(db, order_id, "i_produktion")
        flash(f"{order_id} fejlede QC — returneret til produktion.", "info")
    return redirect(url_for("qc"))


@app.post("/qc/upload_ref/<figure_id>")
def qc_upload_ref(figure_id):
    if figure_id not in REFERENCE_SEQUENCES:
        flash("Ukendt figur-ID.", "error")
        return redirect(url_for("qc"))

    file = request.files.get("ref_image")
    if not file or file.filename == "":
        flash("Ingen fil valgt.", "error")
        return redirect(url_for("qc"))

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        flash("Kun JPG, PNG eller BMP er tilladt.", "error")
        return redirect(url_for("qc"))

    # Remove any existing reference for this figure before saving the new one
    for old_ext in ALLOWED_EXTENSIONS:
        old = os.path.join(REFS_DIR, figure_id + old_ext)
        if os.path.exists(old):
            os.unlink(old)

    save_path = os.path.join(REFS_DIR, figure_id + ext)
    file.save(save_path)

    # Build and cache the colour palette from the new reference image
    palette = build_palette(save_path, REFERENCE_SEQUENCES[figure_id])
    if palette:
        save_palette(palette, os.path.join(REFS_DIR, figure_id + ".json"))
        flash(f"Referencebillede gemt og palette kalibreret for {figure_id}.", "info")
    else:
        flash(f"Referencebillede gemt for {figure_id} (palette-kalibrering fejlede — tjek billedet).", "warning")
    return redirect(url_for("qc"))


# ===========================================================================
# KUNDETILFREDSHED
# ===========================================================================

@app.get("/kundetilfredshed")
def kundetilfredshed():
    db = get_db()
    log = db.execute(
        "SELECT order_id, customer, figure_id, customer_rating, updated_at"
        " FROM orders WHERE customer_rating IS NOT NULL"
        " ORDER BY updated_at DESC"
    ).fetchall()
    avg = db.execute(
        "SELECT ROUND(AVG(customer_rating), 2) FROM orders WHERE customer_rating IS NOT NULL"
    ).fetchone()[0]
    afventer_rating = db.execute(
        "SELECT * FROM orders WHERE status='faktureret' AND customer_rating IS NULL"
        " ORDER BY updated_at DESC"
    ).fetchall()
    return render_template("kundetilfredshed.html",
                           log=log,
                           avg_rating=avg,
                           afventer_rating=afventer_rating)


@app.post("/kundetilfredshed/registrer/<order_id>")
def kundetilfredshed_registrer(order_id):
    db = get_db()
    rating = request.form.get("rating", "").strip()
    if not rating.isdigit() or int(rating) not in range(1, 6):
        flash("Rating skal være mellem 1 og 5.", "error")
        return redirect(url_for("kundetilfredshed"))
    db.execute(
        "UPDATE orders SET customer_rating=?, updated_at=? WHERE order_id=?",
        (int(rating), _now(), order_id),
    )
    db.commit()
    flash(f"Rating {rating}/5 registreret for {order_id}.", "info")
    return redirect(url_for("kundetilfredshed"))


# ===========================================================================
# KUNDER
# ===========================================================================

@app.get("/kunder")
def kunder():
    db = get_db()
    customers = _db.get_customers(db)
    all_figures = list(REFERENCE_SEQUENCES.keys())
    perms = _db.get_permissions_map(db)
    return render_template("kunder.html",
                           customers=customers,
                           all_figures=all_figures,
                           perms=perms)


@app.post("/kunder/opret")
def kunder_opret():
    db = get_db()
    customer_no  = request.form.get("customer_no", "").strip()
    name         = request.form.get("name", "").strip()
    discount_pct = request.form.get("discount_pct", "0").strip()
    auto_approve = 1 if request.form.get("auto_approve") else 0

    if not customer_no or not name:
        flash("Kundenr og navn er påkrævet.", "error")
        return redirect(url_for("kunder"))
    try:
        discount = float(discount_pct) / 100.0
    except ValueError:
        discount = 0.0

    db.execute(
        "INSERT OR IGNORE INTO customers (customer_no, name, discount_pct, auto_approve)"
        " VALUES (?, ?, ?, ?)",
        (customer_no, name, discount, auto_approve),
    )
    for fig in REFERENCE_SEQUENCES.keys():
        db.execute(
            "INSERT OR IGNORE INTO customer_permissions (customer_no, figure_id, approved)"
            " VALUES (?, ?, 1)",
            (customer_no, fig),
        )
    db.commit()
    flash(f"Kunde '{name}' ({customer_no}) oprettet.", "info")
    return redirect(url_for("kunder"))


@app.post("/kunder/<customer_no>/rettigheder")
def kunder_rettigheder(customer_no):
    db = get_db()
    try:
        discount = float(request.form.get("discount_pct", 0)) / 100.0
    except ValueError:
        discount = 0.0
    db.execute(
        "UPDATE customers SET discount_pct=? WHERE customer_no=?",
        (discount, customer_no),
    )
    for fig in REFERENCE_SEQUENCES.keys():
        approved = 1 if request.form.get(f"fig_{fig}") else 0
        db.execute(
            "INSERT OR REPLACE INTO customer_permissions (customer_no, figure_id, approved)"
            " VALUES (?, ?, ?)",
            (customer_no, fig, approved),
        )
    db.commit()
    flash("Rettigheder og rabat opdateret.", "info")
    return redirect(url_for("kunder"))


# ===========================================================================
# HISTORIK
# ===========================================================================

@app.get("/historik")
def historik():
    db = get_db()
    orders = db.execute(
        "SELECT * FROM orders ORDER BY created_at DESC"
    ).fetchall()
    return render_template("historik.html", orders=orders)


@app.get("/kundeordre/<ko_no>")
def kundeordre_detail(ko_no):
    db = get_db()
    ko = db.execute(
        "SELECT * FROM customer_orders WHERE customer_order_no=?", (ko_no,)
    ).fetchone()
    if not ko:
        flash("Kundeordre ikke fundet.", "error")
        return redirect(url_for("historik"))
    items = db.execute(
        "SELECT * FROM orders WHERE customer_order_no=? ORDER BY created_at",
        (ko_no,),
    ).fetchall()

    # Penalty calculation
    penalty_per_min = sum(o["selling_price"] or 0 for o in items) * 0.01
    mins_late = 0
    if ko["expected_delivery_at"]:
        try:
            deadline = datetime.strptime(ko["expected_delivery_at"], "%Y-%m-%d %H:%M:%S")
            now_dt   = datetime.now(_TZ).replace(tzinfo=None)
            mins_late = max(0, int((now_dt - deadline).total_seconds() / 60))
        except ValueError:
            pass

    all_done = all(o["status"] in ("faktureret", "afvist") for o in items)

    return render_template("kundeordre_detail.html",
                           ko=ko,
                           items=items,
                           penalty_per_min=penalty_per_min,
                           mins_late=mins_late,
                           all_done=all_done,
                           status_labels=_db.STATUS_LABELS,
                           now=_now())


@app.get("/ordre/<order_id>")
def ordre_detail(order_id):
    db = get_db()
    order = db.execute(
        "SELECT * FROM orders WHERE order_id=?", (order_id,)
    ).fetchone()
    if not order:
        flash("Ordre ikke fundet.", "error")
        return redirect(url_for("historik"))

    raw_events = db.execute(
        "SELECT * FROM order_events WHERE order_id=? ORDER BY timestamp ASC",
        (order_id,),
    ).fetchall()

    # Calculate time spent in each step
    events = []
    for i, ev in enumerate(raw_events):
        try:
            t_start = datetime.strptime(ev["timestamp"], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            t_start = None

        duration_min = None
        if i + 1 < len(raw_events) and t_start:
            try:
                t_next = datetime.strptime(raw_events[i + 1]["timestamp"], "%Y-%m-%d %H:%M:%S")
                duration_min = int((t_next - t_start).total_seconds() / 60)
            except ValueError:
                pass
        elif i == len(raw_events) - 1 and t_start:
            # Last event: time until now
            duration_min = int((datetime.now(_TZ).replace(tzinfo=None) - t_start).total_seconds() / 60)

        events.append({
            "status":      ev["status"],
            "department":  ev["department"],
            "note":        ev["note"],
            "timestamp":   ev["timestamp"],
            "duration_min": duration_min,
            "is_last":     i == len(raw_events) - 1,
        })

    # Total cycle time
    total_min = None
    if raw_events:
        try:
            t0 = datetime.strptime(raw_events[0]["timestamp"], "%Y-%m-%d %H:%M:%S")
            t1 = datetime.now(_TZ).replace(tzinfo=None)
            total_min = int((t1 - t0).total_seconds() / 60)
        except ValueError:
            pass

    return render_template("ordre_detail.html",
                           order=order,
                           events=events,
                           total_min=total_min,
                           status_labels=_db.STATUS_LABELS)


# ===========================================================================
# Dev entry point
# ===========================================================================

if __name__ == "__main__":
    try:
        app.run(debug=True, threaded=True)
    finally:
        camera.release()
