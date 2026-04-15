import os
import tempfile
import threading
import time
from datetime import date, datetime, timezone

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
        " ('kunde_godkendt','klodser_hentet','paa_lager')"
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
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _update_status(db, order_id: str, status: str, extra: dict | None = None):
    fields = {"status": status, "updated_at": _now()}
    if extra:
        fields.update(extra)
    set_clause = ", ".join(f"{k}=?" for k in fields)
    db.execute(
        f"UPDATE orders SET {set_clause} WHERE order_id=?",
        list(fields.values()) + [order_id],
    )
    db.commit()


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
        "SELECT * FROM orders ORDER BY updated_at DESC"
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
    return jsonify({
        "active_orders": active,
        "pending_approval": pending,
        "paa_lager": paa_lager,
        "produced_today": produced_today,
        "todays_revenue": todays_revenue,
    })


# ===========================================================================
# SALG
# ===========================================================================

@app.get("/salg")
def salg():
    db = get_db()
    pending_kunde = db.execute(
        "SELECT * FROM orders WHERE status='pending_kunde' ORDER BY created_at"
    ).fetchall()
    klar_afhentning = db.execute(
        "SELECT * FROM orders WHERE status='klar_til_afhentning' ORDER BY updated_at DESC"
    ).fetchall()
    afhentet = db.execute(
        "SELECT o.* FROM orders o"
        " LEFT JOIN invoices i ON o.order_id=i.order_id"
        " WHERE o.status='afhentet' AND i.id IS NULL"
        " ORDER BY o.updated_at DESC"
    ).fetchall()
    return render_template("salg.html",
                           pending_kunde=pending_kunde,
                           klar_afhentning=klar_afhentning,
                           afhentet=afhentet)


@app.post("/salg/ny_ordre")
def salg_ny_ordre():
    db = get_db()
    customer  = request.form.get("customer", "").strip()
    figure_id = request.form.get("figure_id", "").strip()

    if not customer or not figure_id:
        flash("Udfyld alle felter.", "error")
        return redirect(url_for("salg"))
    if figure_id not in REFERENCE_SEQUENCES:
        flash("Ukendt figur-ID.", "error")
        return redirect(url_for("salg"))

    order_id      = _db.next_order_id(db)
    cost          = _db.figure_cost(db, figure_id)
    selling_price = _db.figure_selling_price(figure_id, customer)

    db.execute(
        "INSERT INTO orders (order_id, customer, figure_id, status, estimated_cost, selling_price)"
        " VALUES (?, ?, ?, 'oekonomi_check', ?, ?)",
        (order_id, customer, figure_id, cost, selling_price),
    )
    db.commit()

    # Auto-run økonomi check
    available = _db.available_credit(db)
    if available >= cost:
        new_status = "kapital_ok"
    else:
        new_status = "pending_kapital"
    _update_status(db, order_id, new_status)

    # If kapital_ok, check for auto-approved customer
    if new_status == "kapital_ok":
        if customer in _db.AUTO_APPROVE_CUSTOMERS:
            _update_status(db, order_id, "kunde_godkendt")
            flash(f"Ordre {order_id} oprettet og auto-godkendt (prioritetskunde).", "info")
        else:
            _update_status(db, order_id, "pending_kunde")
            flash(f"Ordre {order_id} oprettet — afventer kundegodkendelse.", "info")
    else:
        flash(f"Ordre {order_id} oprettet — afventer kapital-godkendelse.", "info")

    return redirect(url_for("salg"))


@app.post("/salg/godkend_kunde/<order_id>")
def salg_godkend_kunde(order_id):
    _update_status(get_db(), order_id, "kunde_godkendt")
    flash(f"{order_id} godkendt — sendt til logistik.", "info")
    return redirect(url_for("salg"))


@app.post("/salg/afvis_kunde/<order_id>")
def salg_afvis_kunde(order_id):
    _update_status(get_db(), order_id, "afvist")
    flash(f"{order_id} afvist.", "info")
    return redirect(url_for("salg"))


@app.post("/salg/registrer_afhentning/<order_id>")
def salg_registrer_afhentning(order_id):
    db = get_db()
    db.execute(
        "UPDATE orders SET salg_pickup_requested=1, updated_at=? WHERE order_id=?",
        (_now(), order_id),
    )
    db.commit()
    # Check if logistik already confirmed
    row = db.execute(
        "SELECT salg_delivery_confirmed FROM orders WHERE order_id=?", (order_id,)
    ).fetchone()
    if row and row["salg_delivery_confirmed"]:
        _update_status(db, order_id, "afhentet")
        flash(f"{order_id} afhentet — begge parter har bekræftet.", "info")
    else:
        flash(f"{order_id}: Afhentet markeret — venter på logistiks bekræftelse.", "info")
    return redirect(url_for("salg"))


@app.post("/salg/opret_faktura/<order_id>")
def salg_opret_faktura(order_id):
    db = get_db()
    amount = request.form.get("amount", "0").strip()
    try:
        amount = float(amount)
    except ValueError:
        flash("Ugyldigt beløb.", "error")
        return redirect(url_for("salg"))

    row = db.execute(
        "SELECT customer FROM orders WHERE order_id=?", (order_id,)
    ).fetchone()
    if not row:
        flash("Ordre ikke fundet.", "error")
        return redirect(url_for("salg"))

    db.execute(
        "INSERT INTO invoices (order_id, customer, amount, invoice_date)"
        " VALUES (?, ?, ?, ?)",
        (order_id, row["customer"], amount, date.today().isoformat()),
    )
    db.commit()
    _update_status(db, order_id, "faktureret")
    flash(f"Faktura oprettet for {order_id}: {amount:,.2f} kr.", "info")
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
    credit_used  = _db.active_orders_cost(db)
    credit_avail = _db.available_credit(db)
    return render_template("oekonomi.html",
                           pending_kapital=pending_kapital,
                           payments=payments,
                           invoices=invoices,
                           today_sum=today_sum,
                           total_sum=total_sum,
                           credit_used=credit_used,
                           credit_avail=credit_avail,
                           credit_limit=_db.CREDIT_LIMIT)


@app.post("/oekonomi/godkend_kapital/<order_id>")
def oekonomi_godkend_kapital(order_id):
    db = get_db()
    row = db.execute(
        "SELECT customer FROM orders WHERE order_id=?", (order_id,)
    ).fetchone()
    if row:
        customer = row["customer"]
        if customer in _db.AUTO_APPROVE_CUSTOMERS:
            _update_status(db, order_id, "kunde_godkendt")
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

@app.get("/logistik")
def logistik():
    db = get_db()
    kunde_godkendt = db.execute(
        "SELECT * FROM orders WHERE status='kunde_godkendt' ORDER BY updated_at"
    ).fetchall()
    klodser_hentet = db.execute(
        "SELECT * FROM orders WHERE status='klodser_hentet' ORDER BY updated_at"
    ).fetchall()
    pending_prod_confirm = db.execute(
        "SELECT * FROM orders WHERE status='klar_til_produktion'"
        " AND prod_pickup_requested=1 AND prod_delivery_confirmed=0 ORDER BY updated_at"
    ).fetchall()
    paa_lager = db.execute(
        "SELECT * FROM orders WHERE status='paa_lager' ORDER BY updated_at DESC"
    ).fetchall()
    pending_salg_confirm = db.execute(
        "SELECT * FROM orders WHERE status='klar_til_afhentning'"
        " AND salg_pickup_requested=1 AND salg_delivery_confirmed=0 ORDER BY updated_at"
    ).fetchall()

    # Components per order for klodser_hentet display
    components = {}
    for o in klodser_hentet:
        rows = db.execute(
            "SELECT component, quantity, unit_price FROM component_prices WHERE figure_id=?",
            (o["figure_id"],),
        ).fetchall()
        components[o["order_id"]] = rows

    return render_template("logistik.html",
                           kunde_godkendt=kunde_godkendt,
                           klodser_hentet=klodser_hentet,
                           pending_prod_confirm=pending_prod_confirm,
                           paa_lager=paa_lager,
                           pending_salg_confirm=pending_salg_confirm,
                           components=components)


@app.post("/logistik/send_indkoeb/<order_id>")
def logistik_send_indkoeb(order_id):
    _update_status(get_db(), order_id, "indkoeb_afventer")
    flash(f"{order_id} sendt til indkøb.", "info")
    return redirect(url_for("logistik"))


@app.post("/logistik/send_payment_request/<order_id>")
def logistik_send_payment_request(order_id):
    db = get_db()
    supplier    = request.form.get("supplier", "").strip()
    amount      = request.form.get("amount", "0").strip()
    invoice_ref = request.form.get("invoice_ref", "").strip()

    try:
        amount = float(amount)
    except ValueError:
        flash("Ugyldigt beløb.", "error")
        return redirect(url_for("logistik"))

    db.execute(
        "INSERT INTO payments (order_id, supplier, amount, invoice_ref)"
        " VALUES (?, ?, ?, ?)",
        (order_id, supplier, amount, invoice_ref),
    )
    db.execute(
        "UPDATE orders SET payment_request_sent=1, updated_at=? WHERE order_id=?",
        (_now(), order_id),
    )
    db.commit()

    # Advance status if production was already notified
    row = db.execute(
        "SELECT production_notified FROM orders WHERE order_id=?", (order_id,)
    ).fetchone()
    if row and row["production_notified"]:
        _update_status(db, order_id, "klar_til_produktion")

    flash(f"Betalingsanmodning sendt for {order_id}.", "info")
    return redirect(url_for("logistik"))


@app.post("/logistik/notificer_prod/<order_id>")
def logistik_notificer_prod(order_id):
    db = get_db()
    db.execute(
        "UPDATE orders SET production_notified=1, updated_at=? WHERE order_id=?",
        (_now(), order_id),
    )
    db.commit()

    row = db.execute(
        "SELECT payment_request_sent FROM orders WHERE order_id=?", (order_id,)
    ).fetchone()
    if row and row["payment_request_sent"]:
        _update_status(db, order_id, "klar_til_produktion")
        flash(f"{order_id} klar til produktion — begge opgaver fuldført.", "info")
    else:
        flash(f"{order_id}: Produktion notificeret — afventer betalingsanmodning.", "info")
    return redirect(url_for("logistik"))


@app.post("/logistik/bekraeft_prod/<order_id>")
def logistik_bekraeft_prod(order_id):
    db = get_db()
    db.execute(
        "UPDATE orders SET prod_delivery_confirmed=1, updated_at=? WHERE order_id=?",
        (_now(), order_id),
    )
    db.commit()
    row = db.execute(
        "SELECT prod_pickup_requested FROM orders WHERE order_id=?", (order_id,)
    ).fetchone()
    if row and row["prod_pickup_requested"]:
        _update_status(db, order_id, "i_produktion")
        flash(f"{order_id} er nu i produktion — begge parter har bekræftet.", "info")
    else:
        flash(f"{order_id}: Aflevering bekræftet — venter på produktion.", "info")
    return redirect(url_for("logistik"))


@app.post("/logistik/bekraeft_salg/<order_id>")
def logistik_bekraeft_salg(order_id):
    db = get_db()
    db.execute(
        "UPDATE orders SET salg_delivery_confirmed=1, updated_at=? WHERE order_id=?",
        (_now(), order_id),
    )
    db.commit()
    row = db.execute(
        "SELECT salg_pickup_requested FROM orders WHERE order_id=?", (order_id,)
    ).fetchone()
    if row and row["salg_pickup_requested"]:
        _update_status(db, order_id, "afhentet")
        flash(f"{order_id} afhentet — begge parter har bekræftet.", "info")
    else:
        flash(f"{order_id}: Bekræftelse registreret — venter på salg.", "info")
    return redirect(url_for("logistik"))


@app.post("/logistik/notificer_salg/<order_id>")
def logistik_notificer_salg(order_id):
    db = get_db()
    # Allow editing holdeplads
    holdeplads = request.form.get("holdeplads", "").strip() or None
    if holdeplads:
        db.execute(
            "UPDATE orders SET holdeplads=?, updated_at=? WHERE order_id=?",
            (holdeplads, _now(), order_id),
        )
        db.commit()
    _update_status(db, order_id, "klar_til_afhentning")
    flash(f"{order_id} notificeret til salg — klar til afhentning.", "info")
    return redirect(url_for("logistik"))


# ===========================================================================
# INDKØB
# ===========================================================================

@app.get("/indkoeb")
def indkoeb():
    db = get_db()
    orders = db.execute(
        "SELECT * FROM orders WHERE status='indkoeb_afventer' ORDER BY updated_at"
    ).fetchall()
    components = {}
    for o in orders:
        rows = db.execute(
            "SELECT component, quantity, unit_price FROM component_prices WHERE figure_id=?",
            (o["figure_id"],),
        ).fetchall()
        components[o["order_id"]] = rows
    return render_template("indkoeb.html", orders=orders, components=components)


@app.post("/indkoeb/bekraeft/<order_id>")
def indkoeb_bekraeft(order_id):
    db = get_db()
    # Check if actual cost matches estimate
    actual = _db.figure_cost(db, db.execute(
        "SELECT figure_id FROM orders WHERE order_id=?", (order_id,)
    ).fetchone()["figure_id"])
    _update_status(db, order_id, "klodser_hentet", {"actual_cost": actual})
    flash(f"{order_id} klodser hentet — sendt til logistik.", "info")
    return redirect(url_for("indkoeb"))


# ===========================================================================
# PRODUKTION
# ===========================================================================

@app.get("/produktion")
def produktion():
    db = get_db()
    klar = db.execute(
        "SELECT * FROM orders WHERE status='klar_til_produktion' ORDER BY updated_at"
    ).fetchall()
    i_prod = db.execute(
        "SELECT * FROM orders WHERE status='i_produktion' ORDER BY updated_at"
    ).fetchall()
    return render_template("produktion.html", klar=klar, i_prod=i_prod,
                           sequences=REFERENCE_SEQUENCES)


@app.post("/produktion/hent/<order_id>")
def produktion_hent(order_id):
    db = get_db()
    db.execute(
        "UPDATE orders SET prod_pickup_requested=1, updated_at=? WHERE order_id=?",
        (_now(), order_id),
    )
    db.commit()
    row = db.execute(
        "SELECT prod_delivery_confirmed FROM orders WHERE order_id=?", (order_id,)
    ).fetchone()
    if row and row["prod_delivery_confirmed"]:
        _update_status(db, order_id, "i_produktion")
        flash(f"{order_id} er nu i produktion.", "info")
    else:
        flash(f"{order_id}: Hentning registreret — venter på logistiks bekræftelse.", "info")
    return redirect(url_for("produktion"))


@app.post("/produktion/faerdig/<order_id>")
def produktion_faerdig(order_id):
    _update_status(get_db(), order_id, "klar_til_qc",
                   {"produced_at": _now()})
    flash(f"{order_id} klar til QC-inspektion.", "info")
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


@app.post("/qc/godkend/<order_id>")
def qc_godkend(order_id):
    db = get_db()
    holdeplads = _db.assign_holdeplads(db)
    _update_status(db, order_id, "paa_lager", {"holdeplads": holdeplads})
    flash(f"{order_id} godkendt — placeret på lager {holdeplads}.", "info")
    return redirect(url_for("qc"))


@app.post("/qc/godkend_manuel/<order_id>")
def qc_godkend_manuel(order_id):
    """Manual override — approve despite failed or missing CV inspection."""
    db = get_db()
    holdeplads = _db.assign_holdeplads(db)
    _update_status(db, order_id, "paa_lager",
                   {"holdeplads": holdeplads, "qc_result": "pass"})
    flash(f"{order_id} manuelt godkendt — placeret på lager {holdeplads}.", "info")
    return redirect(url_for("qc"))


@app.post("/qc/afvis/<order_id>")
def qc_afvis(order_id):
    # Product is already with Produktion — send straight back to i_produktion,
    # no need to re-fetch from Logistik.
    _update_status(get_db(), order_id, "i_produktion")
    flash(f"{order_id} fejlede QC — returneret til produktion til rettelse.", "info")
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
# HISTORIK
# ===========================================================================

@app.get("/historik")
def historik():
    db = get_db()
    orders = db.execute(
        "SELECT * FROM orders ORDER BY created_at DESC"
    ).fetchall()
    return render_template("historik.html", orders=orders)


# ===========================================================================
# Dev entry point
# ===========================================================================

if __name__ == "__main__":
    try:
        app.run(debug=True, threaded=True)
    finally:
        camera.release()
