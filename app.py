import json, math, os, joblib, pandas as pd
from flask import Flask, jsonify, request, send_from_directory, abort
from generate_data import FEATURES

app = Flask(__name__, static_folder="static")
bundle = joblib.load("model.joblib"); MODEL, MED = bundle["model"], bundle["medians"]
METRICS = json.load(open("metrics.json"))
DF = pd.read_csv("students.csv")

LIMITS = {"attendance_pct": (0, 100), "attendance_drop": (-50, 100), "avg_marks": (0, 100),
          "marks_trend": (-100, 100), "fee_delay_days": (0, 365), "missed_installments": (0, 12),
          "consecutive_absences": (0, 60)}
LABELS = {"attendance_pct": "Low attendance", "attendance_drop": "Recent attendance drop",
          "avg_marks": "Low marks", "marks_trend": "Falling marks trend",
          "fee_delay_days": "Fee payment delay", "missed_installments": "Missed fee instalments",
          "consecutive_absences": "Consecutive absences"}
ACTIONS = {"attendance_pct": "Mentor call this week; ask about commute, health or family issues.",
           "attendance_drop": "Assign a mentor and discuss the recent absences.",
           "avg_marks": "Offer remedial classes and peer tutoring.",
           "marks_trend": "Faculty review of weak subjects and a study plan.",
           "fee_delay_days": "Discuss an instalment plan with the accounts office.",
           "missed_installments": "Refer to the financial aid or scholarship desk.",
           "consecutive_absences": "Contact the student and guardian to understand the absence."}

def tier(p): return "High" if p >= .7 else "Medium" if p >= .4 else "Low"

def risk(row):
    x = pd.DataFrame([{f: float(row[f]) for f in FEATURES}])
    return float(MODEL.predict_proba(x)[0, 1])

def logit(p): 
    p = min(max(p, 1e-6), 1 - 1e-6); return math.log(p / (1 - p))

def explain(row, p):
    """Occlusion in log-odds: swap each feature for the population median and
    measure how much the risk signal drops. impact = share of total risk signal."""
    raw = []
    for f in FEATURES:
        alt = {k: float(row[k]) for k in FEATURES}; alt[f] = MED[f]
        d = logit(p) - logit(float(MODEL.predict_proba(pd.DataFrame([alt]))[0, 1]))
        if d > .15: raw.append((f, d))
    tot = sum(d for _, d in raw) or 1
    out = [{"factor": LABELS[f], "feature": f, "impact": round(d / tot, 3), "action": ACTIONS[f]} for f, d in raw]
    return sorted(out, key=lambda e: -e["impact"])[:3]

def card(row):
    p = risk(row)
    return {"student_id": row["student_id"], "risk": round(p, 3), "tier": tier(p),
            **{f: float(row[f]) for f in FEATURES}}

SCORED = [card(r) for _, r in DF.iterrows()]
SCORED.sort(key=lambda s: -s["risk"])

@app.get("/api/health")
def health(): return jsonify(status="ok", model=METRICS["selected"], students=len(SCORED))

@app.get("/api/metrics")
def metrics(): return jsonify(METRICS)

@app.get("/api/students")
def students():
    t = request.args.get("tier"); lim = min(int(request.args.get("limit", 100)), 500)
    if t and t not in ("High", "Medium", "Low"): return jsonify(error="tier must be High, Medium or Low"), 400
    rows = [s for s in SCORED if not t or s["tier"] == t][:lim]
    counts = {k: sum(1 for s in SCORED if s["tier"] == k) for k in ("High", "Medium", "Low")}
    return jsonify(total=len(SCORED), counts=counts, students=rows)

@app.get("/api/students/<sid>")
def student(sid):
    s = next((x for x in SCORED if x["student_id"] == sid), None)
    if not s: return jsonify(error="student not found"), 404
    return jsonify(**s, reasons=explain(s, s["risk"]))

@app.post("/api/predict")
def predict():
    d = request.get_json(silent=True)
    if not isinstance(d, dict): return jsonify(error="JSON body required"), 400
    row = {}
    for f in FEATURES:
        if f not in d: return jsonify(error=f"missing field: {f}"), 400
        try: v = float(d[f])
        except (TypeError, ValueError): return jsonify(error=f"{f} must be a number"), 400
        lo, hi = LIMITS[f]
        if not lo <= v <= hi: return jsonify(error=f"{f} must be between {lo} and {hi}"), 400
        row[f] = v
    p = risk(row)
    return jsonify(risk=round(p, 3), tier=tier(p), reasons=explain(row, p))

@app.get("/")
def index(): return send_from_directory("static", "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
