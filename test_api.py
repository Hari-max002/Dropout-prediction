import app

c = app.app.test_client()
GOOD = {"attendance_pct": 55, "attendance_drop": 25, "avg_marks": 40, "marks_trend": -12,
        "fee_delay_days": 60, "missed_installments": 2, "consecutive_absences": 8}
SAFE = {"attendance_pct": 95, "attendance_drop": 0, "avg_marks": 85, "marks_trend": 5,
        "fee_delay_days": 0, "missed_installments": 0, "consecutive_absences": 0}

def test_health():
    r = c.get("/api/health"); assert r.status_code == 200 and r.json["status"] == "ok"

def test_frontend_served():
    r = c.get("/"); assert r.status_code == 200 and b"Early Dropout Prediction" in r.data

def test_metrics():
    r = c.get("/api/metrics").json; assert r["selected"] in r["models"] and r["data"] == "synthetic"

def test_students_list_and_filter():
    r = c.get("/api/students?tier=High&limit=5").json
    assert len(r["students"]) == 5 and all(s["tier"] == "High" for s in r["students"])
    assert sum(r["counts"].values()) == r["total"]

def test_students_bad_tier():
    assert c.get("/api/students?tier=Bogus").status_code == 400

def test_student_detail_and_404():
    sid = c.get("/api/students?limit=1").json["students"][0]["student_id"]
    d = c.get(f"/api/students/{sid}").json
    assert d["tier"] == "High" and len(d["reasons"]) >= 1 and "action" in d["reasons"][0]
    assert c.get("/api/students/NOPE").status_code == 404

def test_predict_risky_vs_safe():
    hi = c.post("/api/predict", json=GOOD).json; lo = c.post("/api/predict", json=SAFE).json
    assert hi["risk"] > lo["risk"] and hi["tier"] == "High" and lo["tier"] == "Low"

def test_predict_validation():
    assert c.post("/api/predict", data="x").status_code == 400
    bad = dict(GOOD); del bad["avg_marks"]; assert c.post("/api/predict", json=bad).status_code == 400
    bad = dict(GOOD, attendance_pct=150); assert c.post("/api/predict", json=bad).status_code == 400
    bad = dict(GOOD, avg_marks="abc"); assert c.post("/api/predict", json=bad).status_code == 400

def test_ranked_descending():
    r = c.get("/api/students?limit=50").json["students"]
    assert all(r[i]["risk"] >= r[i+1]["risk"] for i in range(len(r)-1))
