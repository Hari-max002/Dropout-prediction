"""Synthetic student records. Label depends on behaviour features plus noise."""
import numpy as np, pandas as pd

FEATURES = ["attendance_pct", "attendance_drop", "avg_marks", "marks_trend",
            "fee_delay_days", "missed_installments", "consecutive_absences"]

def generate(n=1500, seed=42):
    r = np.random.default_rng(seed)
    g = r.random(n) < 0.3  # latent "struggling" group
    pick = lambda a, b: np.where(g, a, b)
    df = pd.DataFrame({
        "attendance_pct": np.clip(pick(r.normal(64, 14, n), r.normal(85, 8, n)), 20, 100),
        "attendance_drop": np.clip(pick(r.normal(14, 8, n), r.normal(3, 5, n)), -5, 45),
        "avg_marks": np.clip(pick(r.normal(50, 13, n), r.normal(68, 12, n)), 10, 100),
        "marks_trend": np.clip(pick(r.normal(-7, 6, n), r.normal(2, 5, n)), -30, 20),
        "fee_delay_days": np.clip(pick(r.exponential(28, n), r.exponential(6, n)), 0, 120),
        "missed_installments": np.clip(pick(r.poisson(1.3, n), r.poisson(.2, n)), 0, 4),
        "consecutive_absences": np.clip(pick(r.poisson(4, n), r.poisson(1, n)), 0, 15),
    }).round(1)
    z = (-3.2 + .045*(75-df.attendance_pct) + .05*df.attendance_drop + .035*(60-df.avg_marks)
         - .05*df.marks_trend + .02*df.fee_delay_days + .45*df.missed_installments
         + .08*df.consecutive_absences)
    df["dropped_out"] = (r.random(n) < 1/(1+np.exp(-z))).astype(int)
    df.insert(0, "student_id", [f"S{i:04d}" for i in range(1, n+1)])
    return df

if __name__ == "__main__":
    d = generate(); d.to_csv("students.csv", index=False)
    print(d.shape, "dropout rate:", round(d.dropped_out.mean(), 3))
