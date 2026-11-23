"""Nạp 4 file RF/MLP trong tiến trình mới và kiểm tra dự đoán CSV mẫu."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root",type=Path,default=Path(__file__).resolve().parents[1]/"artifacts/models/group_submission")
    args=parser.parse_args()
    root=args.bundle_root.resolve()
    ledger=json.loads((root/"nhu_handoff/checksums.json").read_text(encoding="utf-8"))
    for name,expected in ledger.items():
        path=root/name
        raw=path.read_bytes()
        if path.suffix in {".json",".yaml",".csv",".md",".txt"}:
            raw=raw.replace(b"\r\n",b"\n")
        assert hashlib.sha256(raw).hexdigest()==expected,name
    checked=[]
    for name in ("rf","mlp"):
        folder=root/name
        columns=json.loads((folder/"feature_schema.json").read_text(encoding="utf-8"))["feature_columns"]
        X=pd.read_csv(folder/"example_features.csv")[columns]
        assert len(columns)==68
        for task,labels in (("binary",{0,1}),("multiclass",{1,2,3})):
            model=joblib.load(folder/f"{task}.joblib")
            assert list(model.feature_names_in_)==columns
            assert set(model.classes_)==labels
            predicted=model.predict(X)
            np.testing.assert_array_equal(predicted,pd.read_csv(folder/f"expected_{task}.csv").prediction)
            assert set(predicted).issubset(labels) and len(predicted)==len(X)
            np.testing.assert_allclose(model.predict_proba(X).sum(axis=1),1,atol=1e-8)
            checked.append({"model":name,"task":task,"n_predictions":len(X),"status":"passed"})
            print(f"{name}/{task}: passed",flush=True)
    result={"date":"23/11/2026","status":"passed","fresh_process":True,"models":checked,"checksums_verified":len(ledger)}
    (root/"nhu_handoff/verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
