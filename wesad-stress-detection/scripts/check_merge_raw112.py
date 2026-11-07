"""Basic audit/strict merge of raw112 packages; keep CSV/ZIP outside Git."""
import argparse
from pathlib import Path
import sys
import subprocess
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from wesad.data import SUBJECTS
from wesad.raw_feature_packages import (VERSION, audit_package, compatibility,
                                      extract_package, merge_complete, select_subjects)
from wesad.utils import file_hash, read_json, resolve_path, write_json


def summary(report):
    lines = ["# Kiểm tra và hợp nhất gói G2 raw112 — bản cơ bản", "",
             "Ngày kiểm tra: 07/11/2026. Nguồn chuẩn: README, `docs/G2_NHU_BAN_CO_BAN.md`, "
             "processor `wesad.multisensor:process_subject`. Chỉ kiểm tra/ghép bộ raw112; không áp bộ lọc G2, "
             "không tổng hợp ACC/EMG 5 giây và không thay hợp đồng sinh lý68.", "",
             "## Gói được kiểm tra chính", ""]
    if report.get("status") == "failed":
        return "\n".join(lines + ["Kiểm tra không đạt: " + report["error"], ""])
    primary = report["reference"]
    lines += [f"Kết quả: **hợp lệ**; `{VERSION}`, 112 feature + 8 metadata, 63 ngực/49 cổ tay.",
              f"Tổng **{primary['n_windows']:,} cửa sổ**. Đã kiểm tra config, schema, manifest, chữ ký, checksum, "
              "cột/thứ tự, nhãn, khóa, NaN/Inf và all_subjects so với CSV từng người. "
              "all_subjects chỉ dùng đối chiếu, không được ghép thêm lần thứ hai.", "",
              "| Subject | Cửa sổ | Baseline | Stress | Amusement | NaN / Inf / khóa trùng | Raw hash kiểm tra với bản local |",
              "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for item in primary["subjects"]:
        c = item["label_counts"]
        lines.append(f"| {item['subject_id']} | {item['n_windows']} | {c.get('1', 0)} | {c.get('2', 0)} | {c.get('3', 0)} | 0 / 0 / 0 | {'Khớp' if item['local_raw_hash_verified'] else 'Không kiểm tra'} |")
    lines += ["", f"Archive SHA-256: `{report['archive_sha256']}`.",
              f"Processing signature: `{primary['signature']}`.",
              f"Code SHA-256 được gói ghi nhận: `{primary['code_sha256']}`.",
              f"Git commit ghi trong gói: `{primary['git_commit']}`; git_dirty=`{primary['git_dirty']}`. "
              "Không sửa snapshot/manifest của gói nhận được.", "", "## Đối chiếu các gói đang có", "",
              "| Gói | Kết luận | Lý do |", "| --- | --- | --- |"]
    for item in report["other_packages"]:
        if item["status"] == "compatible":
            status, reason = "Hợp lệ, cùng nguồn chạy", "Đủ điều kiện xét hợp nhất"
        elif item["status"] == "provenance_mismatch":
            status = "Cùng định nghĩa khai báo, chưa cho ghép"
            reason = "Khác " + ", ".join(item["comparison"]["different_fields"])
        elif item["status"] == "excluded_version":
            status, reason = "Loại khỏi tập raw112", "Phiên bản " + str(item.get("feature_version"))
        else:
            status, reason = "Không đủ điều kiện", item.get("error", "Không hợp lệ")
        lines.append(f"| `{Path(item['path']).name}` | {status} | {reason.replace('|', '/')} |")
    lines += ["", "Khác metadata phiên bản/cấu hình/định nghĩa là lý do từ chối. "
              "Đợt cơ bản này còn yêu cầu cùng chữ ký mã/config/môi trường để nhận gói ghép; "
              "các gói cùng raw112 nhưng khác nguồn chạy được ghi riêng, không áp dụng ngoại lệ G1 vào G2.", "",
              "## Kết quả hợp nhất", "",
              "Subject đủ điều kiện trong tập tương thích với gói chính: **" + ", ".join(report["eligible_subjects"]) + "**."]
    if report["missing_subjects"]:
        lines += ["**Chưa hợp nhất đủ 15 người**. Còn thiếu: **" + ", ".join(report["missing_subjects"]) + "**.",
                  "Không xuất all_subjects.csv dưới danh nghĩa bộ hoàn chỉnh; không lấy 68/175 feature hay gói khác chữ ký để lấp thiếu."]
    elif report.get("merged"):
        lines.append("Đã hợp nhất đủ 15 người vào `" + str(report.get("merge_output")) + "`.")
    else:
        lines.append("Đã đủ 15 subject tương thích, nhưng chưa xuất gói hợp nhất vì chưa yêu cầu --merge-output.")
    if report.get("merge_error"):
        lines.append("Hợp nhất bị từ chối: " + report["merge_error"])
    lines += ["", "Khi nhận S2 nhiều lần: kiểm tra raw hash, metadata/khóa và feature theo rtol=1e-7, atol=1e-9; "
              "giữ một subject duy nhất nếu khớp, từ chối nếu xung đột. Chỉ đọc CSV từng subject để ghép.", "",
              "## Lệnh tái kiểm tra", "", "```powershell", report["reproduction_command"], "```", "",
              f"Báo cáo JSON chi tiết: `{report['report_path']}`. Các ZIP/CSV đã nhận/đầu ra giữ ngoài Git theo .gitignore.", "",
              "## TODO sau bản cơ bản", "",
              "- Nhận các gói raw112 còn thiếu và thống nhất source/môi trường trước lượt ghép chung.",
              "- Bổ sung bằng chứng phiên bản source sạch; không tự sửa chữ ký hoặc bỏ kiểm tra để ép khớp.",
              "- Nếu tích hợp tiền xử lý mới, thống nhất phiên bản/schema riêng và kiểm tra lại; giữ bộ raw112 này nguyên định nghĩa.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, help="Primary ZIP/RAR or unpacked package")
    parser.add_argument("--discover-dir", help="Recursively inspect other local package directories")
    parser.add_argument("--extract-root", default="data/features/received_g2")
    parser.add_argument("--raw-root", help="Optional hash-only comparison with local PKL; no signal extraction")
    parser.add_argument("--expected-subjects", nargs="+")
    parser.add_argument("--expected-windows", type=int)
    parser.add_argument("--report", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--merge-output", help="Written only if all 15 compatible subjects are present")
    args = parser.parse_args()
    report = {"status": "failed"}
    try:
        primary_path = resolve_path(args.package)
        archive_hash = None
        if primary_path.suffix.lower() in (".zip", ".rar"):
            archive_hash = file_hash(primary_path)
            target = resolve_path(args.extract_root)/(primary_path.stem+"_"+archive_hash[:12])
            marker = target/".archive_sha256"
            if target.exists():
                if not marker.is_file() or marker.read_text().strip() != archive_hash:
                    raise ValueError("Extraction destination exists without matching archive identity")
                roots = [p.parent for p in target.rglob("config.json") if (p.parent/"run_status.json").exists()]
                if len(roots) != 1:
                    raise ValueError("Ambiguous extracted package")
                primary_path = roots[0]
            else:
                primary_path = extract_package(primary_path, target)
                marker.write_text(archive_hash, encoding="ascii")
        raw_root = resolve_path(args.raw_root) if args.raw_root else None
        primary = audit_package(primary_path, raw_root)
        if args.expected_subjects and set(primary.frames) != set(args.expected_subjects):
            raise ValueError("Primary package subjects differ from expected handoff")
        if args.expected_windows is not None and primary.report["n_windows"] != args.expected_windows:
            raise ValueError("Primary package row count differs from expected handoff")
        candidates = []
        if args.discover_dir:
            candidates = sorted({p.parent.resolve() for p in resolve_path(args.discover_dir).rglob("config.json")
                                 if (p.parent/"run_status.json").exists()} - {primary.path})
        eligible, others = [primary], []
        for path in candidates:
            item = {"path": str(path)}
            try:
                version = read_json(path/"config.json").get("processing", {}).get("feature_version")
                if version != VERSION:
                    item.update(status="excluded_version", feature_version=version)
                else:
                    candidate = audit_package(path, raw_root)
                    result = compatibility(primary, candidate)
                    item.update(status="compatible" if result["eligible_for_strict_merge"] else "provenance_mismatch",
                                audit=candidate.report, comparison=result)
                    if result["eligible_for_strict_merge"]:
                        eligible.append(candidate)
            except (ValueError, KeyError, TypeError, OSError, AssertionError) as exc:
                item.update(status="invalid_or_conflict", error=str(exc))
            others.append(item)
        report.update(reference=primary.report, archive_sha256=archive_hash, other_packages=others)
        chosen, duplicates = select_subjects(eligible)
        missing = sorted(set(SUBJECTS)-set(chosen), key=lambda s: int(s[1:]))
        report = {"status": "waiting_for_compatible_packages" if missing else "ready_to_merge",
                  "reference": primary.report, "archive_sha256": archive_hash, "other_packages": others,
                  "eligible_subjects": sorted(chosen, key=lambda s: int(s[1:])), "missing_subjects": missing,
                  "duplicate_subjects_not_appended": duplicates, "merged": False,
                  "report_path": args.report,
                  "reproduction_command": ".\\.venv-g1\\Scripts\\python.exe scripts/check_merge_raw112.py " + " ".join(sys.argv[1:])}
        if not missing and args.merge_output:
            merged = merge_complete(eligible, resolve_path(args.merge_output))
            report.update(status="merged_complete_15_subjects", merged=True, merge_output=args.merge_output,
                          merged_package=merged.report)
    except (ValueError, KeyError, TypeError, OSError, AssertionError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        report.update(status="failed", error=str(exc))
    write_json(resolve_path(args.report), report)
    path = resolve_path(args.summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary(report), encoding="utf-8")
    print(report["status"])
    if "reference" in report:
        print(f"Primary: {report['reference']['n_windows']} windows; missing: {report.get('missing_subjects', 'not computed')}")
    else:
        print(report.get("error", "Failed"), file=sys.stderr)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
