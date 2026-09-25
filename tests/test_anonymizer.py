"""匿名化引擎测试：UID 一致性、PHI 清除、非法文件诊断、日期平移、ZIP 安全。"""
from __future__ import annotations

import io
import os
import zipfile

import pydicom
import pytest

from app.anonymizer import Anonymizer, build_zip, safe_extract_zip
from app.config import settings

from .sample_data import build_sample_zip, build_zip_bytes, ct_like_bytes

SECRET = "unit-test-secret-0123456789"


@pytest.fixture()
def processed(tmp_path):
    src_zip = tmp_path / "in.zip"
    meta = build_sample_zip(str(src_zip))
    ex = tmp_path / "extracted"
    ex.mkdir()
    members = safe_extract_zip(
        str(src_zip), str(ex),
        settings.max_extracted_bytes, settings.max_extracted_files,
    )
    out = tmp_path / "out"
    engine = Anonymizer(secret=SECRET, shift_dates=True)
    result = engine.run(str(ex), members, str(out))
    return tmp_path, result, meta


def test_invalid_files_are_reported_per_file(processed):
    _, result, _ = processed
    by_name = {r.name: r for r in result.files}
    assert by_name["broken/bad.dcm"].status == "error"
    codes = [d.code for d in by_name["broken/bad.dcm"].diagnostics]
    assert "INVALID_DICOM" in codes
    assert by_name["notes/not-a-dicom.txt"].status == "error"


def test_valid_files_are_ok_and_written(processed):
    _, result, _ = processed
    valid = [r for r in result.files if r.name.startswith("patient_")]
    assert len(valid) == 4
    assert all(r.status in ("ok", "warning") for r in valid)
    assert all(r.output_name and r.output_name.endswith(".dcm") for r in valid)


def test_patient_pseudonym_shared_across_files(processed):
    _, result, _ = processed
    p1_files = [r for r in result.files if r.name.startswith("patient_1/")]
    ids = {
        next(
            c.new for c in r.changes
            if c.keyword == "PatientID" and c.path in ("", "/")
        )
        for r in p1_files
    }
    assert len(ids) == 1  # 同患者三文件伪名一致


def test_study_relationship_preserved(processed):
    _, result, _ = processed
    p1 = [r for r in result.files if r.name.startswith("patient_1/")]
    study_uids = {r.study_instance_uid for r in p1}
    assert len(study_uids) == 1
    # 不同系列（CT vs RTSTRUCT 各自的 series）
    series_uids = {r.series_instance_uid for r in p1}
    assert len(series_uids) == 2


def test_nested_references_remapped_consistently(processed):
    tmp, result, meta = processed
    rt = next(r for r in result.files if r.name.endswith("RS0001.dcm"))
    ct1 = next(
        r for r in result.files
        if r.name == "patient_1/CT0001.dcm"
    )
    ds = pydicom.dcmread(os.path.join(result.output_dir, rt.output_name))
    nested = (
        ds.ReferencedFrameOfReferenceSequence[0]
        .RTReferencedStudySequence[0]
        .ContourImageSequence[0]
    )
    assert nested.ReferencedSOPInstanceUID == ct1.sop_instance_uid
    # file meta 与顶层 SOP 一致
    assert ds.file_meta.MediaStorageSOPInstanceUID == ds.SOPInstanceUID
    assert ds.StudyInstanceUID == ct1.study_instance_uid


def test_dangling_reference_is_warning(processed):
    _, result, meta = processed
    rt = next(r for r in result.files if r.name.endswith("RS0001.dcm"))
    codes = [d.code for d in rt.diagnostics]
    assert "DANGLING_REFERENCE" in codes
    # 悬空引用仍被确定性重写
    ds = pydicom.dcmread(os.path.join(result.output_dir, rt.output_name))
    new_dangling = ds.ReferencedSOPSequence[0].ReferencedSOPInstanceUID
    assert new_dangling != meta["dangling"]
    assert new_dangling.startswith("2.25.")


def test_phi_fields_removed(processed):
    _, result, _ = processed
    ct = next(r for r in result.files if r.name == "patient_1/CT0001.dcm")
    ds = pydicom.dcmread(os.path.join(result.output_dir, ct.output_name))
    assert len(ds.PatientID) == 16
    assert str(ds.PatientName).startswith("ANON^")
    assert ds.AccessionNumber == ""
    assert ds.InstitutionName == ""
    assert not hasattr(ds, "PatientAddress")
    assert not hasattr(ds, "ImageComments")
    assert not [e for e in ds if e.tag.is_private]
    assert ds.PatientIdentityRemoved == "YES"
    assert ds.DeidentificationMethod
    # 像素数据保持不变
    assert ds.PixelData == bytes(range(32))


def test_change_preview_records_replacements(processed):
    _, result, _ = processed
    ct = next(r for r in result.files if r.name == "patient_1/CT0001.dcm")
    keywords = {c.keyword for c in ct.changes}
    assert {"PatientID", "PatientName", "StudyInstanceUID", "SeriesInstanceUID",
            "SOPInstanceUID"} <= keywords
    # 审计标签记录为新增 REPLACE
    assert "PatientIdentityRemoved" in keywords
    assert "DeidentificationMethod" in keywords
    for c in ct.changes:
        assert c.action in ("REPLACE", "EMPTY", "REMOVE")
        assert c.new is not None or c.action in ("EMPTY", "REMOVE")


def test_dates_are_shifted_and_stable(processed):
    _, result, _ = processed
    ct = next(r for r in result.files if r.name == "patient_1/CT0001.dcm")
    ds = pydicom.dcmread(os.path.join(result.output_dir, ct.output_name))
    assert ds.PatientBirthDate != "19800115"
    assert ds.StudyDate != "20240301"
    # 患者出生日期平移与检查日期使用同一患者偏移（-103 天来自该密钥派生）
    pinfo = next(p for p in result.patients.values()
                 if p.new_patient_id == ds.PatientID)
    assert pinfo.date_shift_days != 0


def test_determinism_across_runs(tmp_path):
    """同密钥、同原始 UID → 同新 UID（患者级稳定性）。"""
    zp = tmp_path / "fixed.zip"
    build_sample_zip(str(zp), seed=42)
    ex = tmp_path / "base"
    ex.mkdir()
    members = safe_extract_zip(str(zp), str(ex),
                               settings.max_extracted_bytes,
                               settings.max_extracted_files)

    def run(secret, tag):
        out = tmp_path / tag
        return Anonymizer(secret=secret).run(str(ex), list(members), str(out))

    r1 = run(SECRET, "out-a")
    r2 = run(SECRET, "out-b")
    r3 = run("another-secret-987654321", "out-c")
    sops1 = {r.name: r.sop_instance_uid for r in r1.files if r.output_name}
    sops2 = {r.name: r.sop_instance_uid for r in r2.files if r.output_name}
    sops3 = {r.name: r.sop_instance_uid for r in r3.files if r.output_name}
    assert sops1 == sops2
    assert sops1["patient_1/CT0001.dcm"] != sops3["patient_1/CT0001.dcm"]


def test_output_zip_flat_and_readable(processed):
    tmp, result, _ = processed
    zpath = build_zip(result, str(tmp / "out.zip"))
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        assert len(names) == 4
        assert all("/" not in n and n.endswith(".dcm") for n in names)
        with zf.open(names[0]) as fh:
            pydicom.dcmread(io.BytesIO(fh.read()))  # 输出可被标准解析器读取


def test_shift_dates_disabled_keeps_dates(tmp_path):
    zp = tmp_path / "in.zip"
    build_sample_zip(str(zp))
    ex = tmp_path / "ex"; ex.mkdir()
    members = safe_extract_zip(str(zp), str(ex),
                               settings.max_extracted_bytes,
                               settings.max_extracted_files)
    result = Anonymizer(secret=SECRET, shift_dates=False).run(
        str(ex), members, str(tmp_path / "out"))
    ct = next(r for r in result.files if r.name == "patient_1/CT0001.dcm")
    ds = pydicom.dcmread(os.path.join(result.output_dir, ct.output_name))
    assert ds.StudyDate == "20240301"
    assert ds.PatientBirthDate == ""  # 出生日期不平移则清空


def _extract(tmp_path, tag="ex"):
    zp = tmp_path / f"{tag}.zip"
    build_sample_zip(str(zp))
    ex = tmp_path / tag
    ex.mkdir()
    members = safe_extract_zip(str(zp), str(ex),
                               settings.max_extracted_bytes,
                               settings.max_extracted_files)
    return str(ex), members


def test_custom_shift_range_bounded_and_deterministic(tmp_path):
    """自定义闭区间：偏移落在区间内，且跨批次（两次独立处理）完全一致。"""
    ex1, m1 = _extract(tmp_path, "ex1")
    ex2, m2 = _extract(tmp_path, "ex2")
    r1 = Anonymizer(
        secret=SECRET, shift_dates=True, date_shift_min=-5, date_shift_max=-5
    ).run(ex1, m1, str(tmp_path / "o1"))
    r2 = Anonymizer(
        secret=SECRET, shift_dates=True, date_shift_min=-5, date_shift_max=-5
    ).run(ex2, m2, str(tmp_path / "o2"))

    shifts1 = {p.key: p.date_shift_days for p in r1.patients.values()}
    shifts2 = {p.key: p.date_shift_days for p in r2.patients.values()}
    assert shifts1 == shifts2  # 跨批次、范围相同 -> 一致
    assert set(shifts1.values()) == {-5}

    ct = next(r for r in r1.files if r.name == "patient_1/CT0001.dcm")
    ds = pydicom.dcmread(os.path.join(r1.output_dir, ct.output_name))
    # 2024-03-01 前移 5 天 -> 2024-02-25
    assert ds.StudyDate == "20240225"
    # 出生日期与检查日期使用同一患者偏移
    assert ds.PatientBirthDate == "19800110"


def test_shift_disabled_ignores_range_and_uses_zero(tmp_path):
    """关闭平移时即使携带范围，偏移仍为 0，DA 保持原值、出生日期清空。"""
    ex, members = _extract(tmp_path, "exoff")
    result = Anonymizer(
        secret=SECRET, shift_dates=False, date_shift_min=-10, date_shift_max=10
    ).run(ex, members, str(tmp_path / "outoff"))
    assert {p.date_shift_days for p in result.patients.values()} == {0}
    ct = next(r for r in result.files if r.name == "patient_1/CT0001.dcm")
    ds = pydicom.dcmread(os.path.join(result.output_dir, ct.output_name))
    assert ds.StudyDate == "20240301"
    assert ds.PatientBirthDate == ""


def test_default_range_matches_pre_range_version(tmp_path):
    """缺省区间下的偏移与未引入范围参数前的 v1 派生完全相同。"""
    from app import crypto

    ex, members = _extract(tmp_path, "exdef")
    result = Anonymizer(secret=SECRET, shift_dates=True).run(
        ex, members, str(tmp_path / "outdef"))
    for p in result.patients.values():
        assert p.date_shift_days == crypto.date_shift_days(SECRET, p.key)


def test_missing_identity_elements_get_pseudonym(tmp_path):
    """输入文件完全没有 PatientID/PatientName 时，输出仍被赋予患者伪名。"""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import MRImageStorage, ExplicitVRLittleEndian

    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = MRImageStorage
    fm.MediaStorageSOPInstanceUID = "1.2.3.9.1"
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = "1.2.3.9.1"
    ds.StudyInstanceUID = "1.2.3.99"
    ds.SeriesInstanceUID = "1.2.3.999"
    ds.Modality = "MR"
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)

    zp = tmp_path / "noid.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("x.dcm", buf.getvalue())
    ex = tmp_path / "ex"
    ex.mkdir()
    members = safe_extract_zip(str(zp), str(ex),
                               settings.max_extracted_bytes,
                               settings.max_extracted_files)
    result = Anonymizer(secret=SECRET).run(str(ex), members, str(tmp_path / "out"))
    rep = result.files[0]
    assert rep.status == "warning"
    out = pydicom.dcmread(os.path.join(result.output_dir, rep.output_name))
    assert len(out.PatientID) == 16
    assert str(out.PatientName).startswith("ANON^")
    pid_changes = [c for c in rep.changes if c.keyword == "PatientID"]
    assert len(pid_changes) == 1  # 不重复记录


# --------------------------------------------------------------------------- #
# ZIP 安全
# --------------------------------------------------------------------------- #
def test_zip_path_traversal_rejected(tmp_path):
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../../etc/passwd", b"x")
    dest = tmp_path / "ex"
    dest.mkdir()
    with pytest.raises(ValueError):
        safe_extract_zip(str(evil), str(dest),
                         settings.max_extracted_bytes,
                         settings.max_extracted_files)


def test_zip_bomb_size_rejected(tmp_path):
    big = tmp_path / "big.zip"
    with zipfile.ZipFile(big, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("a.dcm", b"0" * 1024)
    dest = tmp_path / "ex"
    dest.mkdir()
    with pytest.raises(ValueError):
        safe_extract_zip(str(big), str(dest), max_total_bytes=512, max_files=100)


def test_too_many_files_rejected(tmp_path):
    many = tmp_path / "many.zip"
    with zipfile.ZipFile(many, "w") as zf:
        for i in range(5):
            zf.writestr(f"f{i}", b"x")
    dest = tmp_path / "ex"
    dest.mkdir()
    with pytest.raises(ValueError):
        safe_extract_zip(str(many), str(dest),
                         settings.max_extracted_bytes, max_files=3)


# --------------------------------------------------------------------------- #
# ZIP 同名条目（重复成员名 / 规范化后同址）
# --------------------------------------------------------------------------- #
def _run_zip(tmp_path, zip_bytes, tag="ex"):
    zp = tmp_path / f"{tag}.zip"
    zp.write_bytes(zip_bytes)
    ex = tmp_path / tag
    ex.mkdir()
    members = safe_extract_zip(str(zp), str(ex),
                               settings.max_extracted_bytes,
                               settings.max_extracted_files)
    result = Anonymizer(secret=SECRET).run(
        str(ex), members, str(tmp_path / f"out-{tag}"))
    return result, members


def test_duplicate_member_names_each_get_own_output(tmp_path):
    """两个同名 scan.dcm 条目（不同 SOP）：各自独立处理与写出，互不覆盖。"""
    pixels_a = bytes([7] * 32)
    pixels_b = bytes([9] * 32)
    blob_a = ct_like_bytes("1.2.3.200.1", "1.2.3.200.10", "1.2.3.200.20",
                           "P-200", "Dup^One", "1.2.3.200.30", pixels=pixels_a)
    blob_b = ct_like_bytes("1.2.3.200.2", "1.2.3.200.10", "1.2.3.200.20",
                           "P-200", "Dup^One", "1.2.3.200.30", pixels=pixels_b)
    result, members = _run_zip(
        tmp_path, build_zip_bytes([("scan.dcm", blob_a), ("scan.dcm", blob_b)]))

    # 每个条目独立暂存
    assert len(members) == 2
    assert len({m.staging_name for m in members}) == 2
    assert [m.entry_index for m in members] == [0, 1]

    reps = result.files
    assert [r.name for r in reps] == ["scan.dcm", "scan.dcm"]
    assert [r.entry_index for r in reps] == [0, 1]
    assert all(r.status == "ok" for r in reps)
    # 同名不同 SOP：各自写出为不同的匿名文件
    out0, out1 = reps[0].output_name, reps[1].output_name
    assert out0 and out1 and out0 != out1
    assert reps[0].sop_instance_uid != reps[1].sop_instance_uid
    # 两遍读取的都是各自字节：输出像素与源条目一一对应（未互相覆盖）
    ds0 = pydicom.dcmread(os.path.join(result.output_dir, out0))
    ds1 = pydicom.dcmread(os.path.join(result.output_dir, out1))
    assert ds0.PixelData == pixels_a
    assert ds1.PixelData == pixels_b


def test_normalized_same_path_members_stay_independent(tmp_path):
    """./、a/../ 等规范化后同址的条目同样各自独立暂存与处理。"""
    names = ["./scan.dcm", "scan.dcm", "a/../scan.dcm"]
    blobs = [
        ct_like_bytes(f"1.2.3.201.{i}", "1.2.3.201.10", "1.2.3.201.20",
                      "P-201", "Dup^Two", "1.2.3.201.30", pixels=bytes([i] * 32))
        for i in (1, 2, 3)
    ]
    result, members = _run_zip(
        tmp_path, build_zip_bytes(list(zip(names, blobs))))

    # 三者规范化相对路径相同，但暂存文件各自独立
    assert len({m.rel_path for m in members}) == 1
    assert len({m.staging_name for m in members}) == 3
    # 原始条目名原样回显，条目序号可区分
    assert [r.name for r in result.files] == names
    assert [r.entry_index for r in result.files] == [0, 1, 2]
    assert len({r.output_name for r in result.files}) == 3
    for i, rep in enumerate(result.files):
        ds = pydicom.dcmread(os.path.join(result.output_dir, rep.output_name))
        assert ds.PixelData == bytes([i + 1] * 32)


def test_duplicate_member_same_sop_keeps_duplicate_rule(tmp_path):
    """同名且同 SOPInstanceUID：沿用重复实例改名规则（<uid>.dcm / <uid>-2.dcm）。"""
    pixels_a = bytes([3] * 32)
    pixels_b = bytes([4] * 32)
    blob_a = ct_like_bytes("1.2.3.202.1", "1.2.3.202.10", "1.2.3.202.20",
                           "P-202", "Dup^Three", "1.2.3.202.30", pixels=pixels_a)
    blob_b = ct_like_bytes("1.2.3.202.1", "1.2.3.202.10", "1.2.3.202.20",
                           "P-202", "Dup^Three", "1.2.3.202.30", pixels=pixels_b)
    result, _ = _run_zip(
        tmp_path, build_zip_bytes([("scan.dcm", blob_a), ("scan.dcm", blob_b)]))

    reps = result.files
    assert len(reps) == 2
    new_sop = reps[0].sop_instance_uid
    assert reps[1].sop_instance_uid == new_sop
    assert {r.output_name for r in reps} == {f"{new_sop}.dcm", f"{new_sop}-2.dcm"}
    # 后处理的条目给出重复实例警告
    second = next(r for r in reps if r.entry_index == 1)
    assert "DUPLICATE_SOP_INSTANCE" in [d.code for d in second.diagnostics]
    # 两份输出各自保留源条目字节
    first = next(r for r in reps if r.entry_index == 0)
    ds_first = pydicom.dcmread(os.path.join(result.output_dir, first.output_name))
    ds_second = pydicom.dcmread(os.path.join(result.output_dir, second.output_name))
    assert ds_first.PixelData == pixels_a
    assert ds_second.PixelData == pixels_b


def test_duplicate_member_names_invalid_and_valid(tmp_path):
    """同名条目中一个非法：只影响对应条目，另一份正常写出（不再互相覆盖）。"""
    good = ct_like_bytes("1.2.3.203.1", "1.2.3.203.10", "1.2.3.203.20",
                         "P-203", "Dup^Four", "1.2.3.203.30")
    result, _ = _run_zip(
        tmp_path, build_zip_bytes([("scan.dcm", b"not dicom"), ("scan.dcm", good)]))

    bad, ok = result.files
    assert bad.entry_index == 0
    assert bad.status == "error"
    assert any(d.code == "INVALID_DICOM" for d in bad.diagnostics)
    assert ok.entry_index == 1
    assert ok.status == "ok"
    assert ok.output_name and ok.output_name.endswith(".dcm")
