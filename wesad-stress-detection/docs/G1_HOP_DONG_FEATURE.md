# Hợp đồng feature sinh lý G1 — physiological_68_v1

Đã khóa thiết kế danh mục, công thức, thông số, đơn vị, nhãn và dữ liệu thiếu ngày 05/11/2026 theo yêu cầu hoàn thiện phần Mai. Các lựa chọn detector/ngưỡng/PSD là lựa chọn thiết kế của phần Mai theo yêu cầu người dùng cho lần triển khai kế tiếp, chưa phải kết quả thực nghiệm hoặc xác nhận review của thành viên khác. Không dùng trạng thái thiết kế này thay cho nghiệm thu extractor.

## Đầu ra chính

- `configs/feature_schema.json`: hợp đồng sinh lý 68 feature, `design_locked_implementation_not_validated`.
- `docs/G1_DANH_MUC_FEATURE.csv`: 68 tên/công thức/đơn vị/thứ tự đồng nhất JSON, 37 ngực và 31 cổ tay.
- Bằng chứng đơn vị được gộp vào `unit_policy.unit_evidence` trong schema chính và phần đơn vị ở tài liệu này.
- Schema raw 112 thực tế lấy từ `feature_schema.json` sinh cùng thư mục gói S2 ngoài Git; không tạo thêm schema/danh mục raw trùng lặp trong Git.
- Danh mục sinh lý trước đây đã được hoàn thiện ngay trong schema chính; không duy trì thêm bản kế hoạch trùng lặp.

## Cửa sổ, nhãn và cột

15 người S2–S11/S13–S17, seed 42. Cửa sổ 60 giây, bước 5 giây, khoảng [start,end) trên cùng mốc PKL. Majority ≥90% tính trên toàn bộ nhãn gốc; chỉ nhận 1/2/3, không nối các đoạn sau khi bỏ nhãn. Nhị phân: 2→1, 1/3→0; nhãn khác không thành Non-stress.

Tám metadata đứng trước: subject_id, window_id, start_s, end_s, label_original, label_binary, label_fraction, quality_flag. window_id dùng chỉ số bắt đầu trên lưới 700 Hz. Khóa (subject_id,window_id) duy nhất. Feature theo đúng thứ tự JSON/CSV; chỉ feature làm X. Các so sánh cảm biến dùng chung cửa sổ đủ điều kiện; combined = chest nối wrist.

## Đơn vị đã quyết định

Theo README WESAD gốc II.1–II.3: chest ECG/EMG mV, EDA µS, TEMP °C, ACC g, RESP %. Mẫu PKL S2 đã khớp công thức chuyển đổi tại đúng dòng raw 92017; không chuyển ADC lần nữa. Wrist EDA µS, TEMP °C; BVP a.u.; ACC PKL khớp CSV 1/64 g nên chia 64 đúng một lần trước feature. Phép kiểm chứng mẫu này không thay một cuộc kiểm tra toàn bộ 15 phiên thu; hợp đồng chỉ nhận PKL WESAD chính thức, nguồn khác cần adapter có khai báo đơn vị.

HR/pulse rate bpm; RR/IBI ms; pNN50 tỷ lệ 0–1; RESP breaths/min. PSD amplitude ECG mV², BVP a.u.²; không phải LF/HF từ chuỗi RR. Tích phân ACC g·s, EMG mV·s; slope TEMP °C/s; waveform length EMG mV/s. Bằng chứng cụ thể nằm trong `unit_policy.unit_evidence` của schema chính: 100 mẫu ngực S2 căn tại dòng raw 92017 khớp công thức gốc; 20 mẫu đầu của từng kênh cổ tay khớp một đoạn trong CSV gốc. Đây là kiểm chứng mẫu, không phải kiểm tra toàn bộ 15 người.

## Định nghĩa chung

Mean/std tín hiệu và tốc độ dùng ddof=0; SD RR/IBI dùng ddof=1. RR/IBI tính bằng chênh lệch đỉnh. RMSSD và pNN50 chỉ dùng các cặp khoảng liền nhau trong chuỗi gốc cùng hợp lệ, không nối qua khoảng bị loại. Tên mean_nn/sdnn được đổi rõ thành mean_rr/sd_rr vì detector heuristic không xác minh nhịp ngoại tâm thu; danh sách đổi tên có trong change_policy.

TEMP slope = sum((t−mean(t))*(x−mean(x)))/sum((t−mean(t))²), t tính bằng giây. ACC giữ trọng lực, magnitude=sqrt(x²+y²+z²). ACC/EMG chia đúng 12 đoạn 5 giây; mỗi feature cuối là mean của 12 giá trị. ACC tính mean/std/sum(abs(x))/fs; EMG tính RMS/MAV/sum(abs(diff(x)))/5/sum(abs(x))/fs. Không lấy nanmean trên ít hơn 12 đoạn.

## Thông số thuật toán đã khóa

Các mục dưới đây được xuất trực tiếp từ `methods` trong schema để tránh tài liệu và JSON lệch nhau. Thiết kế giữ cutoff EDA/RESP của Như; bổ sung detector, PSD và biên tonic để triển khai G2. Lọc trên đoạn hữu hạn liên tục trước cắt cửa sổ, offline hai chiều. Không lọc xuyên NaN/Inf.
### numeric

```json
{
  "dtype": "float64",
  "signal_std_ddof": 0,
  "interval_sd_ddof": 1,
  "pnn50_threshold_ms": 50,
  "pnn50_unit": "fraction 0..1",
  "band_power": "integral of amplitude-signal PSD, not RR spectral HRV; do not call LF/HF"
}
```

### filter

```json
{
  "api": "scipy.signal.butter(order=4,fs=native_hz,output=sos), sosfiltfilt(padtype=odd,padlen=3*(2*n_sections+1-min(zeros_at_origin,poles_at_origin)))",
  "scope": "each continuous finite subject segment before slicing; never cross NaN/Inf or concatenate label-separated spans",
  "edges_seconds_each_end": {
    "ecg": 10,
    "bvp": 10,
    "emg": 1,
    "eda_cleaned": 10,
    "eda_tonic_phasic": 60,
    "resp": 30
  },
  "short_segment": "unusable when len<=padlen or no eligible interior after removing edges",
  "offline_only": true
}
```

### ecg

```json
{
  "bandpass_hz": [
    5,
    40
  ],
  "rate_hz": 700,
  "detection_bandpass_hz": [
    5,
    20
  ],
  "detector": "squared detection-band signal; centered 0.10 s boxcar (70 samples), clipped segment boundaries; find_peaks(distance=210,prominence=max(0.5*std(envelope),1e-12)); refine each candidate to argmax(abs(detection_signal)) in +/-0.10 s; unique sorted indices; if refinement distance<0.30 s retain larger absolute peak, ties earlier",
  "interval_method": "cardiac_intervals",
  "interpretation": "ECG RR variability with heuristic artifact rejection; no clinical normal-to-normal classification"
}
```

### bvp

```json
{
  "bandpass_hz": [
    0.5,
    5
  ],
  "rate_hz": 64,
  "detector": "find_peaks on bandpassed BVP, distance=ceil(0.30*64)=20, prominence=max(0.5*std(filtered),1e-12)",
  "interval_method": "cardiac_intervals",
  "interpretation": "pulse rate and PRV; not ECG HRV"
}
```

### cardiac_intervals

```json
{
  "rule": "RR=diff(peaks)/fs. Stage1 keep 0.30<=RR<=2.00 s. m=median(stage1 RR); Stage2 keep stage1 intervals with abs(RR-m)<=0.20*m, inclusive. No interpolation or filling.",
  "minimum_accepted_intervals": 20,
  "minimum_accepted_fraction": 0.8,
  "minimum_originally_adjacent_accepted_pairs": 10,
  "fail": "all six rate/interval features NaN; flag device_sensor:insufficient_or_unreliable_beats",
  "pairs": "diff(original RR) only where both adjacent original RR pass; never bridge a rejected RR",
  "formula": "rate=60/RR; mean/std(rate,ddof=0); mean(RR)*1000; std(RR,ddof=1)*1000; sqrt(mean(adjacent_diff**2))*1000; mean(abs(adjacent_diff)>0.05)"
}
```

### eda

```json
{
  "clean_lowpass_hz": {
    "chest": 5,
    "wrist": 1
  },
  "tonic_lowpass_hz": 0.05,
  "decomposition": "tonic=Butterworth4 SOS forward/back lowpass of cleaned EDA; phasic=cleaned-tonic. Explicit filter-based approximation, not cvxEDA.",
  "detector": "find_peaks(phasic,distance=ceil(fs*1.0),prominence=0.01). For each peak find last minimum in preceding 4 s (inclusive), capped strictly after previous accepted peak; onset-to-peak amplitude = phasic[peak]-phasic[onset]. Accept when amplitude>=0.01 uS and 0.25<=rise_seconds<=4.0, onset and peak both inside the eligible window.",
  "count_rule": "detect on continuous finite segments, assign only complete onset/peak events within [start,end); count each event once per window, overlapping windows may contain same event",
  "no_event": "usable signal with no accepted events: count=0,rate=0, amplitude_mean/max=NaN; flag sensor:no_scr_events",
  "unusable": "all eight EDA features NaN and flag sensor:flat_or_unusable; never encode detector failure as zero"
}
```

### resp

```json
{
  "bandpass_hz": [
    0.1,
    0.35
  ],
  "detector": "find_peaks(filtered,distance=ceil(2.0*fs),prominence=max(0.5*std(filtered),1e-12)); use adjacent peaks both in window",
  "interval_range_seconds": [
    2,
    15
  ],
  "minimum_valid_intervals": 3,
  "minimum_valid_fraction": 0.8,
  "formula": "rate_i=60/diff(peaks_seconds); rate_mean=mean(rate_i), rate_std=std(rate_i,ddof=0)",
  "fail": "both rate features NaN; chest_resp:insufficient_cycles"
}
```

### psd

```json
{
  "api": "scipy.signal.welch",
  "input": "ECG 5-40 Hz signal and BVP 0.5-5 Hz signal after locked filtering; no extra resampling",
  "window": "get_window(hann,4*fs,fftbins=True)",
  "nperseg": "4*fs",
  "noverlap": "2*fs",
  "nfft": "4*fs",
  "detrend": "constant",
  "scaling": "density",
  "return_onesided": true,
  "average": "mean",
  "bands_hz": {
    "chest_ecg": [
      [
        5,
        10
      ],
      [
        10,
        20
      ],
      [
        20,
        40
      ]
    ],
    "wrist_bvp": [
      [
        0.5,
        1
      ],
      [
        1,
        2
      ],
      [
        2,
        5
      ]
    ]
  },
  "integration": "numpy.trapezoid(Pxx[low<=f<=high],f[low<=f<=high]); all boundaries lie exactly on 0.25-Hz bins; adjacent bands sharing endpoints do not duplicate area",
  "output_units": {
    "chest_ecg": "mV^2",
    "wrist_bvp": "a.u.^2"
  }
}
```

### temp

```json
{
  "input": "original calibrated signal",
  "mean": "mean(x)",
  "slope": "sum((t-mean(t))*(x-mean(x)))/sum((t-mean(t))**2), t=arange(n)/fs seconds",
  "known_quality": "S2 chest TEMP flagged suspect_contact per subject readme: output both chest TEMP features NaN for the whole recording; retain common windows; flag chest_temp:suspect_contact"
}
```

### acc_emg

```json
{
  "block_seconds": 5,
  "blocks_per_window": 12,
  "aggregation": "arithmetic mean of all 12 per-block feature values; require all 12 blocks",
  "acc": "convert wrist ACC /64 once; chest ACC unchanged; retain gravity; x,y,z,magnitude=sqrt(x*x+y*y+z*z)",
  "acc_formulas": [
    "mean(x)",
    "std(x,ddof=0)",
    "sum(abs(x))/fs"
  ],
  "emg_bandpass_hz": [
    20,
    250
  ],
  "emg_formulas": [
    "sqrt(mean(x*x))",
    "mean(abs(x))",
    "sum(abs(diff(x)))/5",
    "sum(abs(x))/fs"
  ],
  "missing_block": "exclude entire common window; do not nanmean fewer than 12 blocks"
}
```

### qa

```json
{
  "common_window_selection": "all required channels must cover full window and pass finite/sample/filter-edge checks; reject once for every sensor group, record reason counts",
  "flat_threshold": "std(channel)<=1e-10 in calibrated output units. Mark signal-dependent physiological features NaN; do not automatically discard otherwise eligible common windows. ACC/TEMP descriptive values remain defined except known suspect-contact mask.",
  "detector_exception": "catch expected detector failure only; mark affected feature group NaN and detector_failure reason; unexpected errors fail run",
  "nonfinite_output": "NaN allowed only for declared reasons; Inf raises validation error",
  "flags": "unique sorted sensor:reason codes joined by semicolon; if none use heuristic_checks_passed_not_physiological_validation",
  "no_imputation": "do not impute during extraction; fit imputer/scaler/selection/resampling on train-fold only"
}
```

### reproducibility

```json
{
  "design_choices": "All thresholds/algorithms are group design decisions for implementation, not claims of original paper parameters or clinical validation. Changes require new feature_version.",
  "python": "3.12.14",
  "libraries": "requirements.lock.txt",
  "comparison": {
    "metadata_and_order": "exact",
    "feature_rtol": 1e-06,
    "feature_atol": 1e-08,
    "nan_masks": "exactly equal; equal_nan allowed only when quality reasons match"
  },
  "acceptance": "verify implementation on synthetic invariants and physiological S2 before accepting 15-subject extraction; raw 21/112 S2 does not certify this design"
}
```

## Thiếu và chất lượng

Feature thiếu là NaN, CSV ô trống; không dùng 0/-999 để che lỗi. Metadata không được thiếu. Inf là lỗi xác thực. Detector không đủ tin cậy tạo NaN cho nhóm liên quan và mã nguyên nhân; không loại riêng từng nhóm cảm biến khỏi tập khóa chung. SCR không sự kiện: count/rate=0, amplitude mean/max=NaN và no_scr_events. S2 chest TEMP có cảnh báo tiếp xúc từ readme: giữ cửa sổ nhưng hai feature TEMP là NaN, chest_temp:suspect_contact.

Cờ được sắp xếp, loại trùng, nối bằng dấu chấm phẩy. Khi không có cảnh báo chỉ ghi heuristic_checks_passed_not_physiological_validation; không tuyên bố chất lượng lâm sàng. Imputer/scaler/selection/resampling học trong train-fold.

## Phạm vi kiểm tra S2 của Mai

Theo phân công G1, cả bốn chạy mẫu S2 trên cùng raw/mã/môi trường/cấu hình. Việc riêng của Mai là danh mục feature, đơn vị, nhãn nhị phân, cách ghi thiếu và schema. Kết quả trong phần “Kiểm tra S2 và người tổng hợp” của `docs/BAN_GIAO.md` chỉ ghi kiểm tra gói S2 trên máy Mai trước khi gửi, không nhận nhiệm vụ tổng hợp hay ký nghiệm thu bốn máy.

Theo cách tổ chức nhóm người dùng đã xác định, Như nhận các gói ZIP và tổng hợp đối chiếu. Mai gửi nguyên gói `data/features/g1_s2_raw112_py31214_fa7b694/` ngoài Git; không gửi môi trường Python. Gói phải giữ CSV, schema, config, manifest và checksum. Kết quả `passed` cục bộ không thay kết quả đối chiếu của Như.

## Phiên bản và nghiệm thu

Đổi cột, công thức, đơn vị, detector, ngưỡng hoặc cửa sổ phải tăng feature_version. Danh sách 68 cột và thông số đã chốt để triển khai; ready_for_extraction/training vẫn false vì extractor đúng hợp đồng chưa triển khai/kiểm chứng. Bộ thử physio_baseline_v1 (175 cột) không được coi là implementation của hợp đồng này.

Mẫu S2 đã passed trước đây là 21 raw, không chứng minh bộ 68 sinh lý đạt; bộ 112 raw cũng không thay nghiệm thu sinh lý. Cần kiểm tra implementation trên dữ liệu tổng hợp và S2 trước khi trích đủ 15 người. So metadata/cột/NaN mask và lý do chính xác; feature sinh lý rtol=1e-6, atol=1e-8 theo lựa chọn thiết kế đã ghi; chưa tuyên bố bốn máy đạt.

Nguồn phương pháp/API: [WESAD tác giả](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html), [SciPy Welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html), [SciPy find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html). Các tham số cụ thể ở đây do nhóm chọn để triển khai, không quy cho tác giả/API.
