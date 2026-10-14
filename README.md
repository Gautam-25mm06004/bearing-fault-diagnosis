# Bearing Fault Diagnosis on CWRU, Evaluated Honestly

Vibration-based fault diagnosis of rolling-element bearings on the Case Western Reserve University (CWRU) benchmark, with signal-processing features (time statistics, spectral band energies, Hilbert-envelope amplitudes at the bearing's fault frequencies) and three classifiers.

The point of the project is evaluation. CWRU papers routinely report 99–100% accuracy, and this project reproduces that number. It then shows that the number mostly measures the split, not the model:

| Protocol (what the test set contains) | Random Forest | Hist. Gradient Boosting | SVM (RBF) |
|---|---|---|---|
| Random window split: windows from recordings also seen in training (leaky) | 99.9% | 100.0% | 100.0% |
| Leave-one-load-out, 10 classes: an unseen motor load (same damaged bearings) | 99.96% | 98.5% | 99.9% |
| **Leave-one-fault-size-out, 4 classes: an unseen damaged bearing** | **66.7%** | **72.3%** | **50.9%** |

The unseen-bearing protocol is the only one where the test bearing is physically different from every training bearing. Fold by fold, Gradient Boosting scores 99.8% when the 0.007" faults are held out, but 50.6% and 66.3% for the 0.014" and 0.021" faults (`results/leave_one_fault_size_out_folds.csv`).

**Feature-group ablation** (Gradient Boosting, unseen-bearing protocol):

| Features used | Count | Accuracy |
|---|---|---|
| Time-domain statistics only | 10 | 46.5% |
| Envelope fault-frequency amplitudes only | 12 | 42.9% |
| Spectral band energies only | 16 | **83.1%** |
| All 38 | 38 | 72.3% |

Two findings come out of this. First, the "near-perfect" CWRU number does not transfer to a new bearing. Second, the coarse spectral shape generalises better than either the impulsiveness statistics or the theoretical fault-frequency peaks. Adding time and envelope features to the spectral bands lowers accuracy on unseen bearings, a sign that they fit bearing-specific detail.

All numbers above come from `results/metrics.json`, produced by `python run_pipeline.py` on 5,886 windows (2,048 samples, 50% overlap) from 40 recordings.

## Data

* CWRU Bearing Data Center, 12 kHz drive-end accelerometer, SKF 6205-2RS bearing, motor loads 0–3 hp (1797–1730 rpm).
* 10 classes: Normal; ball (B), inner-race (IR) and outer-race at 6 o'clock (OR@6) faults of 0.007", 0.014" and 0.021".
* `scripts/download_data.py` fetches the 40 files (about 90 MB) from the GitHub mirror [XiongMeijing/CWRU-1](https://github.com/XiongMeijing/CWRU-1), which keeps the original CWRU files.
* I checked the sample rate of the Normal files from the data itself: at 12 kHz their spectrum peaks exactly at 1×, 2× and 3× shaft speed (29.9 Hz), so all classes share one sample rate.

## Method

1. **Windowing** (`src/bearing/data.py`): 2,048-sample windows (0.17 s), 1,024-sample hop.
2. **Features** (`src/bearing/features.py`):
   * 10 time-domain statistics: RMS, kurtosis, skewness, crest, impulse, shape and clearance factors, peak-to-peak.
   * 16 normalised FFT band energies.
   * 12 envelope features: band-pass 2–5 kHz → Hilbert envelope → amplitude at 1×, 2× and 3× of BPFI (5.4152×), BPFO (3.5848×), BSF (4.7135×) and FTF (0.39828×) shaft speed, divided by the median envelope level so it is comparable across loads.
3. **Models**: Random Forest (400 trees), HistGradientBoosting, and RBF-SVM with scaling (scikit-learn).
4. **Protocols** (`src/bearing/evaluate.py`): random split, leave-one-load-out, and leave-one-fault-size-out.

Figures in `results/figures/`:
* `protocol_comparison.png`: accuracy of the same models under the three protocols.
* `confusion_leave_one_fault_size_out.png`: where the unseen-bearing errors go.
* `confusion_leave_one_load_out.png`: the 10-class load-out confusion matrix.
* `envelope_spectra.png`: envelope spectra with BPFI, BPFO and BSF marked.

## Run it in VS Code (Windows)

```powershell
cd bearing-fault-diagnosis
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py      # one time, ~90 MB
python run_pipeline.py               # ~1-3 minutes on a laptop CPU
python -m pytest -q                  # unit tests, no data needed
```

Open the folder in VS Code and pick the `.venv` interpreter. The Run and Debug panel has ready configurations for the download, the full pipeline, a quick run (`--quick`) and the tests.

## Project layout

```
src/bearing/data.py        CWRU loading, windowing
src/bearing/features.py    time, spectral and envelope features
src/bearing/evaluate.py    three evaluation protocols
run_pipeline.py            end-to-end study -> results/
scripts/download_data.py   fetch the 40 .mat files
tests/test_features.py     5 tests, including a simulated inner-race fault
```

## Explaining it in an interview

* **Why is 99% on CWRU suspicious?** Windows cut from the same 10-second recording are near-duplicates. A random split puts copies of the test signal in training, so the model recognises the recording rather than the fault.
* **Why does leave-one-load-out still score ~100%?** The same damaged bearing is recorded at all four loads. Changing the load shifts shaft speed by about 4%, which is not a new bearing.
* **What would you do next?** Normalise spectra by shaft order (order tracking) so features are speed-invariant. Train a 1-D CNN (WDCNN) under the same unseen-bearing protocol. Test on a second dataset, such as Paderborn, for true transfer.
* **What is the envelope spectrum for?** A local defect produces impacts at a characteristic rate (BPFI, BPFO, BSF) that excite high-frequency resonances. Band-passing around the resonance and demodulating with the Hilbert transform recovers that impact rate.

## References

* W. A. Smith and R. B. Randall, "Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study," *Mechanical Systems and Signal Processing*, 64–65, 2015.
* R. B. Randall and J. Antoni, "Rolling element bearing diagnostics—A tutorial," *Mechanical Systems and Signal Processing*, 25(2), 2011.
* W. Zhang, G. Peng, C. Li, Y. Chen and Z. Zhang, "A new deep learning model for fault diagnosis with good anti-noise and domain adaptation ability on raw vibration signals" (WDCNN), *Sensors*, 17(2), 425, 2017.
* "Towards a more realistic evaluation of machine learning models for bearing fault diagnosis," arXiv:2509.22267.
* "Leakage-Safe, Reproducible Benchmarking for Vibration-Based Fault Diagnosis," PHM Society European Conference.
* Case Western Reserve University Bearing Data Center (original data); GitHub mirror [XiongMeijing/CWRU-1](https://github.com/XiongMeijing/CWRU-1) (data and a 1-D CNN baseline).
