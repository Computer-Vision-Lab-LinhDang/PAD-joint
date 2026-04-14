# OMFR Dataset Survey

## Identity Datasets

| Dataset | #Images | #Subjects | Format | Sensors | Train/Test Split | Access |
|---------|---------|-----------|--------|---------|-----------------|--------|
| **FVC2004** | 3,200 | 100 | BMP (480×640) | 4 sensors (DB1–DB4) | 100 fingers × 8 samples; DB1–3 optical/capacitive, DB4 synthetic | [http://bias.csr.unibo.it/fvc2004/](http://bias.csr.unibo.it/fvc2004/) — free registration |
| **NIST SD302** | 27,000+ | 200 | PNG/WSQ (variable) | 6 sensors (plain/rolled, optical/capacitive/thermal) | No fixed split; use subject-level 80/20 split | [https://www.nist.gov/srd/nist-special-database-302](https://www.nist.gov/srd/nist-special-database-302) — $825 |
| **NIST SD300** | 9,000+ | 75 | WSQ / PNG | Optical livescan (Crossmatch, Identix) | 75 subjects × 10 fingers × 12 impressions; 60/15 subject split | [https://www.nist.gov/srd/nist-special-database-300](https://www.nist.gov/srd/nist-special-database-300) — $850 |
| **NIST SD302a** | 15,000+ | 200 | PNG (500 dpi) | Plain + rolled, multiple sensors | Subset of SD302; annotated quality scores; subject-level 80/20 | Same as SD302 above |

### Notes — Identity Datasets
- **FVC2004**: Most widely used academic benchmark. DB1 (optical), DB2 (optical), DB3 (capacitive), DB4 (synthetic). 8 impressions/finger. Use 4 impressions for training, 4 for testing.
- **NIST SD302**: Large-scale collection. Includes plain and rolled impressions. 6 sensor types. Best for training large-scale identity models.
- **NIST SD300**: Controlled indoor collection. High-quality impressions. 75 subjects × 10 fingers.
- **NIST SD302a**: Annotated quality subset of SD302. Includes per-image NFIQ2 quality scores — useful for quality-aware training.

---

## PAD (Liveness) Datasets

| Dataset | #Images | Live/Spoof | Sensors | Spoof Materials | Train/Test | Access |
|---------|---------|------------|---------|----------------|-----------|--------|
| **LivDet 2013** | 14,230 | 7,168 live / 7,062 spoof | 4 (Biometrika, CrossMatch, Italdata, Swipe) | Ecoflex, gelatin, latex, wood glue, silicone | Per-sensor split; ~3,500/sensor | [https://livdet.org/competitions.php](https://livdet.org/competitions.php) — free |
| **LivDet 2015** | 19,000 | 9,500 live / 9,500 spoof | 4 (Biometrika, Digital Persona, GreenBit, Hi Scan) | Ecoflex, gelatin, latex, wood glue, silicone, body double | ~2,500 train / ~2,000 test per sensor | [https://livdet.org/competitions.php](https://livdet.org/competitions.php) — free |
| **LivDet 2017** | 20,148 | 10,000 live / 10,148 spoof | 4 (Biometrika, Digital Persona, GreenBit, Hi Scan) | Ecoflex, gelatin, latex, wood glue, silicone, RTV | Same sensors as 2015, new materials | [https://livdet.org/competitions.php](https://livdet.org/competitions.php) — free |

### Notes — PAD Datasets
- **LivDet 2013**: First large-scale LivDet. Unknown attack protocol (test spoofs made with materials not seen in training).
- **LivDet 2015**: Improved collection protocol. Includes Digital Persona sensor commonly used in deployment.
- **LivDet 2017**: Cross-sensor and unknown-material challenge. Most difficult of the three.
- All LivDet datasets use **unknown attack** protocol: spoof materials in test ≠ spoof materials in train.

---

## Joint Dataset (Identity + Liveness)

| Dataset | #Images | #Subjects | Live/Spoof | Access |
|---------|---------|-----------|-----------|--------|
| **MSU-FPAD v2.0** | 4,968 | 100 | 2,484 / 2,484 | [http://biometrics.cse.msu.edu/Publications/Databases/MSU_FPAD/](http://biometrics.cse.msu.edu/Publications/Databases/MSU_FPAD/) — request form |

### MSU-FPAD v2.0 Notes
- Per-finger identity labels + liveness labels simultaneously.
- Spoofs: Ecoflex, Play-Doh, gelatin, latex.
- Used exclusively in Phase 3 joint training.

---

## OMFR Dataset Assignment

```
Phase 1 (Identity only):
    identity_ds = IdentityDataset(
        sources=['FVC2004', 'NIST_SD302', 'NIST_SD300', 'NIST_SD302a']
    )

Phase 2 (Alternating):
    identity_ds = same as Phase 1
    pad_ds = PADDataset(sources=['LivDet2013', 'LivDet2015', 'LivDet2017'])

Phase 3 (Joint):
    identity_ds = same
    pad_ds = same
    joint_ds = JointDataset(source='MSU-FPAD')
```

## Download Instructions

### LivDet (2013 / 2015 / 2017)
1. Register at https://livdet.org/competitions.php
2. Download ZIP per competition year
3. Extract to `data/raw/livdet{year}/`
4. Structure: `Train/Live/`, `Train/Fake/`, `Test/Live/`, `Test/Fake/`

### NIST SD302 / SD300 / SD302a
1. Purchase from NIST: https://www.nist.gov/srd/
2. Extract to `data/raw/nist_sd{XXX}/`
3. Structure: one folder per subject, images inside

### FVC2004
1. Register at http://bias.csr.unibo.it/fvc2004/
2. Download DB1–DB4
3. Extract to `data/raw/fvc2004/`
4. Structure: `DB{N}/subject_{NNN}_{impression}.bmp`
