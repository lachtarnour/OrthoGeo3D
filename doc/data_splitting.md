# Data Splitting

The segmentation baseline uses the official TeethSeg22 split definition:

- training combines `public-training-set-1.txt` and
  `public-training-set-2.txt`;
- validation uses `private-testing-set.txt`;
- no separate test split is used during baseline development.

The split is patient-disjoint. It contains 600 patients and 1,200 raw scans for
training, and 300 patients and 600 raw scans for validation. The generated
`patient_overlaps.csv` report must contain zero shared patients. After
documented preprocessing exclusions, 1,196 training scans and 592 validation
scans are available.

Generate the split with:

```bash
python -m scripts.create_patient_splits --source teethseg22
```

The command writes scan lists, patient lists, jaw-specific lists, split
statistics and the patient-overlap report under
`data/splits/teethseg22/`. Preprocessing exclusions are recorded under
`data/processed/teethseg22/_reports/`.

The implementation also supports a seeded patient-random split for controlled
experiments. It is not the reference split used for the reported baseline.
