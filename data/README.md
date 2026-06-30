# Data

This folder contains data files used by the RARE-PHENIX scripts.

## Files

| File | Purpose |
|---|---|
| `HPO_ID_TERM_DEFN.xlsx` | Human Phenotype Ontology term dictionary used by lightweight Module 2 HPO standardization. |

## Required columns for `HPO_ID_TERM_DEFN.xlsx`

The lightweight Module 2 script expects the following columns:

| Column | Description |
|---|---|
| `id` | HPO identifier, for example `HP_0001250`. The script converts this to `HP:0001250` in outputs. |
| `lbl` | HPO term label. |
| `definition` | HPO term definition. |

## Notes

The lightweight Module 2 script excludes obsolete HPO terms by default when the label begins with `obsolete`.

To include obsolete terms, pass:

~~~bash
--include-obsolete
~~~
