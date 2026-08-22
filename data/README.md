# Data Layout

The small prepared training files are included in `verl-agent/text/`:

- `train.parquet`
- `test.parquet`

The raw ALFWorld installation is intentionally not included because it is
multi-gigabyte. Set `ALFWORLD_DATA` to a local ALFWorld data directory before
running training. The expected directory contains the standard ALFWorld
`json_2.1.1`, `detectors`, and `logic` subdirectories.
