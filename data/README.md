# Input data

The inputs are **not redistributed** in this repository. They belong to the
upstream model (see `../UPSTREAM.md`) and are fetched from there.

Expected layout under `BPA_DATA_ROOT` (default: this directory):

    data/
    ├── cmip5/<scenario>.csv.zip          daily inflow, 97 CMIP5 scenarios
    ├── lulc/<scenario>.csv               demand-multiplier scenarios
    └── reference/
        ├── nodes.json                    capacity, safe release per node
        ├── historical_medians.csv        median release/storage by dowy
        └── params.json                   upstream baseline policy parameters

Fetch with:

    bash data/fetch_data.sh

If the data already lives elsewhere (e.g. fast local disk on a cluster, which is
strongly preferable — the daily loops open these files constantly):

    export BPA_DATA_ROOT=/scratch/$USER/bpa_data

Every entry point calls `bdps.paths.require_data()`, so a missing or misplaced
input fails immediately with the path it looked for, rather than part-way through
a long run.

## Synthetic streamflow (FIND)

The atlas can be enriched with synthetic flood/drought scenarios from the FIND
generator, which is maintained in its own repository. Only its *outputs* are
consumed here (as additional `cmip5/`-style scenario files); the generator itself
is not vendored.
