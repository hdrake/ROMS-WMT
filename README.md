# ROMS-WMT

I ran into memory problems when I tried running the analysis on the entire dataset at once, but a proof-of-concept for the first time-slice is provided by:
- `zarrify_roms.ipynb` - this applies some postprocessing of the raw output and selects just the first time-average (and first 2 snapshots, which bound that time average)
- `wmt.ipynb` - this reads in the above postprocessed zarr file and carries out a full WMT budget calculation


