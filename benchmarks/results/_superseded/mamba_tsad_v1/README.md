# Superseded: MambaTSAD v1 results

These 48 runs (`mamba_tsad_faithful` / `mamba_tsad_fixed`, 96 files) were produced by
the **v1** implementation, which was written without reading the official code. After
checking the official repository (aaaceo890/mamba_tsad @ 08fec0531712), most of the v1
"known issues" turned out not to match the official behavior. For example, the official
model feeds the HP *cycle*, not the trend, and v1's "CPU disables selectivity" branch does
not exist upstream.

These results are kept for the record but are **excluded from aggregation**:
`load_all_results` reads only the top level of `benchmarks/results/`. Do not cite them.
The v2 implementation (`model_version = 2`) produces new config hashes, so v1 and v2 runs
can never be merged. See the module docstring of
`tsad_forge/models/gen5_ssm_foundation/mamba_tsad.py` for the verified issue list.
