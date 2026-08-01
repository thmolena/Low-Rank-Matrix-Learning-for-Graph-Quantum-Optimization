# Data provenance

**Dataset:** Sixteen Stanford Gset graphs.  Eight reproducibly permuted
breadth-first induced subgraphs are constructed from each authenticated source.

**Original source:** https://web.stanford.edu/~yyye/yyye/Gset/

**Split:** Six entire source graphs for training, eight for method development,
and two (G67 and G77) for confirmation.  No induced task crosses a source-graph
split.

**Integrity:** `checksums.sha256` pins every downloaded source byte stream.

Run from the repository root:

```bash
python -m pip install -e code
python code/scripts/download_data.py
```

Downloaded third-party files remain governed by the source terms documented in
`THIRD_PARTY.md`. When redistribution is not explicit, the package fetches
the data into an external cache instead of committing the source bytes.
