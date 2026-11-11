# Evaluation Methodology

## Philosophy
Most RAG demos report "100% accuracy" on 3 hand-picked queries. This project reports **honest, statistically-annotated metrics** including known limitations.

## Metrics Reference

### Ingestion Module (A)
- **Regulatory coverage:** Target 6/6 (100%)
- **Chunk integrity:** Chunks start uppercase, ≥50 chars (>99%)

### Retrieval Module (B)
- **Hit@K:** Right CELEX + Article in top-K (80%+ at K=2)
- **False-negative rate:** <20%

### Generation Module (C)
- **Unsupported-claim rate:** <10%

### Statistical Rigor
Every rate is reported with raw fractions and Wilson 95% confidence intervals. Sample sizes below 30 are flagged as anecdotal.
