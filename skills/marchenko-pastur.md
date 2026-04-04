Apply **Technique 2: Marchenko-Pastur Denoising** to the provided return matrix.

Steps:
1. Call `marchenko_pastur_denoise` MCP tool with the (T, n) returns matrix
2. Report λ+ (MP upper edge), number of signal eigenvalues, and % noise removed
3. Describe what the clean correlation matrix looks like vs the raw
4. Recommend whether to use cleaned or raw for downstream optimisation

Key interpretation:
- Eigenvalues above λ+ carry genuine factor signal
- Everything below = sampling noise → should be replaced
- High noise% (>70%) → data is too short or too many assets for the sample size
- Use cleaned matrix for all portfolio construction steps
