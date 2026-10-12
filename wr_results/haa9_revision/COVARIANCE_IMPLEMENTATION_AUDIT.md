# Covariance implementation audit

The final software audit on 19 September 2026 found that the environmental
binomial GLM covariance had been labelled HC3 even though the installed
statsmodels 0.14.6 implementation calculated the unadjusted Huber-White
sandwich, HC0. `GLMResults` has no `cov_HC3` attribute. The generic covariance
dispatcher in `statsmodels/base/covtype.py` falls back to
`cov_white_simple(..., use_correction=False)` for this result class.

The code now requests HC0 explicitly and the methods and figure captions use
HC0. The stored coefficients and intervals were not manually edited. A
deterministic binomial fixture returned identical covariance matrices for the
old HC3-labelled call and the explicit HC0 call. A complete chemistry rerun is
compared against the pre-correction CSV hashes in the accompanying check.

This is a correction to the covariance description, not a choice of uncertainty
method made to change statistical significance. The lower-substitution TOC
interval continues to include one. These are independent-PWS robust association
models, not causal models or laboratory measurement-error models.
