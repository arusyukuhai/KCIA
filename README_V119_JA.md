> この文書はv119時点の記録です。v120の既定値（公開4・再初期化20）とGUI修正は [README_V120_JA.md](README_V120_JA.md) を参照してください。

# KCIA v119 — room learning intent, GUI spacing and refresh policy

- NEW MODEL now contains PT/SFT/PPO objective settings and a public HF-revision-pinned dataset selection. The signed room manifest includes the compact immutable `training` section when enabled. Local file paths never leave the peer.
- Mixed policy is bound at model creation and restored by new participants. A peer without the necessary datasets may disable SFT/PPO and train PT only, retaining the shared signed objective policy. Note: distributed Critic arrangements may not accept a PT-only participant as a Critic worker; no silent policy rewrites occur.
- LoHA/LoRA reinit cadence is separate from local publication cadence, but NEW rooms default to 4 signed publications rather than 20. `optimization.reinit_every` is immutable per new room, preventing peers from disagreeing. Legacy rooms retain their existing settings.
- GUI spacing restored for KCIA logo (aligned with sidebar content), training metrics, chart toolbar and New Model objective controls.

Caveats: First-party Mac WebView and WAN WebRTC end-to-end testing need actual devices. HF streaming datasets require reachability/auth and exact commit SHAs. Failed runtime dataset loading requires PT-only selection; creating a model does not download or verify all HF data.
