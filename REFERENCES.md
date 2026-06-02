# HERMES — Literature & Bibliographic References

This document collects the books, papers, and technical reports behind the
**multiple techniques** combined in HERMES (*Hierarchical Entropy Routing Model
with Efficient State-spaces*), a neural adaptive byte compressor.

HERMES is deliberately a *multi-technique* architecture: a byte→patch
tokenizer feeds a stack of **selective state-space (Mamba)** layers, then a
stack of **sparse mixture-of-experts local-window attention** blocks with
**adaptive early-exit** gates, and finally a patch→byte decoder whose logits
drive an **asymmetric-numeral-system (rANS) entropy coder**. Training adds
**EMA weight averaging**, **knowledge distillation**, **mixed precision**, a
**two-phase curriculum**, and an **online test-time-adaptation (OTTA)
meta-loss**.

Each section below names the technique, points to where it lives in the code,
and lists the primary literature. Citations are given with arXiv IDs / DOIs
where available so they can be looked up directly.

> Legend: 📄 paper · 📕 book · 📝 technical report / blog · 🔧 software

---

## 0. Foundations: compression as prediction

The central premise — that a good probabilistic *predictor* of the next byte is
equivalent to a good *compressor* — is information theory. The model outputs a
distribution; arithmetic/ANS coding turns it into bits at the Shannon limit.

- 📄 Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System
  Technical Journal, 27(3), 379–423. — Source coding theorem; entropy as the
  lower bound on lossless code length (the BPC metric this project minimizes).
- 📕 Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory* (2nd
  ed.). Wiley. — Standard reference for entropy, KL divergence, and the
  prediction↔compression equivalence.
- 📕 MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning
  Algorithms.* Cambridge University Press. — Free online; excellent on
  arithmetic coding and probabilistic modeling together.
- 📄 Delétang, G., Ruoss, A., Duquenne, P.-A., et al. (2024). *Language Modeling
  Is Compression.* ICLR 2024. arXiv:2309.10668. — Modern empirical statement of
  the equivalence; directly motivates HERMES's design.
- 📄 Rissanen, J. (1978). *Modeling by shortest data description.* Automatica,
  14(5), 465–471. — Minimum Description Length (MDL); the theoretical frame for
  "predict well = describe shortly".

---

## 1. Selective State-Space Models (Mamba)

**Where:** `hermes/mamba_block.py`, `hermes/model.py` (2 × `MambaBlock`).
Pure-PyTorch selective SSM with ZOH discretization and a memory-efficient
chunked sequential scan; SSM hidden states are carried across chunks for
long-range, file-level context.

- 📄 Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with
  Selective State Spaces.* arXiv:2312.00752. — **The paper the block is built
  on** (input-dependent A/B/C/Δ, selective scan). Cited directly in the source.
- 📄 Dao, T. & Gu, A. (2024). *Transformers are SSMs: Generalized Models and
  Efficient Algorithms Through Structured State Space Duality (Mamba-2).* ICML
  2024. arXiv:2405.21060. — Connects SSMs and attention; chunked scan algorithm.
- 📄 Gu, A., Goel, K. & Ré, C. (2022). *Efficiently Modeling Long Sequences with
  Structured State Spaces (S4).* ICLR 2022. arXiv:2111.00396. — The structured
  SSM that Mamba descends from.
- 📄 Gu, A., Dao, T., Ermon, S., Rudra, A. & Ré, C. (2020). *HiPPO: Recurrent
  Memory with Optimal Polynomial Projections.* NeurIPS 2020. arXiv:2008.07669.
  — The initialization/memory theory underlying S4/Mamba state matrices.
- 📄 Smith, J. T. H., Warrington, A. & Linderman, S. W. (2023). *Simplified State
  Space Layers for Sequence Modeling (S5).* ICLR 2023. arXiv:2208.04933. —
  Parallel-scan SSM formulation.

### 1a. Scan algorithms & discretization

**Where:** `_selective_scan_chunked` in `mamba_block.py` (chunked sequential
scan chosen over a parallel Blelloch scan to fit T4 memory); zero-order-hold
(ZOH) discretization `dA = exp(Δ·A)`.

- 📄 Blelloch, G. E. (1990). *Prefix Sums and Their Applications.* Technical
  Report CMU-CS-90-190, Carnegie Mellon. — The parallel-scan primitive the
  comment in the code refers to (and deliberately avoids for memory reasons).
- 📄 Martin, E. & Cundy, C. (2018). *Parallelizing Linear Recurrent Neural Nets
  Over Sequence Length.* ICLR 2018. arXiv:1709.04057. — Linear recurrences as
  parallel scans; background for why the SSM recurrence can be chunked.
- 📕 Åström, K. J. & Wittenmark, B. (1997). *Computer-Controlled Systems: Theory
  and Design* (3rd ed.). Prentice Hall. — Zero-order-hold discretization of
  continuous linear systems (the `exp(Δ·A)` step).

---

## 2. Sparse Mixture-of-Experts (MoE) routing

**Where:** `hermes/moe_attention.py` (`SparseMoEAttentionBlock`): 4 experts,
per-token top-2 routing, softmax gate, load-balance auxiliary loss.

- 📄 Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017). *Outrageously Large
  Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR 2017.
  arXiv:1701.06538. — The modern top-k sparse gating + load-balancing loss.
- 📄 Jacobs, R. A., Jordan, M. I., Nowlan, S. J. & Hinton, G. E. (1991).
  *Adaptive Mixtures of Local Experts.* Neural Computation, 3(1), 79–87. — The
  original mixture-of-experts idea.
- 📄 Fedus, W., Zoph, B. & Shazeer, N. (2022). *Switch Transformers: Scaling to
  Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 23.
  arXiv:2101.03961. — Top-1 routing, capacity, load-balance loss design.
- 📄 Lepikhin, D., Lee, H., Xu, Y., et al. (2021). *GShard: Scaling Giant Models
  with Conditional Computation and Automatic Sharding.* ICLR 2021.
  arXiv:2006.16668. — Top-2 routing and the auxiliary balancing loss form.
- 📄 Zoph, B., Bello, I., Kumar, S., et al. (2022). *ST-MoE: Designing Stable and
  Transferable Sparse Expert Models.* arXiv:2202.08906. — Router stability and
  the load-balancing/z-loss recipe HERMES's `router_loss` echoes.
- 📄 Jiang, A. Q., Sablayrolles, A., Roux, A., et al. (2024). *Mixtral of
  Experts.* arXiv:2401.04088. — A widely used per-token top-2 MoE; good modern
  reference point.

---

## 3. Attention, local windows, and Flash Attention

**Where:** `LocalWindowAttention` in `hermes/moe_attention.py`: causal
multi-head attention restricted to a 128-token sliding window, computed with
`F.scaled_dot_product_attention` (Flash-Attention path, avoids float16
overflow).

- 📄 Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You
  Need.* NeurIPS 2017. arXiv:1706.03762. — Multi-head scaled dot-product
  attention.
- 📄 Dao, T., Fu, D. Y., Ermon, S., Rudra, A. & Ré, C. (2022). *FlashAttention:
  Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.
  arXiv:2205.14135. — The fused kernel behind `scaled_dot_product_attention`.
- 📄 Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism
  and Work Partitioning.* arXiv:2307.08691. — The kernel generation PyTorch SDPA
  dispatches to on Ampere/T4.
- 📄 Beltagy, I., Peters, M. E. & Cohan, A. (2020). *Longformer: The
  Long-Document Transformer.* arXiv:2004.05150. — Sliding-window (local)
  attention to bound the O(T²) cost — exactly HERMES's `window`.
- 📄 Child, R., Gray, S., Radford, A. & Sutskever, I. (2019). *Generating Long
  Sequences with Sparse Transformers.* arXiv:1904.10509. — Local/strided sparse
  attention patterns.
- 📄 Zaheer, M., Guruganesh, G., Dubey, A., et al. (2020). *Big Bird:
  Transformers for Longer Sequences.* NeurIPS 2020. arXiv:2007.14062. —
  Window + global + random sparse attention.

### 3a. Activations & normalization

**Where:** `_SwiGLU` gated FFN (`moe_attention.py`), GELU + GroupNorm in the
patch encoder/decoder (`byte_patch.py`), LayerNorm throughout.

- 📄 Shazeer, N. (2020). *GLU Variants Improve Transformer.* arXiv:2002.05202. —
  SwiGLU, the gated FFN used in each MoE block.
- 📄 Hendrycks, D. & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs).*
  arXiv:1606.08415. — The GELU activation in the conv tokenizer.
- 📄 Ramachandran, P., Zoph, B. & Le, Q. V. (2017). *Searching for Activation
  Functions (Swish/SiLU).* arXiv:1710.05941. — SiLU, used in the SSM/SwiGLU
  gates.
- 📄 Ba, J. L., Kiros, J. R. & Hinton, G. E. (2016). *Layer Normalization.*
  arXiv:1607.06450.
- 📄 Wu, Y. & He, K. (2018). *Group Normalization.* ECCV 2018. arXiv:1803.08494.
  — GroupNorm in the strided-conv patch encoder/decoder.

---

## 4. Byte-level modeling & patch (sub-word-free) tokenization

**Where:** `hermes/byte_patch.py` — strided `Conv1d` collapses `patch_size=4`
raw bytes into one patch token (4× shorter sequences for the SSM/attention
layers); `ConvTranspose1d` upsamples back to per-byte logits. Vocabulary is the
raw 256 byte values — no learned sub-word tokenizer.

- 📄 Xue, L., Barua, A., Constant, N., et al. (2022). *ByT5: Towards a Token-Free
  Future with Pre-trained Byte-to-Byte Models.* TACL. arXiv:2105.13626. —
  Operating directly on UTF-8 bytes, vocabulary 256.
- 📄 Yu, L., Simig, D., Flaherty, C., et al. (2023). *MEGABYTE: Predicting
  Million-byte Sequences with Multiscale Transformers.* NeurIPS 2023.
  arXiv:2305.07185. — Fixed-size byte *patches* as the unit of computation —
  the same idea as HERMES's patch tokens.
- 📄 Pagnoni, A., Pasunuru, R., Rodriguez, P., et al. (2024). *Byte Latent
  Transformer: Patches Scale Better Than Tokens.* arXiv:2412.09871. —
  Entropy-driven byte patching; closely aligned with HERMES's entropy framing.
- 📄 Clark, J. H., Garrette, D., Turc, I. & Wieting, J. (2022). *Canine:
  Pre-training an Efficient Tokenization-Free Encoder for Language
  Representation.* TACL. arXiv:2103.06874. — Character/byte input with strided
  convolution downsampling (the `patch_conv` mechanism).
- 📄 Graves, A. (2013). *Generating Sequences With Recurrent Neural Networks.*
  arXiv:1308.0850. — Early byte/character-level autoregressive prediction +
  the prediction-as-compression view.

### 4a. Format conditioning ("format token")

**Where:** `hermes/format_sniffer.py` + `format_emb` in `model.py` — the first
bytes are sniffed to a format-class ID, embedded, and prepended as a control
token so the model can specialize per file type.

- 📄 Keskar, N. S., McCann, B., Varshney, L. R., Xiong, C. & Socher, R. (2019).
  *CTRL: A Conditional Transformer Language Model for Controllable Generation.*
  arXiv:1909.05858. — Prepended control codes that condition the distribution —
  exactly the role of the format token.

---

## 5. Adaptive computation & early exit

**Where:** `hermes/early_exit.py` (`EarlyExitGate`) + the exit loop in
`model.py`: each MoE block has a probe head + a confidence head; at inference,
low-entropy (predictable) data skips the remaining blocks.

- 📄 Graves, A. (2016). *Adaptive Computation Time for Recurrent Neural
  Networks.* arXiv:1603.08983. — The foundational "spend compute proportional to
  difficulty" idea.
- 📄 Teerapittayanon, S., McDanel, B. & Kung, H. T. (2016). *BranchyNet: Fast
  Inference via Early Exiting from Deep Neural Networks.* ICPR 2016.
  arXiv:1709.01686. — Confidence-gated early exits with auxiliary classifiers —
  the direct analogue of the probe + confidence heads.
- 📄 Elbayad, M., Gu, J., Grave, E. & Auli, M. (2020). *Depth-Adaptive
  Transformer.* ICLR 2020. arXiv:1910.10073. — Per-token depth selection in
  transformers.
- 📄 Schuster, T., Fisch, A., Gupta, J., et al. (2022). *Confident Adaptive
  Language Modeling (CALM).* NeurIPS 2022. arXiv:2207.07061. — Confidence
  thresholds for early exit in autoregressive LMs; calibration of the exit
  decision.
- 📄 Zhou, W., Xu, C., Ge, T., et al. (2020). *BERT Loses Patience: Fast and
  Robust Inference with Early Exit (PABEE).* NeurIPS 2020. arXiv:2006.04152.
- 📄 Xin, J., Tang, R., Lee, J., Yu, Y. & Lin, J. (2020). *DeeBERT: Dynamic Early
  Exiting for Accelerating BERT Inference.* ACL 2020. — Per-layer exit
  classifiers.

---

## 6. Knowledge distillation (early-exit → final logits)

**Where:** `training/trainer.py::compute_loss` — KL divergence from each
early-exit block's logits to the (detached) final logits (`distill_weight=0.2`),
i.e. self-distillation across depth.

- 📄 Hinton, G., Vinyals, O. & Dean, J. (2015). *Distilling the Knowledge in a
  Neural Network.* NIPS 2014 DL Workshop. arXiv:1503.02531. — Soft-target KD,
  the template for the exit-distillation loss.
- 📄 Zhang, L., Song, J., Gao, A., et al. (2019). *Be Your Own Teacher: Improve
  the Performance of Convolutional Neural Networks via Self Distillation.* ICCV
  2019. arXiv:1905.08094. — Distilling deeper layers into shallower exits —
  precisely HERMES's early-exit distillation.
- 📄 Bucilă, C., Caruana, R. & Niculescu-Mizil, A. (2006). *Model Compression.*
  KDD 2006. — The original "mimic a stronger model" idea.

---

## 7. Entropy coding: arithmetic coding & ANS / rANS

**Where:** `coding/coder.py` — logits → 14-bit CDF quantization
(`batch_logits_to_cdfs`), then ANS coding via the `constriction` library, with a
pure-Python rANS fallback (`_PureANSEncoder`/`_PureANSDecoder`).

- 📄 Duda, J. (2013). *Asymmetric numeral systems: entropy coding combining speed
  of Huffman coding with compression rate of arithmetic coding.*
  arXiv:1311.2540. — **The rANS/ANS algorithm** the coder implements.
- 📄 Witten, I. H., Neal, R. M. & Cleary, J. G. (1987). *Arithmetic Coding for
  Data Compression.* Communications of the ACM, 30(6), 520–540. — Classic
  arithmetic coding; the precision/renormalization techniques mirrored here.
- 📄 Rissanen, J. & Langdon, G. G. (1979). *Arithmetic Coding.* IBM Journal of
  Research and Development, 23(2), 149–162. — Foundational arithmetic coding.
- 📄 Bamler, R. (2022). *Understanding Entropy Coding With Asymmetric Numeral
  Systems (ANS) and Benchmarking with Constriction.* arXiv:2201.01741. — **The
  paper for the `constriction` library** used as the production coder.
- 🔧 Bamler, R. *constriction* — Rust/Python entropy-coding library.
  https://bamler-lab.github.io/constriction/ — the rANS backend.
- 📝 Giesen, F. (2014). *rANS notes / "Interleaved entropy coders".*
  arXiv:1402.3392 and the *ryg_rans* blog series. — Practical rANS
  implementation details (state renormalization, the `1<<16` stream words used
  in the pure-Python fallback).
- 📕 Sayood, K. (2017). *Introduction to Data Compression* (5th ed.). Morgan
  Kaufmann. — Textbook coverage of arithmetic/range/ANS coding and modeling.

---

## 8. Neural & context-mixing lossless compressors (prior art)

These are the systems HERMES is in dialogue with: model-predicts-distribution +
arithmetic/ANS coder.

- 📝 Bellard, F. (2019–2021). *NNCP: Lossless Data Compression with Neural
  Networks.* Technical report. https://bellard.org/nncp/ — Transformer/LSTM byte
  predictor + arithmetic coding; the closest spiritual predecessor.
- 📄 Mahoney, M. (2005). *Adaptive Weighing of Context Models for Lossless Data
  Compression.* Florida Tech TR-CS-2005-16. — Context mixing (PAQ); the
  state-of-the-art classical approach.
- 📄 Knoll, B. & de Freitas, N. (2012). *A Machine Learning Perspective on
  Predictive Coding with PAQ8.* DCC 2012. arXiv:1108.3298.
- 📄 Schmidhuber, J. & Heil, S. (1996). *Sequential Neural Text Compression.*
  IEEE Transactions on Neural Networks, 7(1), 142–146. — First neural predictive
  text compressor.
- 📄 Goyal, M., Tatwawadi, K., Chandak, S. & Ochoa, I. (2019). *DeepZip: Lossless
  Data Compression using Recurrent Neural Networks.* DCC 2019. arXiv:1811.08162.
- 📄 Mao, Y., Cui, Y., Kuo, T.-W. & Xue, C. J. (2022). *TRACE: A Fast
  Transformer-based General-Purpose Lossless Compressor.* WWW 2022.
  arXiv:2203.16114. — Fast transformer byte compressor; a direct benchmark peer.
- 📄 Valmeekam, C. S. K., Narayanan, K., Kalathil, D., Chamberland, J.-F. &
  Shakkottai, S. (2023). *LLMZip: Lossless Text Compression using Large Language
  Models.* arXiv:2306.04050.

---

## 9. Online test-time adaptation (the OTTA meta-loss)

**Where:** `training/trainer.py::_train_epoch` — with probability `otta_prob`,
the SSM state is reset mid-sequence and the tail is re-predicted, penalizing slow
recovery. This trains the SSM to be a *fast in-context adapter*; at inference,
state carried across chunks (`coding/coder.py`) is the adaptation mechanism.

- 📄 Sun, Y., Wang, X., Liu, Z., et al. (2020). *Test-Time Training with
  Self-Supervision for Generalization under Distribution Shifts.* ICML 2020.
  arXiv:1909.13231. — The test-time-training paradigm the OTTA loss emulates.
- 📄 Wang, D., Shelhamer, E., Liu, S., Olshausen, B. & Darrell, T. (2021). *Tent:
  Fully Test-Time Adaptation by Entropy Minimization.* ICLR 2021.
  arXiv:2006.10726. — Adapting at inference without labels.
- 📄 Liang, J., He, R. & Tan, T. (2024). *A Comprehensive Survey on Test-Time
  Adaptation under Distribution Shifts.* IJCV. arXiv:2303.15361. — Survey that
  frames *online* TTA (OTTA), the regime named in the code.
- 📄 Finn, C., Abbeel, P. & Levine, S. (2017). *Model-Agnostic Meta-Learning for
  Fast Adaptation of Deep Networks (MAML).* ICML 2017. arXiv:1703.03400. — The
  meta-learning "learn to adapt fast" framing behind the meta-loss.
- 📄 Hochreiter, S., Younger, A. S. & Conwell, P. R. (2001). *Learning to Learn
  Using Gradient Descent.* ICANN 2001. — RNN states as in-context learners; the
  conceptual root of "the SSM state adapts within a file".

---

## 10. Optimization, weight averaging & mixed-precision training

**Where:** `training/trainer.py` — AdamW (decoupled weight decay, β=(0.9,0.95)),
warmup + cosine LR (`_lr_lambda`), AMP float16 with `GradScaler`, gradient
accumulation, and EMA of weights for inference/export.

- 📄 Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization
  (AdamW).* ICLR 2019. arXiv:1711.05101. — The optimizer used.
- 📄 Kingma, D. P. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.*
  ICLR 2015. arXiv:1412.6980.
- 📄 Loshchilov, I. & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with
  Warm Restarts.* ICLR 2017. arXiv:1608.03983. — The cosine-decay schedule.
- 📄 Micikevicius, P., Narang, S., Alben, J., et al. (2018). *Mixed Precision
  Training.* ICLR 2018. arXiv:1710.03740. — float16 AMP + loss scaling (the
  `GradScaler`).
- 📄 Polyak, B. T. & Juditsky, A. B. (1992). *Acceleration of Stochastic
  Approximation by Averaging.* SIAM J. Control & Optimization, 30(4), 838–855.
  — Iterate averaging; the theory behind EMA weights.
- 📄 Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D. & Wilson, A. G.
  (2018). *Averaging Weights Leads to Wider Optima and Better Generalization
  (SWA).* UAI 2018. arXiv:1803.05407. — `torch.optim.swa_utils.AveragedModel`,
  used here in EMA mode.
- 📄 Tarvainen, A. & Valpola, H. (2017). *Mean Teachers Are Better Role Models.*
  NeurIPS 2017. arXiv:1703.01780. — EMA "teacher" weights for stable targets.

---

## 11. Curriculum learning (two-phase text → binary)

**Where:** `training/trainer.py` + `training/data_pipeline.py` — Phase 1 trains
on text (wikitext-103 + code), Phase 2 on binaries (ELF, `.pyc`, Silesia).

- 📄 Bengio, Y., Louradour, J., Collobert, R. & Weston, J. (2009). *Curriculum
  Learning.* ICML 2009. — Ordering data easy→hard; the staged-phase rationale.
- 📄 Elman, J. L. (1993). *Learning and Development in Neural Networks: The
  Importance of Starting Small.* Cognition, 48(1), 71–99. — Foundational
  "start simple" result.
- 🔧 Silesia Compression Corpus. https://sun.aau.at/~pmeerw/Silesia/ (Deorowicz)
  — The standard binary/mixed benchmark used in Phase 2 and evaluation.

---

## 12. Deployment / native inference (context, not a core method)

**Where:** `export/torchscript_export.py` — TorchScript export of an inference
wrapper for LibTorch (C++) / tch-rs (Rust).

- 🔧 Paszke, A., Gross, S., Massa, F., et al. (2019). *PyTorch: An Imperative
  Style, High-Performance Deep Learning Library.* NeurIPS 2019.
  arXiv:1912.01703. — The framework; TorchScript is its serialization/JIT path.

---

## How the pieces fit (one-paragraph synthesis)

HERMES treats **lossless compression as next-byte prediction** (§0): the model
emits a probability distribution and an **rANS coder** (§7) converts it to bits
near the Shannon limit. To predict bytes cheaply over long files it (a)
**patchifies bytes** with strided convolutions (§4) to shorten sequences 4×, (b)
mixes **selective state-space layers** (§1) for linear-time long-range memory —
whose hidden state is *carried across chunks* and trained to adapt fast via an
**OTTA meta-loss** (§9) — with (c) **sparse-MoE local-window attention** (§2,§3)
for content-specialized, bounded-cost mixing, and (d) **early-exit gates** (§5)
distilled from the final layer (§6) so predictable data spends less compute.
Training stability and quality come from **AdamW + cosine warmup + AMP + EMA**
(§10) over a **text→binary curriculum** (§11).
