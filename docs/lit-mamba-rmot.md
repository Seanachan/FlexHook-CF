# Literature Review: Mamba / State-Space Models for Tracking & RMOT

> Deep-research workflow (4 angles → web search → fetch → 3-vote adversarial verify → cited synthesis). Run 2026-06-06.

## Research question

Have Mamba / selective state-space models (SSMs) been used for tracking, and specifically for Referring Multi-Object Tracking (RMOT)? (1) Mamba/SSM in multi-object tracking (MOT) — motion modeling, trajectory/state prediction, data association (e.g. any MambaTrack or SSM-based MOT motion models, 2024-2026); real papers with venue/year/arXiv id. (2) Mamba/SSM in single-object & visual object tracking (SOT/VOT) and in video understanding backbones relevant to temporal trajectory modeling (VideoMamba, Vision Mamba/Vim, VMamba). (3) Mamba/SSM in RMOT or language-conditioned / referring tracking or referring video object segmentation (RVOS) specifically — does ANY published work exist? Note clearly if absent. (4) Reported tradeoffs of Mamba vs transformer attention vs LSTM for temporal modeling, especially short vs long sequence/temporal windows, and linear-time long-sequence claims. Prioritize verifiable citations; explicitly flag where evidence is absent.

## Executive summary

Yes, Mamba/selective state-space models (SSMs) have been actively applied to tracking across 2024-2026, but the coverage is highly uneven by sub-domain. In multi-object tracking (MOT), SSMs are well-established as learned motion/trajectory predictors that replace or augment the Kalman filter — MambaTrack (ACM MM 2024), TrackSSM (arXiv 2024 / Science China Info Sci), Samba/SambaMOTR (ICLR 2025 Spotlight), SportMamba (CVPRW 2025), and MM-Tracker (AAAI 2025) are all real, verifiable, peer-reviewed works. In single-object/visual tracking (SOT/VOT) Mamba is likewise established as a backbone (MambaVT, MambaEVT/IEEE TCSVT 2025, Mamba-FETrack V2, MambaLCT/AAAI 2025), and SSM video-understanding backbones (Vision Mamba/ICML 2024, VideoMamba/ECCV 2024) provide temporal-modeling infrastructure. Critically, NO published work was found applying Mamba/SSMs to Referring Multi-Object Tracking (RMOT) or language-conditioned/referring tracking specifically — this is an evidence-absent gap, representing a genuine open niche for FlexHook-CF-style RMOT research. Across all works the consistent motivation is Mamba's linear-time long-sequence modeling versus transformer attention's O(n^2) cost, with quantified efficiency gains (e.g., Vision Mamba 2.8x faster and 86.8% less GPU memory than DeiT at 1248x1248).

## Findings

### 1. Mamba/SSMs are established as learned motion/trajectory-state predictors in multi-object tracking (MOT), used as data-driven replacements for or augmentations of the Kalman filter. Verifiable methods: MambaTrack (ACM MM 2024) with its bi-directional Mamba moTion Predictor (MTP); TrackSSM (arXiv 2024, also Science China Information Sciences) with a Mamba-Block motion encoder + Flow-SSM decoder; and Samba/SambaMOTR (ICLR 2025 Spotlight) synchronizing multiple selective state-spaces to jointly model multiple tracklets.

- **Confidence:** high  |  **Vote:** 3-0 (each constituent claim)

- **Evidence:** MambaTrack (arXiv:2408.09178, ACM MM 2024, DOI 10.1145/3664647.3680944): 'we introduce a Mamba-based motion model named Mamba moTion Predictor (MTP)... takes the spatial-temporal location dynamics of objects as input, captures the motion pattern using a bi-Mamba encoding layer, and predicts the next motion.' Ablation: KF baseline HOTA 45.9 -> +MTP 54.9. TrackSSM (arXiv:2409.00487, also Sci China Info Sci DOI 10.1007/s11432-024-4849-2): 'a unified encoder-decoder motion framework that uses data-dependent state space model to perform temporal motion of trajectories' and 'utilizes a simple Mamba-Block to build a motion encoder for historical trajectories' (real stacked Mamba modules in encoder; custom Flow-SSM in decoder). ByteTrack+TrackSSM = 57.7 HOTA on DanceTrack. Samba (arXiv:2410.01806, ICLR 2025 Spotlight, OpenReview id=OeBY9XqiTz): 'a novel linear-time set-of-sequences model designed to jointly process multiple tracklets by synchronizing the multiple selective state-spaces used to model each tracklet'; 'autoregressively predicts the future track query for each sequence while maintaining synchronized long-term memory representations across tracklets,' addressing long-range dependencies and occlusions. SSM is confined to motion/state prediction; MambaTrack still uses the Hungarian algorithm for data association.

- **Sources:**

  - https://arxiv.org/abs/2408.09178

  - https://arxiv.org/abs/2409.00487

  - https://arxiv.org/abs/2410.01806

  - https://github.com/JackWoo0831/Mamba_Trackers

  - https://github.com/Xavier-Lin/TrackSSM

  - https://sambamotr.github.io


### 2. Additional peer-reviewed MOT works confirm SSMs are used for both motion modeling and association-adjacent pipelines as of 2025: SportMamba (CVPRW 2025) uses a Mamba-attention motion predictor for non-linear player motion plus a hybrid matching metric; MM-Tracker (AAAI 2025) introduces a bidirectional-scanning 'Motion Mamba' module (vertical + horizontal SSM scans) for long-range UAV-platform object motion.

- **Confidence:** high  |  **Vote:** 3-0 (each constituent claim)

- **Evidence:** SportMamba (arXiv:2506.03335, CVPR 2025 CVSports workshop / IEEE Xplore doc 11147465): 'We introduce a mamba-attention mechanism that models non-linear motion by implicitly focusing on relevant embedding dependencies'; a 'mamba-attention motion predictor estimates player positions in subsequent frames,' with downstream data association via a hybrid Re-ID + height-adaptive IoU metric. MM-Tracker (arXiv:2407.10485v3, AAAI 2025): 'We propose the Motion Mamba module, which models object motion by local correlation of detection features and global scan of bi-directional mamba block'; uses V-SSM (vertical) and H-SSM (horizontal) selective scanning to 'predict long-range object motion better' on UAV-MOT datasets. Both are peer-reviewed (CVPRW 2025, AAAI 2025), defeating any 'isolated/preprint-only' concern. Note: both are generic MOT, not RMOT.

- **Sources:**

  - https://arxiv.org/html/2506.03335

  - https://arxiv.org/html/2407.10485v3

  - https://openaccess.thecvf.com/content/CVPR2025W/CVSports/papers/Khanna_SportMamba_Adaptive_Non-Linear_Multi-Object_Tracking_with_State_Space_Models_for_CVPRW_2025_paper.pdf

  - https://ojs.aaai.org/index.php/AAAI/article/view/33019

  - https://github.com/YaoMufeng/MMTracker


### 3. Mamba/SSMs are established as backbones for single-object/visual object tracking (SOT/VOT), not just image classification. Verifiable methods: MambaVT (pure-Mamba RGB-T tracking), MambaEVT (event-stream SOT, IEEE TCSVT 2025), Mamba-FETrack V2 (frame-event RGB-Event VOT, 2025), and MambaLCT (long-term context SSM tracker, AAAI 2025).

- **Confidence:** high  |  **Vote:** 3-0 (each constituent claim)

- **Evidence:** MambaVT (arXiv:2408.07889): 'this work innovatively proposes a pure Mamba-based framework (MambaVT) to fully exploit spatio-temporal contextual modeling for robust visible-thermal tracking' (RGB-T SOT, LasHeR/RGBT234-class benchmarks). MambaEVT (arXiv:2408.10487, IEEE TCSVT 2025, code github.com/Event-AHU/MambaEVT): 'a novel Mamba-based visual tracking framework that adopts the state space model with linear complexity as a backbone network. The search regions and target template are fed into the vision Mamba network for simultaneous feature extraction and interaction'; adds a Memory Mamba dynamic-template module (EventVOT/VisEvent/FE240hz). Mamba-FETrack V2 (arXiv:2506.23783): 'efficient RGB-Event object tracking framework based on the linear-complexity Vision Mamba network' (COESOT/FE108/FELT V2). MambaLCT (arXiv:2412.13615, AAAI 2025): unidirectional Context Mamba module with selective scanning, evaluated only on SOT benchmarks (LaSOT/GOT-10k/TrackingNet/TNL2K/UAV123) with NO language branch — it treats the nominally 'natural-language' TNL2K purely as a visual benchmark.

- **Sources:**

  - https://arxiv.org/pdf/2408.07889

  - https://arxiv.org/abs/2408.10487

  - https://arxiv.org/abs/2506.23783

  - https://arxiv.org/html/2412.13615v1

  - https://github.com/Event-AHU/MambaEVT

  - https://dl.acm.org/doi/10.1609/aaai.v39i5.32528


### 4. Mamba/SSM video-understanding and vision backbones relevant to temporal/trajectory modeling exist and are peer-reviewed: Vision Mamba (Vim, ICML 2024) is a generic vision backbone built purely on bidirectional Mamba blocks instead of self-attention; VideoMamba (ECCV 2024) adapts Mamba to the video domain with a linear-complexity operator for short- and long-term video understanding. Neither addresses tracking.

- **Confidence:** high  |  **Vote:** 3-0 (each constituent claim)

- **Evidence:** Vision Mamba (arXiv:2401.09417, ICML 2024): 'the reliance on self-attention for visual representation learning is not necessary and propose a new generic vision backbone with bidirectional Mamba blocks (Vim)... compresses the visual representation with bidirectional state space models.' Evaluated on still-image tasks (ImageNet/COCO/ADE20k), so temporal-modeling relevance is by family analogy, not by direct demonstration. VideoMamba (arXiv:2403.06977, ECCV 2024, DOI 10.1007/978-3-031-73347-5_14): 'innovatively adapts the Mamba to the video domain'; evaluated on ImageNet, Kinetics-400, Something-Something V2, Breakfast/COIN/LVU, and zero-shot video-text retrieval — confirmed NO tracking/MOT/RMOT/referring-expression/RVOS experiments. Verifier explicitly checked the full text and found 'No mention of tracking, MOT, RMOT, referring expressions, or RVOS.'

- **Sources:**

  - https://arxiv.org/abs/2401.09417

  - https://arxiv.org/abs/2403.06977

  - https://github.com/hustvl/Vim


### 5. Across MOT, SOT, and video backbones, the consistent reported tradeoff motivating Mamba over transformer attention is linear-time long-sequence modeling versus attention's O(n^2) quadratic cost, with quantified efficiency gains for long visual sequences.

- **Confidence:** high  |  **Vote:** 3-0 (each constituent claim)

- **Evidence:** Vision Mamba (arXiv:2401.09417): 'Vim is 2.8x faster than DeiT and saves 86.8% GPU memory when performing batch inference to extract features on images with a resolution of 1248x1248' (=6084 tokens). VideoMamba (arXiv:2403.06977): 'Its linear-complexity operator enables efficient long-term modeling, which is crucial for high-resolution long video understanding'; 'Compared to transformers based on quadratic-complexity attention, Mamba excels at processing long sequences with linear complexity'; empirically 6x faster than TimeSformer and 40x less GPU memory at 64 frames. MM-Tracker (arXiv:2407.10485v3): Mamba 'can realize linear-time global attention calculation while also performing parallel training well,' vs Transformer 'O(n2)... difficult to reach real-time tracking.' SportMamba (arXiv:2506.03335): 'State-Space Models (SSMs) such as Mamba have emerged as an alternative that demonstrates strong sequence modeling capabilities in linear time.' MambaVT (arXiv:2408.07889): attention's 'intrinsic high quadratic complexity' motivates a Mamba backbone 'renowned for its impressive long sequence modeling capabilities and linear computational complexity.' Mamba-FETrack V2 (arXiv:2506.23783): ViT 'high computational complexity... substantial computational overhead' vs 'linear-complexity Vision Mamba.' Foundational claim grounded in Mamba (Gu & Dao, arXiv:2312.00752, 'Linear-Time Sequence Modeling with Selective State Spaces').

- **Sources:**

  - https://arxiv.org/abs/2401.09417

  - https://arxiv.org/abs/2403.06977

  - https://arxiv.org/html/2407.10485v3

  - https://arxiv.org/html/2506.03335

  - https://arxiv.org/pdf/2408.07889

  - https://arxiv.org/abs/2506.23783

  - https://arxiv.org/abs/2312.00752


### 6. NO published work was found applying Mamba/SSMs to Referring Multi-Object Tracking (RMOT), language-conditioned/referring tracking, or referring video object segmentation (RVOS). This is an evidence-absent gap, not a confirmed presence.

- **Confidence:** medium  |  **Vote:** N/A (absence finding inferred across all verified claims)

- **Evidence:** Every verified Mamba/SSM tracking work is scoped to either generic MOT (MambaTrack, TrackSSM, Samba/SambaMOTR, SportMamba, MM-Tracker) or SOT/VOT (MambaVT, MambaEVT, Mamba-FETrack V2, MambaLCT) — none take a language/referring expression as input or score trajectory<->expression matches. Verifiers explicitly noted the language gap: VideoMamba 'provides no evidence for Mamba/SSM use in RMOT/referring tracking' (no tracking/RMOT/RVOS/referring-expression experiments); MambaLCT 'has no text/language input branch... does not address multi-object, referring (RMOT), or language-conditioned tracking,' treating the nominally natural-language TNL2K benchmark purely visually. Multiple verifiers flagged that their respective papers are MOT/SOT and make 'no RMOT assertion.' This is an absence inferred from the scope of all surviving claims rather than a positively verified negative search, so confidence is medium rather than high.

- **Sources:**

  - https://arxiv.org/abs/2403.06977

  - https://arxiv.org/html/2412.13615v1

  - https://arxiv.org/abs/2410.01806

  - https://arxiv.org/html/2506.03335


## All cited sources

- https://arxiv.org/abs/2408.09178
- https://arxiv.org/abs/2409.00487
- https://arxiv.org/abs/2410.01806
- https://github.com/JackWoo0831/Mamba_Trackers
- https://github.com/Xavier-Lin/TrackSSM
- https://sambamotr.github.io
- https://arxiv.org/html/2506.03335
- https://arxiv.org/html/2407.10485v3
- https://openaccess.thecvf.com/content/CVPR2025W/CVSports/papers/Khanna_SportMamba_Adaptive_Non-Linear_Multi-Object_Tracking_with_State_Space_Models_for_CVPRW_2025_paper.pdf
- https://ojs.aaai.org/index.php/AAAI/article/view/33019
- https://github.com/YaoMufeng/MMTracker
- https://arxiv.org/pdf/2408.07889
- https://arxiv.org/abs/2408.10487
- https://arxiv.org/abs/2506.23783
- https://arxiv.org/html/2412.13615v1
- https://github.com/Event-AHU/MambaEVT
- https://dl.acm.org/doi/10.1609/aaai.v39i5.32528
- https://arxiv.org/abs/2401.09417
- https://arxiv.org/abs/2403.06977
- https://github.com/hustvl/Vim
- https://arxiv.org/abs/2312.00752

## Application note (FlexHook-CF)

Every Mamba-MOT work above uses Mamba as a **stage-1 motion/state predictor** (next-position for data association). FlexHook's tracker is off-the-shelf; FlexHook is the **stage-2 referring head**. The open, unfilled idea: a Mamba **trajectory-motion encoder inside the referring head** whose output feature is matched against motion-typed expressions ("turning", "parked", "same direction") — inheriting MambaTrack's evidence that Mamba captures motion, while filling the RMOT/SSM gap (Finding 6). Gates before building: (a) confirm motion expressions are FlexHook's weak subset via per-attribute HOTA breakdown; (b) bound payoff by motion-expression fraction; (c) the linear-time edge only materializes on long/full trajectories, not the 8-frame sampled window; (d) `mamba-ssm` CUDA-kernel install friction on the cu124/cu130 env.

