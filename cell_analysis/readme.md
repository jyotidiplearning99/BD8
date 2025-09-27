Pipeline Architecture

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           BD S8 CELL ANALYSIS PIPELINE v9.0                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────────┐
                    ▼                      ▼                          ▼
         ╔═══════════════════╗  ╔═══════════════════╗  ╔═══════════════════════╗
         ║  TRAINING MODE    ║  ║  INFERENCE MODE   ║  ║  CLINICAL ANALYSIS   ║
         ╚═══════════════════╝  ╚═══════════════════╝  ╚═══════════════════════╝
                    │                      │                          │
    ┌───────────────┴──────┐   ┌──────────┴──────────┐   ┌──────────┴──────────┐
    │ 555k TIFF Dataset    │   │ Calibration Phase   │   │ Specimen Discovery  │
    │ • AML samples        │   │ • 60-70% data       │   │ • Smart matching    │
    │ • Healthy BM         │   │ • AML column detect │   │ • FCS+TIFF pairing  │
    └───────────┬──────────┘   └──────────┬──────────┘   └──────────┬──────────┘
                │                          │                          │
    ┌───────────▼──────────┐   ┌──────────▼──────────┐   ┌──────────▼──────────┐
    │ Multi-task Model     │   │ Testing Phase       │   │ Feature Extraction  │
    │ • Segmentation       │   │ • 70-100% data      │   │ • Image features    │
    │ • Classification     │   │ • 16,516 samples    │   │ • Flow cytometry    │
    └───────────┬──────────┘   └──────────┬──────────┘   └──────────┬──────────┘
                │                          │                          │
    ┌───────────▼──────────┐   ┌──────────▼──────────┐   ┌──────────▼──────────┐
    │ PyTorch Lightning    │   │ Metrics & Results   │   │ Advanced Analysis   │
    │ • AdamW optimizer    │   │ • ROC-AUC: 0.XXX    │   │ • K-means clustering│
    │ • Mixed precision    │   │ • Accuracy: XX%     │   │ • PCA/t-SNE/UMAP    │
    └───────────┬──────────┘   └──────────┬──────────┘   └──────────┬──────────┘
                │                          │                          │
         ┌──────▼──────┐           ┌──────▼──────┐           ┌──────▼──────┐
         │ models/*.ckpt│           │logs/results│           │outputs/v9/  │
         └──────────────┘           └─────────────┘           └─────────────┘

Module Dependency Matrix

┌─────────────────┬────────┬──────────┬─────────┬──────────┬──────────┬─────────┐
│ Module          │ main.py│ train.py │models.py│inference │ clinical │ loader  │
├─────────────────┼────────┼──────────┼─────────┼──────────┼──────────┼─────────┤
│ Entry Point     │   ★    │          │         │          │          │         │
│ Dataset Loading │        │    ✓     │         │    ✓     │    ✓     │    ★    │
│ Model Training  │        │    ★     │    ✓    │          │          │         │
│ Model Inference │        │          │    ✓    │    ★     │          │         │
│ FCS Processing  │        │          │         │          │    ★     │         │
│ Image Analysis  │        │    ✓     │         │    ✓     │    ★     │    ✓    │
│ Clustering      │        │          │         │          │    ★     │         │
│ Visualization   │        │          │         │          │    ★     │         │
└─────────────────┴────────┴──────────┴─────────┴──────────┴──────────┴─────────┘
Legend: ★ = Primary responsibility, ✓ = Uses functionality

Data Processing Pipeline Stages

STAGE_1_INPUT:
  ├─ 555,053 TIFF images (16-bit, various dimensions)
  ├─ FCS flow cytometry files
  └─ Excel metadata sheets

STAGE_2_PREPROCESSING:
  ├─ Image Processing:
  │   ├─ Resize: 256×256 or 512×512
  │   ├─ Normalize: 16-bit → [0,1]
  │   └─ Augment: Rotation, Flip, Brightness
  ├─ FCS Processing:
  │   ├─ Filter: Remove controls/compensation
  │   ├─ Extract: CD34/CD117 markers
  │   └─ Calculate: Blast percentages

STAGE_3_FEATURE_EXTRACTION:
  ├─ Image Features (100-dim):
  │   ├─ Intensity: mean, std, percentiles
  │   ├─ Texture: Sobel gradients
  │   ├─ Morphology: contour statistics
  │   └─ Histogram: 32 bins
  ├─ FCS Features (200-dim):
  │   ├─ Channel statistics
  │   ├─ Arcsinh transformation
  │   └─ Marker expressions

STAGE_4_ANALYSIS:
  ├─ Training Path:
  │   ├─ Model: EfficientNet-B4 + UNet++
  │   ├─ Loss: Dice + Focal + CrossEntropy
  │   └─ Output: Checkpoint files
  ├─ Inference Path:
  │   ├─ Calibration: Threshold optimization
  │   ├─ Testing: Large-scale evaluation
  │   └─ Output: Metrics JSON
  ├─ Clinical Path:
  │   ├─ Clustering: K-means (5-15 clusters)
  │   ├─ Reduction: PCA → t-SNE/UMAP
  │   └─ Output: Reports & visualizations

STAGE_5_OUTPUT:
  ├─ models/*.ckpt (trained models)
  ├─ logs/test_results.json (metrics)
  ├─ outputs/clinical_v9/ (full analysis)
  └─ cache/*.npz (feature cache)

 Execution Sequence Timeline

 Time ──────────────────────────────────────────────────────────────────────────►

[TRAINING MODE]
00:00 │ Load config.yaml
00:01 │ Initialize RealTIFFDataset
00:05 │ Load 30k train + 5k val samples
00:10 │ Create BD_S8_Model (EfficientNet-B4 + UNet++)
00:11 │ Setup PyTorch Lightning trainer
00:15 │ Begin training epochs
02:00 │ Early stopping triggered
02:05 │ Save best checkpoint
02:06 │ Complete

[INFERENCE MODE]
00:00 │ Load last.ckpt
00:01 │ Create calibration dataset (60-70% range)
00:05 │ Extract calibration features
00:10 │ Auto-detect AML column via AUC
00:11 │ Optimize threshold (accuracy/youden/f1)
00:12 │ Optional: Temperature scaling
00:15 │ Create test dataset (70-100% range)
00:20 │ Extract test features (16,516/class)
00:40 │ Apply threshold & calculate metrics
00:42 │ Save results to JSON
00:43 │ Complete

[CLINICAL ANALYSIS MODE]
00:00 │ Initialize pipeline v9.0
00:01 │ Phase 1: Specimen discovery
00:05 │   ├─ Index TIFF files (skip thumbnails)
00:10 │   ├─ Index FCS files (skip controls)
00:15 │   └─ Smart FH_XXXX_X matching
00:16 │ Phase 2: Feature extraction
00:20 │   ├─ Process 200 images/sample
00:40 │   ├─ Extract FCS features
00:45 │   └─ Calculate blast percentages
00:46 │ Phase 3: Advanced analysis
00:50 │   ├─ Outlier removal (3%)
00:51 │   ├─ PCA (50 components)
00:55 │   ├─ K-means clustering
01:00 │   ├─ t-SNE projection
01:05 │   └─ UMAP embedding
01:06 │ Phase 4: Visualization
01:10 │   └─ Generate 6-panel summary
01:11 │ Phase 5: Reports
01:15 │   ├─ Export JSON/CSV/Markdown
01:16 │   ├─ FH_7087_2 analysis (BD S8 vs Quanteon)
01:17 │   └─ FH_8445_2 analysis (DMSO vs Venetoclax)
01:18 │ Complete


Component Architecture

╔══════════════════════════════════════════════════════════════╗
║                     MAIN CONTROLLER (main.py)                ║
╟──────────────────────────────────────────────────────────────╢
║  • Argument parsing (--mode train/inference/clinical)        ║
║  • Configuration loading (config.yaml)                       ║
║  • Module orchestration                                      ║
╚══════════════════════════════════════════════════════════════╝
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
╔═══════════════╗    ╔═══════════════╗    ╔═══════════════╗
║  DATA LAYER   ║    ║  MODEL LAYER  ║    ║ ANALYSIS LAYER║
╟───────────────╢    ╟───────────────╢    ╟───────────────╢
║ RealTIFFDataset║    ║ BD_S8_Model   ║    ║ ImageAnalyzer ║
║ • Stratified   ║    ║ • EfficientNet║    ║ • Feature ext ║
║   splits       ║    ║ • UNet++      ║    ║ • Quality QC  ║
║ • Augmentation ║    ║ • Multi-task  ║    ╟───────────────╢
╟───────────────╢    ╟───────────────╢    ║ FCSAnalyzer   ║
║ BalancedTest   ║    ║ Lightning     ║    ║ • Blast calc  ║
║ • Percentage   ║    ║ • Training    ║    ║ • Marker ext  ║
║   ranges       ║    ║ • Checkpoints ║    ╟───────────────╢
║ • Disjoint sets║    ║ • Early stop  ║    ║ Clustering    ║
╚═══════════════╝    ╚═══════════════╝    ║ • K-means     ║
                                           ║ • PCA/t-SNE   ║
                                           ╚═══════════════╝

project/
│
├── src/                    [Core modules]
│   ├── data_loader.py      → TIFF loading, augmentation
│   ├── models.py           → Neural network architecture
│   ├── train.py            → PyTorch Lightning training
│   └── inference.py        → Two-phase evaluation
│
├── outputs/                 [Results storage]
│   ├── models/             → Checkpoints (*.ckpt)
│   ├── logs/               → Training logs, metrics
│   └── clinical_v9/        → Clinical analysis
│       ├── cache/          → Feature cache (*.npz)
│       ├── reports/        → JSON/CSV/MD reports
│       ├── visualizations/ → Plots and figures
│       └── cluster_exemplars/ → Image montages
│
├── data/                    [Input data - not in repo]
│   ├── AML/                → Cancer samples
│   ├── Healthy BM/         → Control samples
│   └── FCS/                → Flow cytometry
│
├── main.py                  [Entry point]
├── AML_Healthy_v9.0_final.py [Clinical pipeline]
├── config.yaml              [Configuration]
├── requirements.txt         [Dependencies]
└── train_real.sbatch        [SLURM job script]



Pipeline Components Overview
1. Data Loading Module (src/data_loader.py)
Handles 555k+ TIFF images from AML and Healthy BM datasets
Supports 16-bit TIFF normalization
Stratified train/val/test splits (70%/15%/15%)
Real-time data augmentation with Albumentations
2. Model Architecture (src/models.py)
Multi-task learning with shared encoder
Segmentation: UNet++ architecture
Classification: EfficientNet-B4 backbone
Three task heads: extraction method, viability, cell type
3. Training Pipeline (src/train.py)
PyTorch Lightning integration
Mixed precision training (16-bit)
Early stopping & model checkpointing
Support for both synthetic and real TIFF data
4. Inference Module (src/inference.py)
Two-phase evaluation: calibration + testing
Automatic AML column detection via AUC
Multiple threshold selection methods
Optional temperature scaling
Handles 33k+ samples efficiently
5. Clinical Analysis (AML_Healthy_v9.0_final.py)
Smart specimen matching between TIFF and FCS files
Multi-modal analysis combining imaging and flow cytometry
Blast percentage calculation from CD34/CD117 markers
Unsupervised clustering with outlier detection
Specialized analyses for specific specimens (FH_7087_2, FH_8445_2)
Key Features
Large-scale processing: Handles 555k+ medical images
Multi-modal fusion: Combines microscopy and flow cytometry
Robust preprocessing: Handles 8/16-bit TIFFs, multiple channels
Production-ready: Caching, error handling, audit trails
Clinical insights: Blast percentage, instrument comparison, drug response
Execution Modes
bash


# Training
python main.py --mode train

# Inference  
python main.py --mode inference

# Clinical Analysis
python AML_Healthy_v9.0_final.py