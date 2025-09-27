graph TD
    %% Main Entry Points
    Start([Start: main.py]) --> Mode{Execution Mode?}
    
    %% Training Branch
    Mode -->|--mode train| Train[Train Pipeline]
    Train --> DataLoad[BD_S8_RealDataset<br/>Load TIFF Images]
    DataLoad --> Preprocess[Image Preprocessing<br/>• Resize to 256x256<br/>• Normalize 16-bit TIFFs<br/>• Data Augmentation]
    Preprocess --> Model[BD_S8_Model<br/>Multi-task Architecture]
    
    Model --> SegBranch[Segmentation Branch<br/>UnetPlusPlus]
    Model --> ClassBranch[Classification Branch<br/>EfficientNet-B4]
    
    ClassBranch --> MTHeads[Multi-Task Heads<br/>• Extraction Method<br/>• Viability<br/>• Cell Type]
    
    SegBranch --> Loss1[Dice + Focal Loss]
    MTHeads --> Loss2[Cross Entropy Loss]
    
    Loss1 --> TotalLoss[Combined Loss]
    Loss2 --> TotalLoss
    TotalLoss --> Optimize[AdamW Optimizer<br/>Cosine Annealing]
    Optimize --> Checkpoint[Save Checkpoint<br/>models/*.ckpt]
    
    %% Inference Branch
    Mode -->|--mode inference| Inference[Inference Pipeline]
    Inference --> LoadModel[Load Best Model]
    LoadModel --> CalibData[Calibration Dataset<br/>60-70% of data]
    LoadModel --> TestData[Test Dataset<br/>70-100% of data]
    
    CalibData --> CalibProcess[Calibration<br/>• AML column selection<br/>• Threshold optimization<br/>• Temperature scaling]
    
    CalibProcess --> TestProcess[Large-scale Testing<br/>16,516 samples/class]
    TestData --> TestProcess
    
    TestProcess --> Metrics[Performance Metrics<br/>• Accuracy/F1/MCC<br/>• ROC-AUC/PR-AUC<br/>• Confusion Matrix]
    
    Metrics --> SaveResults[Save Results<br/>logs/test_results.json]
    
    %% Clinical Analysis Branch
    Mode -->|Clinical Analysis| Clinical[Clinical Pipeline v9.0]
    Clinical --> Discovery[Smart Specimen Discovery]
    
    Discovery --> ImageIdx[Image Indexing<br/>Filter: .tif/.tiff<br/>Skip: thumbnails/masks]
    Discovery --> FCSIdx[FCS Indexing<br/>Filter: .fcs files<br/>Skip: controls/comp]
    
    ImageIdx --> Matching[Specimen Matching<br/>• FH_XXXX_X patterns<br/>• Multi-modal pairing]
    FCSIdx --> Matching
    
    Matching --> FeatExt[Feature Extraction]
    
    FeatExt --> ImgFeat[Image Features<br/>• Intensity stats<br/>• Texture (Sobel)<br/>• Morphology<br/>• Histogram]
    FeatExt --> FCSFeat[FCS Features<br/>• CD34/CD117 markers<br/>• Blast % calculation<br/>• Flow cytometry data]
    
    ImgFeat --> Clustering[Advanced Clustering]
    FCSFeat --> Clustering
    
    Clustering --> OutlierRem[Outlier Removal<br/>Isolation Forest]
    OutlierRem --> KMeans[K-Means Clustering<br/>Optimal k selection]
    
    KMeans --> DimRed[Dimensionality Reduction<br/>• PCA (50 components)<br/>• t-SNE<br/>• UMAP]
    
    DimRed --> Viz[Visualizations<br/>• Cluster plots<br/>• Purity analysis<br/>• Blast % distribution]
    
    Viz --> Exemplars[Cluster Exemplars<br/>Export montages]
    
    %% Specialized Analyses
    Clustering --> Special[Specialized Analyses]
    Special --> FH7087[FH_7087_2<br/>BD S8 vs Quanteon]
    Special --> FH8445[FH_8445_2<br/>DMSO vs Venetoclax]
    
    %% Output
    Viz --> Report[Generate Reports<br/>• JSON summary<br/>• Markdown report<br/>• CSV exports]
    Exemplars --> Report
    FH7087 --> Report
    FH8445 --> Report
    
    Report --> Output([Output Directory<br/>outputs/clinical_v9/])
    SaveResults --> Output
    Checkpoint --> Output
    
    %% Data Sources
    DataSource1[(AML Dataset<br/>555k TIFF files)] -.-> DataLoad
    DataSource2[(Healthy BM Dataset<br/>TIFF files)] -.-> DataLoad
    DataSource3[(FCS Files<br/>Flow Cytometry)] -.-> FCSIdx
    
    %% Configuration
    Config[/config.yaml/] -.-> Train
    Config -.-> Inference
    Config -.-> Clinical
    
    style Start fill:#e1f5e1
    style Output fill:#ffe1e1
    style Model fill:#fff3cd
    style Clustering fill:#d4edff
    style Report fill:#f0e6ff

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