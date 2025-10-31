# PostOpPainGuard™

### Predicting and Optimizing Post-Operative Analgesia in Rabbits Using Synthetic Data and Artificial Intelligence

---

## Phase 1 — Project Setup

### 1. Project Overview & Rationale

#### 1.1 Research Problem

**Concept:** Develop *PostOpGuard*, a predictive framework that identifies **rabbits at risk of requiring additional analgesics post-surgery** across multiple procedures.
**Objective:** Compute an **Analgesia Need Score (0–100)** to guide **timely clinical intervention within 24 hours post-operatively**.
**Scope:** Employs **demographic, surgical, anesthetic, and pre-operative clinical features only**, without reliance on imaging or telemetry.
**Ethical Compliance:** Fully aligned with **AAALAC**, **IACUC**, and the **3Rs (Replacement, Reduction, Refinement)**, emphasizing refinement, non-invasive monitoring, and actionable welfare outcomes.

---

#### 1.2 Context & Background

Recent work using deep learning on video data achieved ≈ 87 % accuracy in rabbit post-operative pain detection ([Feighelstein et al., 2023](https://www.nature.com/articles/s41598-023-41774-2)).
While valuable, such methods depend on high-quality imaging and tightly controlled environments—factors that constrain scalability and reproducibility.

##### Advantages of PostOpGuard

* **Ethical & 3Rs-Aligned:** Uses **fully synthetic structured data**, removing any live-animal data collection.
* **Structured Clinical Modeling:** Relies on routinely captured demographic and anesthetic variables, integrating seamlessly with laboratory workflows.
* **Interpretability & Transparency:** Every prediction is traceable to physiological and procedural parameters.
* **Scalability & Adaptability:** Generalizes across surgery types and species without additional animal use.
* **Complementary Approach:** Extends beyond video-based systems by emphasizing explainable AI and structured data integration.

##### Synthetic Data Simulation

* Models a **laboratory of 5 000 rabbits**, **5 surgery types**, and **10 veterinarians (VetID)**
* Introduces **inter-observer variability**, missing data, and rare high-pain events
* Maintains plausible physiological and procedural distributions
* Reproducible through controlled randomness and fixed seeds

---

#### 1.3 Importance of Multi-Surgery Modeling

Pain perception and analgesic demand vary by procedure. Incorporating diverse surgeries (Ovariohysterectomy, Castration, Bone Defect, Catheterization, Ocular Implant) enhances:

* Model generalization
* Procedure-specific validation
* Capture of inter-procedure variability

---

#### 1.4 Ethical & Regulatory Framework

* All data generation adheres to **animal-welfare and research-ethics standards**.
* **IACUC Alignment:** Model predictions augment but never replace professional veterinary judgment.
* **3Rs Compliance:**

  * *Reduction* — minimizes exploratory animal use
  * *Refinement* — improves analgesic precision and welfare outcomes
  * *Replacement* — demonstrates feasibility of synthetic data for proof-of-concept research
* Model outputs remain **auditable**, with interpretable feature contributions.

---

#### 1.5 Species Context & Adaptability

**Species Adaptability:** Although initially trained on rabbits, the framework is conceptually transferable to rats, mice, or zebrafish with limited retraining.

**Why Rabbits?**

* Widely used in ocular, orthopedic, and pharmacological research
* Established behavioral and grimace-based pain metrics
* Manageable body size enabling precise dosing and reproducible procedures
* Encompasses a full spectrum of post-surgical pain intensity

**Key Takeaway:**
Rabbits constitute a scientifically and ethically balanced model for developing interpretable, high-fidelity AI systems in post-operative pain prediction.

---

### 2. Library Dependencies

| Category                     | Libraries                                                                                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Core**                     | `numpy`, `pandas`                                                                                                                              |
| **Visualization**            | `matplotlib`, `seaborn`                                                                                                                        |
| **Modeling**                 | `scikit-learn` (LogisticRegression, LassoCV, RandomForest, GradientBoosting, HistGradientBoosting), `xgboost`, `lightgbm`, `mord (LogisticAT)` |
| **Statistics & Diagnostics** | `statsmodels (VIF)`, `scikit-learn (StandardScaler, LabelEncoder)`                                                                             |
| **Explainability**           | `shap`                                                                                                                                         |
| **Utilities**                | `joblib`, `os`, `warnings`, `missingno`                                                                                                        |

---

## Phase 2 — Dataset Generation

### 1. Synthetic Data Creation

**Objectives**

* Generate a **comprehensive synthetic dataset** reflecting a multi-veterinarian laboratory environment.
* Encode procedure-specific distributions for age, weight, and pain expectation.
* Incorporate pre-operative, intra-operative, and observer-related variability.
* Produce both **continuous (AnalgesiaNeedScore)** and **categorical (AnalgesiaNeedCategory)** labels.
* Preserve rare, high-pain events for sensitivity testing.

#### Key Features and Veterinary Rationale

| Category                      | Features                                                                                    | Purpose                             |
| ----------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------------- |
| **Demographics & Physiology** | `AgeWeeks`, `WeightKg`, `Sex`, `Strain`                                                     | Capture biological variability      |
| **Surgical Parameters**       | `SurgeryType`, `SurgeryPainBoost`, `DurationPerKg`                                          | Encode procedural intensity         |
| **Pre-Operative Indicators**  | `PreOpGrimace`, `PreOpBehaviorScore`, `PreOpBiomarker`, `PreOpRiskScore`                    | Quantify baseline pain              |
| **Interventions**             | `LocalBlock`, `IntraOpAnalgesicMgPerKg`                                                     | Represent intra-operative analgesia |
| **Observer Effects**          | `VetID`, `VetBias`                                                                          | Simulate inter-observer variability |
| **Outcomes**                  | `AnalgesiaNeedScore`, `AnalgesiaNeedCategory`, `FirstPostOpGrimace`, `RescueAnalgesiaGiven` | Define pain and treatment endpoints |

**Veterinary Rationale:**
Mirrors realistic laboratory diversity and clinically interpretable cause-effect patterns while retaining ethically significant high-pain edge cases.

**Machine-Learning Rationale:**
Comprehensive features reduce underfitting, balance class representation, and introduce realistic noise via observer bias, thereby enhancing generalization.

---

#### Limitations & Considerations

* **Surgical Modeling:** Excludes explicit complications; pain duration approximated with small stochastic variance.
* **Biological Factors:** Minimal strain effect to maintain generalizability; simplified biomarker set for methodological clarity.
* **Scope:** Designed as a proof-of-concept using synthetic data only.

**Outcome:**
A biologically plausible synthetic dataset suitable for feature engineering, model development, and interpretability analysis.

**Key Takeaway:**
This dataset combines biological realism with computational control, enabling reproducible, ethically defensible research in predictive veterinary analytics.

---

### 2. Data Quality Checks & Imputation

**Objectives**

1. Validate physiologic and procedural ranges (`WeightKg`: 1–5 kg; `DurationMin`: 20–180 min; `IntraOpAnalgesicMgPerKg`: 0–5 mg/kg).
2. Detect and clip outliers.
3. Visualize missingness via matrix and bar plots.
4. Impute missing values:

   * *Numeric:* Median by `SurgeryType`
   * *Categorical:* Mode for `LocalBlock`, `Housing`, `AnesthesiaProtocol`
5. Derive composite features (`DurationPerKg`, `PreOpRiskScore`, `SurgeryComplexity`).
6. Map latent pain to continuous and categorical outcomes.
7. Simulate `FirstPostOpGrimace` and `RescueAnalgesiaGiven`.
8. Oversample rare high-pain cases to preserve class balance.

**Veterinary Perspective:** Maintains biological plausibility and realistic pain distribution across procedures.
**Machine-Learning Perspective:** Prevents data artifacts and supports learning stability for low-frequency but high-impact outcomes.

---

### 3. Exploratory Data Analysis (EDA)

**Objectives**

1. Summarize descriptive statistics per `SurgeryType`.
2. Visualize distributions of `AnalgesiaNeedScore` and `RescueAnalgesiaGiven`.
3. Evaluate variability using boxplots and histograms.
4. Quantify correlations and feature–target interactions.
5. Validate latent pain patterns through pairwise visualization.

**Veterinary Rationale:** Ensures the dataset captures clinically consistent pain gradients and expected postoperative trends.
**ML Rationale:** Confirms structural soundness and informs subsequent feature engineering and model validation.

---

## Phase 3 — Feature Preparation

### 1. Derived Feature Engineering

#### Objectives

* Generate **biologically meaningful features** for postoperative analgesia prediction.
* Incorporate **interaction terms** that capture nonlinear physiological relationships and reduce underfitting.

#### Derived Features

| Feature                 | Description                                                                                          | Rationale                                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **`DurationPerKg`**     | Duration (minutes) normalized by weight (kg)                                                         | Controls for body-size variability; correlates with postoperative pain magnitude. |
| **`AnalgesicScore`**    | Log-transformed and standardized `IntraOpAnalgesicMgPerKg`                                           | Captures diminishing marginal analgesic efficacy; enhances numerical stability.   |
| **`PreOpRiskScore`**    | Weighted composite of `PreOpGrimace` and `PreOpBiomarker`                                            | Quantifies baseline pain susceptibility prior to surgery.                         |
| **`SurgeryComplexity`** | Binary indicator: Minor vs. Major procedures (Ovariohysterectomy, BoneDefect, OcularImplant = Major) | Reflects procedure-induced pain potential.                                        |
| **`VetBiasFlag`**       | Normalized residual of `PreOpGrimace` per `ObserverID` (clipped to -2 to +2)                         | Corrects inter-observer variability in pain assessment.                           |

#### Interaction Features

* `DurationPerKg × PreOpGrimace` — captures the synergy between surgical duration and preoperative pain indicators.
* `PreOpRiskScore × SurgeryComplexity` — reflects how baseline pain interacts with procedural invasiveness.
* `DurationPerKg × AnalgesicScore` — models analgesic effectiveness relative to surgical intensity.
* `LocalBlock × PreOpGrimace` — examines modulation of baseline pain by local anesthesia.

#### Veterinary Perspective

* Weight-adjusted metrics normalize pain stimuli across individuals.
* Preoperative indicators serve as early pain-risk predictors.
* Observer bias correction supports cross-veterinarian consistency.
* Interaction features mimic realistic biological dependencies.

#### Machine Learning Perspective

* Handcrafted variables enrich model expressiveness and reduce underfitting.
* Interaction terms capture nonlinear physiological dynamics.
* Normalization ensures stable optimization across model architectures.

---

### 2. Dataset Partitioning

#### Objectives

1. Split the dataset into **training (80%)** and **testing (20%)** sets.
2. **Stratify** by `AnalgesiaNeedCategory` to preserve class balance.
3. Optionally apply **surgery- or time-based stratification** for longitudinal analyses.

#### Veterinary Perspective

* Ensures balanced representation of pain categories (Low, Medium, High).
* Preserves biological heterogeneity across procedures.
* Avoids over-stratification that may distort rare or clinically relevant cases.

#### Machine Learning Perspective

* Mitigates class imbalance bias.
* Enhances model generalization across pain levels.
* Supports reproducible evaluation through stable partitioning protocols.

---

### 3. Feature Validation

#### Objectives

1. Assess **feature–target correlations** with `AnalgesiaNeedScore`.
2. Detect and manage **multicollinearity** (e.g., using Variance Inflation Factor).
3. Retain **clinically significant variables** even when correlated.

#### Veterinary Perspective

* Prioritize clinically interpretable features (`PreOpGrimace`, `LocalBlock`, `DurationPerKg`).
* Prevent removal of variables tied to genuine physiological processes.
* Maintain transparency for translational and regulatory review.

#### Machine Learning Perspective

* Prevent unstable coefficient estimates in linear and ordinal models.
* Remove redundancy while preserving biological signal.
* Retain interaction terms that model complex physiological behavior.

> **Key Principle:**
> Balance **biological interpretability** with **statistical discipline** — eliminate redundancy, never clinical relevance.

---

### 4. Feature Selection and Dimensionality Control

#### Objectives

1. Eliminate **highly redundant features** (correlation > 0.95).
2. Use **tree-based importance** metrics to drop features contributing <1%.
3. Retain **biologically and interaction-rich variables** essential for interpretability.
4. Apply **Lasso regularization** for larger biomarker subsets (>10 variables).

#### Veterinary Perspective

* Maintain clinical transparency by preserving pain-relevant predictors.
* Retain interaction features critical for rare or high-pain scenarios.
* Safeguard biological diversity within the model’s explanatory structure.

#### Machine Learning Perspective

* Prevent overfitting through principled, minimal pruning.
* Maintain a strong signal-to-noise ratio.
* Apply regularization to ensure numerical stability and compliance readiness.

> **Key Principle:**
> **Prune with precision.** Remove only redundancy — preserve every feature that adds biological or predictive value.

---

## Phase 4 — Modeling Framework

### 1. Baseline Analgesia Indicator (Rule-Based)

#### Objectives

1. Develop a **transparent, interpretable baseline analgesia score** derived from domain expertise.
2. Capture **strong biological signals** prior to ML modeling.
3. Integrate surgical, anesthetic, and physiological factors — including `Duration`, `LocalBlock`, `PreOpGrimace`, `PreOpBiomarker`, and `Weight`.
4. Introduce **controlled biological noise (±3–7)** to emulate natural variability.

#### Example Rules

* **BoneDefect:** `DurationMin > 60` and `LocalBlock == 0` → High pain, amplified by elevated `PreOpGrimace` and `PreOpBiomarker`.
* **Ovariohysterectomy:** `DurationMin > 45` and `PreOpGrimace > 3` → Medium to High pain.
* **Castration / Catheter:** Minor surgeries → Low pain unless duration or grimace elevated.
* **OcularImplant:** Medium pain baseline, elevated when `DurationMin > 50` or `PreOpGrimace > 3`.
* **Weight Adjustment:** +5 points if `WeightKg > 3.5` (heavier rabbits exhibit higher tissue strain).

**Outputs:**

* `BaselineAnalgesiaNeedScore` (0–100)
* `BaselineAnalgesiaNeedCategory` (Low / Medium / High)

#### Veterinary Rationale

* Encodes **established pain-response patterns** from rabbit literature.
* Captures **baseline risk** through clinically interpretable variables.
* Mirrors **veterinary decision heuristics** used for analgesic intervention.
* Provides a **reference benchmark** for later ML performance evaluation.

#### Machine Learning Rationale

* Serves as a **domain-informed baseline** to calibrate model learning.
* Captures nonlinear biological relationships before statistical modeling.
* Establishes a transparent **pre-training signal** for performance comparison.

---

### 2. Machine Learning Models

#### Objectives

1. Deploy multiple modeling algorithms to exploit complementary strengths:

   * **Ordinal Logistic Regression (LogisticAT)** for interpretable, ordered prediction.
   * **Random Forest / Gradient Boosting / XGBoost** for flexible nonlinear regression.
2. Enable **hybrid or per-surgery submodels** for rare or heterogeneous surgical datasets.
3. Integrate **interaction and derived features** for enhanced realism.

#### Veterinary Rationale

* Models **procedure-specific analgesic responses** and physiological variation.
* Captures nonlinear dependencies among grimace, duration, and analgesic use.
* Retains interpretability for **clinical validation and ethical review**.

#### Machine Learning Rationale

* Ordinal models enforce **monotonic pain order consistency**.
* Ensemble methods model **nonlinear, non-additive effects** without excessive preprocessing.
* Hybridization strengthens performance for **low-sample, high-variance subsets**.
* Parameter control ensures **biological plausibility and model stability**.

---

### 3. Overfitting and Underfitting Controls

#### Objectives

1. Apply **grouped cross-validation** by `RabbitID` and `SurgeryType` to prevent leakage.
2. Constrain **tree depth** and **minimum leaf size** to stabilize variance.
3. Use **early stopping** for boosting algorithms.
4. Maintain interaction features to avoid underfitting.
5. **Oversample High-pain cases** to balance category representation.

#### Veterinary Rationale

* Upholds **intra-surgery consistency** and biological realism.
* Enhances model sensitivity to **rare, high-severity outcomes**.
* Prevents overinterpretation of statistical artifacts as clinical patterns.

#### Machine Learning Rationale

* Grouped CV enables **honest generalization** across individuals and surgeries.
* Parameter constraints improve **robustness and reproducibility**.
* Early stopping minimizes **overfitting in iterative models**.
* Oversampling safeguards recall for clinically critical minority classes.
* Retaining interactions preserves **complex, biologically grounded relationships**.

---

### 4. Biological Plausibility Validation

#### Objectives

1. Verify predictions align with **established surgical pain hierarchies**:

   * **High pain:** BoneDefect, OcularImplant
   * **Low pain:** Catheter, Castration
2. Reassess features or model parameters when deviations occur.
3. Ensure interpretability suitable for **veterinary and ethical evaluation**.

#### Veterinary Rationale

* Ensures **alignment with physiological evidence** and expected surgical outcomes.
* Detects anomalies such as **implausible pain elevation** in minor procedures.
* Guarantees **clinical reliability** and welfare compliance.

#### Machine Learning Rationale

* Acts as a **biological sanity check** on model logic.
* Guides **feature recalibration** and iterative refinement.
* Enforces explainability for **regulatory and scientific integrity**.

**Key Takeaway:**
PostOpGuard integrates **veterinary realism** with **machine learning rigor**, delivering interpretable, biologically valid, and ethically compliant pain-prediction systems for laboratory animal welfare optimization.

---

## Phase 5 — Evaluation and Interpretation

### 1. Model Evaluation Metrics

#### Objectives

* Evaluate **regression and classification** performance.
* **Regression metrics:** R², MAE, RMSE — for continuous `AnalgesiaNeedScore`.
* **Classification metrics:** Precision, Recall, F1 — for High-pain detection.
* Assess both predictive quality and **clinical actionability**.

#### Veterinary Rationale

* Confirms accurate modeling of **pain distribution and severity**.
* Validates model responsiveness to **critical High-pain cases**.
* Supports **early, targeted analgesic decisions** in post-operative care.

#### Machine Learning Rationale

* Regression metrics quantify **continuous accuracy and calibration**.
* Classification metrics evaluate **sensitivity to rare outcomes**.
* Balanced evaluation minimizes underfitting risk in skewed datasets.

---

### 2. Explainability and Visualization

#### Objectives

1. Compute **feature importance** for all ensemble models.
2. Employ **SHAP (SHapley Additive exPlanations)** to:

   * Quantify local and global feature contributions.
   * Visualize drivers of **High-pain predictions**.
3. Generate interpretive visualizations:

   * `AnalgesiaNeedScore` distribution by surgery type.
   * Confusion matrices per surgery.
   * SHAP summary and dependence plots.
4. Ensure transparency for **clinical and regulatory audit**.

#### Veterinary Rationale

* Identifies **dominant biological drivers** such as `PreOpGrimace`, `DurationPerKg`, and `LocalBlock`.
* Validates reliance on **physiologically plausible features**.
* Detects **outlier or implausible predictions** for expert review.
* Builds veterinary confidence through **transparent interpretability**.

#### Machine Learning Rationale

* Feature importance reveals **global influence hierarchy**.
* SHAP enhances **local interpretability** for individual cases.
* Visualization confirms **alignment between model predictions and domain knowledge**.
* Confusion matrices highlight **surgery-specific prediction fidelity**.
* Meets expectations for **explainable and accountable AI**.

#### Outcome

* **Veterinarians** can understand why a model assigns a particular pain level.
* **Data scientists** can trace the quantitative drivers and ensure fairness.
* Graphical summaries provide **auditable, clinically interpretable insights** into model behavior.

---

# Phase 6 — Reporting and Deployment

## Step 1: Prediction Function

### Objective

Develop a unified function capable of predicting **post-operative analgesia needs** for new surgical cases, providing both clinical and data-driven interpretability.

**Outputs:**

1. `AnalgesiaNeedScore` (0–100) — continuous prediction of analgesic requirement.
2. `ProbHigh` (0–1) — probability of a high-pain outcome.
3. `AnalgesiaNeedCategory` — categorical classification (Low / Medium / High).
4. `ActionRecommendation` — suggested clinical action (ImmediateRescue / Reassess1h / RoutineMonitor).
5. `TopContributingFeatures` — SHAP-derived top explanatory variables.
6. `ConfidenceFlag` — model certainty indicator based on variance and boundary conditions.

### Veterinary Rationale

* Offers **actionable, transparent analgesia guidance** for each individual rabbit.
* Integrates **physiological reasoning** into every prediction output.
* Supports **ethical, reproducible clinical decisions** in compliance with welfare standards.

### Machine Learning Rationale

* Produces both **continuous and categorical** predictions for clinical flexibility.
* Leverages **explainable AI** through SHAP-based interpretability.
* Provides a **confidence estimate** to flag uncertain or outlier predictions.

---

## Step 2: Save Artifacts

### Objectives

1. Preserve **raw and processed datasets** to enable full reproducibility.
2. Save **trained models**, preprocessing pipelines, and configuration metadata.
3. Retain **SHAP explainers** for auditability and longitudinal validation.

### Veterinary Rationale

* Ensures **traceability and accountability** for every prediction generated.
* Facilitates **retrospective review and ethical compliance** under regulatory oversight.
* Enables reproducible veterinary science supporting animal welfare.

### Machine Learning Rationale

* Promotes **transparent, reproducible modeling pipelines**.
* Supports future **model retraining and drift analysis**.
* Maintains explainability for continuous deployment or cross-institutional validation.

---

## Step 3: Deployment Interface and Feedback Loop

### Objectives

1. Develop a **clinician-facing interface** for surgical data entry and model output visualization.
2. Display key indicators:

   * `AnalgesiaNeedScore`
   * `ActionRecommendation`
   * `TopContributingFeatures`
3. Establish a **feedback loop**:

   * Incorporate post-operative outcomes into the dataset.
   * Schedule periodic model retraining.
   * Maintain complete interpretability and audit trails.

### Veterinary Rationale

* Provides **real-time, evidence-based decision support** for veterinary teams.
* Improves precision for **rare or extreme pain scenarios**.
* Adapts to **individual biological variation** while ensuring animal welfare.

### Machine Learning Rationale

* Enables **continuous learning and performance monitoring**.
* Mitigates model drift through regular retraining.
* Preserves **model explainability** for transparency and compliance.

---

## Conclusion and Key Takeaways

### 1. Biological Insights

* **High-pain procedures:** BoneDefect, OcularImplant, Ovariohysterectomy.
* **Low-pain procedures:** Castration, Catheter (modulated by grimace, biomarkers, and analgesic dose).
* Body weight and surgery duration amplify post-operative pain and analgesic requirements.
* *Impact:* Informs targeted analgesic strategies for welfare optimization.

### 2. Model Performance

* **Random Forest Regression:** R² ≈ 0.92, MAE ≈ 3.4, RMSE ≈ 4.7 — demonstrating strong biological realism.
* **High-pain classification:** Precision ≈ 0.93, Recall ≈ 0.90, F1 ≈ 0.92.
* SHAP confirms leading predictors: Duration, PreOpGrimace, PreOpBiomarker, Analgesic dosing.
* *Significance:* Outperforms prior video-based approaches (~87% accuracy) while maintaining interpretability.

### 3. Practical Application

* Functions `predictAnalgesiaNeed` and `deploySurgeryPredictionFullSHAP` support both single and batch inference.
* Outputs include **feature attributions and confidence scores**, enabling focused clinical review.
* *Clinical Utility:* Facilitates rapid, transparent analgesia decisions.

### 4. Limitations and Future Work

* Rare extreme-pain cases require additional data or specialized submodels.
* Continuous data collection will refine performance stability.
* *Next Steps:* Evaluate ensemble generalization, EMR integration, and cross-species validation.

---

## Summary

**PostOpPainGuard™** represents a rigorously developed, interpretable, and biologically valid AI framework for predicting post-operative analgesia needs in rabbits.
By uniting **veterinary expertise** with **machine learning transparency**, it advances ethical decision support, reproducibility, and animal welfare in laboratory research.

---

## Future Work

Potential areas of expansion include:

* Integration of **multimodal data streams** (telemetry, video, or behavioral signals).
* Adaptation to **other species and surgical modalities**.
* Deployment as a **real-time clinical decision support system**.
* Investigation of **personalized analgesic recommendations** through reinforcement learning.

---

## References

1. Feighelstein, R., et al. (2023). *Video-based Deep Learning for Rabbit Post-operative Pain Assessment.* *Journal of Laboratory Animal Science.* [Link](https://www.nature.com/articles/s41598-023-41774-2)
2. Understanding Animal Research. [Rabbit Species Overview](https://www.understandinganimalresearch.org.uk/using-animals-in-scientific-research/animal-research-species/rabbit)
3. AAALAC, IACUC, and 3Rs Guidelines for Ethical Animal Research.

---

## **License**

**PostOpPainGuard™** is released under the **MIT License** — free for academic, research, and non-commercial use.

