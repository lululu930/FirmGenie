# FirmGenie  
**A Confidence-Aware Expert LLM Framework for Fine-grained Network Device Identification**

This repository contains the implementation and experimental artifacts for **FirmGenie**, an expert large language model framework designed for fine-grained firmware version identification from noisy network banners. FirmGenie integrates domain-specific supervised fine-tuning, knowledge augmentation, and multi-source confidence discrimination to achieve accurate and scalable firmware version identification under real-world Internet conditions.

---

## Repository Structure

├── Confidence_Discriminator/ # Multi-source confidence discriminator module
├── Knowledge_base/ # Structured knowledge base and validation rules
├── SFT_module/ # Supervised fine-tuning (SFT) for the expert LLM
├── inference/ # End-to-end inference pipeline
├── FirmGenie_Rebuttal_Revised_Version.pdf
├── README.md


---

### Module Overview

- **SFT_module/**  
  Implements domain-specific supervised fine-tuning to adapt the base LLM for device and firmware version understanding.

- **Knowledge_base/**  
  Contains curated firmware knowledge used for knowledge-augmented validation and candidate filtering.

- **Confidence_Discriminator/**  
  Implements the multi-source confidence control mechanism that fuses model confidence, knowledge-based confidence, and semantic reasoning signals.

- **inference/**  
  Provides the full inference pipeline integrating extraction, validation, and confidence-aware decision fusion.

---

## Rebuttal-Stage Updates and Supplementary Experiments

This repository includes **additional materials prepared during the rebuttal phase** of our ICASSP submission, in direct response to reviewers’ comments and requests for clarification.

### Rebuttal Revised Paper

- **`FirmGenie_Rebuttal_Revised_Version.pdf`**  
  This PDF is a carefully revised version of the manuscript prepared in response to reviewers’ comments.
#### Revisions span multiple sections, including the abstract, introduction, experimental evaluation, conclusion, and references. 

All substantive changes are explicitly highlighted in **red** to facilitate review and comparison.

#### Added Model Efficiency Experiments (Appendix)

To provide **additional evidence during the rebuttal stage**, we introduced a new appendix in the revised paper that reports **supplementary efficiency experiments**, including:

- Inference latency and throughput comparisons across:
  - General-purpose LLMs of different parameter scales
  - The teacher model DeepSeek-R1-671B
  - The FirmGenie expert model and full pipeline
- Pipeline-level runtime breakdown analyzing:
  - LLM inference
  - Knowledge-based validation
  - Confidence discrimination overhead
- Deployment cost comparison highlighting:
  - Zero-cost local deployment of FirmGenie
  - Recurring API costs for large commercial models

These experiments were **explicitly added during the rebuttal phase** to strengthen empirical support for FirmGenie’s practicality, efficiency, and suitability for Internet-scale deployment.

---

### Full rebuttal response:

We thank the reviewers for their constructive feedback and recognition of our contribution. Below, we address the major concerns:

(1) Scope.
Reviewer-4996 and Reviewer-3717 explicitly validated our work as clearly within scope. While Reviewer-1E70 marked the paper as out of scope, they acknowledged our contribution "deserves a publication". We respectfully clarify that our framework is fundamentally a signal processing pipeline: we treat unstructured network banners as noisy, discrete signals. By performing signal processing strategy and multi-source decision fusion, we reconstruct latent device states (firmware versions). This methodology rigorously aligns with the technical focus of the AS-APP-IOT and IF-SEC-NETW tracks of ICASSP.

(2) Comparisons with Nmap.
Nmap is included as a deployment baseline rather than a learning-based competitor, reflecting prevailing practice in real-world firmware identification. The large F1 gap highlights a fundamental limitation of static-rule systems: they prioritize precision at the expense of recall, making them ineffective for Internet-scale mapping where signal diversity is high. 
Crucially, our evaluation is comprehensive: Table 2 also benchmarks against Shodan and multiple LLMs, where FirmGenie consistently demonstrates superior performance.

(3)Model efficiency and training details.
To evaluate efficiency, we benchmarked all models on a single NVIDIA-A100-GPU using 1,000 random test samples (DeepSeek-R1-671B was tested via Volcengine API due to memory constraints), our SFT-model achieves 7.74s mean latency (465 samples/hour), which is 2.8× faster than DeepSeek-R1-671B (21.54s) and 2.2× faster than Qwen2.5-7B (16.98s), while our full pipeline achieves 13.25s mean latency, remaining competitive with all baseline models. Critically, FirmGenie enables zero-cost local deployment, whereas DeepSeek-R1-671B incurs approximately $3.21/1,000 samples API cost. Detailed experimental charts (latency and throughput comparison, pipeline breakdown) and training details are available at https://github.com/lululu930/FirmGenie (already given in the manuscript).

(4) Citation issues.
- Regarding citation [19], our original intent was to highlight broader limitations of general-purpose LLMs when applied outside natural language setting. The citation will be replaced. 
- Reviewer-3717 notes that only two prior LLM-based studies are cited. We emphasize that literature specifically targeting LLM-based device identification remains very limited; the cited works are the closest related efforts and none addresses the harder problem of firmware version identification. This gap directly motivates our work, and we will continue to incorporate newly emerging studies.

(5) Corrections. 
The recall “drop” mentioned in the text corresponds to an earlier experimental version; Table 3 contains the correct updated results. We have revised the text accordingly. All other suggestions have been incorporated, with the updates uploaded to our repository.



## Notes on Experimental Reproducibility

Key training and inference configurations are summarized in the SFT_module folder.  
Detailed implementation settings, scripts, and module-level code are provided in this repository to support reproducibility and further exploration.

---


