# Relation Extraction - Initial Predictions  

This folder contains the initial predictions for relation extraction, obtained using an improved baseline model. The predictions are structured to facilitate post-evaluation with **RelCheck**.  

## 📌 How Initial Predictions are Obtained  

The initial predictions for this project are generated using the repository:  
**Improved Baseline for Relation Extraction**  
🔗 [GitHub Repository](https://github.com/wzhouad/RE_improved_baseline)  

We have modified the `evaluate.py` file to output **Predictions + Confidence Scores** in a structured format. This enables easy post-evaluation using **RelCheck**.  

### 📖 Citation  

If you use this work, please cite the original paper:  

```bibtex
@inproceedings{zhou2022improved,
  title={An Improved Baseline for Sentence-level Relation Extraction},
  author={Zhou, Wei and Chen, Muhao and Chang, Kai-Wei},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2022}
}
