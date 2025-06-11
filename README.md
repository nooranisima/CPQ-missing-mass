
This is the official repository codebase for the paper: Conformal Prediction Beyond the Seen: A Missing Mass Perspective for Uncertainty Quantification in Generative Models

**"Conformal Prediction Beyond the Seen: A Missing Mass Perspective for Uncertainty Quantification in Generative Models"**  
*by Sima Noorani\*, Shayan Kiyani\*, George Pappas, and Hamed Hassani*

📄 We introduce a principled framework for uncertainty quantification based on the concept of **missing mass**. By modeling the probability of failing to observe the true label through a query oracle, we derive two algorithmic principles that guide both query allocation under a budget and prediction set construction. Our resulting algorithm achieves significantly more informative prediction sets while preserving distribution-free coverage guarantees in settings with black-box access to models like large language models (LLMs).

---

## 🔧 Repository Structure

```text
.
├── data/               # data generated from the models for each dataset for reproducibility
├── results/            # final results from running the algorithm for different coverage levels (1-\alpha)
├── scripts/
│   ├── sampling_module.py  # functions needed to simulate the online optimal sampling module + clustering from the data
│   └── calibration_module.py  # optimal calibration and prediction set construction
│   └── suboptimal_baselines.py  # includes the sub-optimal, yet valid calibration and prediction set construction for fine-grained comparison 
├── cpq_experiments          # Run the experiments from the data, and save the results + reproduce the figures in the paper
└── README.md           # you are here!
```
## 📬 Contact

For questions or updates, please contant:
*nooranis@seas.upenn.edu*


## Citation

If you find this repository useful for your research, please consider citing:

```bibtex
@misc{noorani2025conformalpredictionseenmissing,
  title        = {Conformal Prediction Beyond the Seen: A Missing Mass Perspective for Uncertainty Quantification in Generative Models},
  author       = {Sima Noorani and Shayan Kiyani and George Pappas and Hamed Hassani},
  year         = {2025},
  eprint       = {2506.05497},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2506.05497},
}
