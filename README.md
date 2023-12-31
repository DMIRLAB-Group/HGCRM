# Source code of 'Granger Causal Representation Learning for Groups of Time Series'

## Overview

Provides the HGCRM method to learn granger causal representation for groups of time series.

Details of the algorithm can be found in "Granger Causal Representation Learning for Groups of Time Series".

## Requirements

The scripts in this repository were tested with the following package versions: 

- **python** 3.8.3
- **numpy** 1.18.1
- **matplotlib** 3.1.3
- **scikit-learn** 1.1.2
- **seaborn** 0.12.0
- **scipy** 1.9.1
- **pytorch** 1.12.1
- **tigramite** 5.0.0.3
- **causalnex** 0.11.0

# Reproduce synthetic experiments:

Run the model on synthetic dataset, for example:

```python
python exp_synthetic.py
```

# Citation

If you find this useful for your research, we would be appreciated if you cite the following papers:

```
@article{cai2023hgcrm,
  title={Granger Causal Representation Learning for Groups of Time Series},
  author={Cai, Ruichu and Wu, Yunjin and Huang, Xiaokai and Chen, Wei and Fu, Tom Z. J. and Hao, Zhifeng},
  journal={Science China Information Sciences},
  year={2023}
}
```

