This repo contains the code for the experiments in the paper ''Efficient Prior Calibration From Indirect Data''. 

Please cite as:

@article{akyildiz2024efficient,
  title={Efficient Prior Calibration From Indirect Data},
  author={Akyildiz, O Deniz and Girolami, Mark and Stuart, Andrew M and Vadeboncoeur, Arnaud},
  journal={arXiv preprint arXiv:2405.17955},
  year={2024}
}

*****

Non-pip dependencies:

https://github.com/adam-coogan/jaxinterp2d


To run experiments first go to repo base directory and run:

pip install -e . 


To run examples first generate data:

python make_data.py -f <data_filename>.pkl

Then, to train models

python train.py -fd <data_filename>.pkl -ft <saved_model_filename>.pkl

results can be visualized with visualize.ipynb notebooks.