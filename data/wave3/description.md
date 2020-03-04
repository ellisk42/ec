This model performs type-directed top-down enumeration in (approximately) decreasing order of prior probability. This enumeration algorithm is described in detail in the appendix to dreamcoder. You can get the appendix at http://www.mit.edu/~ellisk/dreamcoder.pdf (S4.4), but please don't distribute this link. If you want to cite something published you can use EC^2, whose enumeration algorithm is identical:
@inproceedings{Ellis2018LearningLO,
  title={Learning Libraries of Subroutines for Neurally-Guided Bayesian Program Induction},
  author={Kevin Ellis and Lucas Morales and Mathias Sabl{\'e}-Meyer and Armando Solar-Lezama and Joshua B. Tenenbaum},
  booktitle={NeurIPS},
  year={2018}
}


I used a single CPU per task. There was no off-line training or learning of parameters.  Enumeration proceeds in depth-first fashion, with an outer loop of iterative deepening: it first enumerates programs whose description length lies between 0-Delta, then all programs whose description length lies between Delta-2Delta, then 2Delta-3Delta, etc., until a 10 minute timeout is reached. Delta is a hyper parameter which I have set to 1.5 nats, but I did not tune this value and it should have little effect on the final result.