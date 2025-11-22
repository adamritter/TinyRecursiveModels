Ideas for toy neurosat
=======================

One of my goals is to get rid of exact match being the only measure for early stopping.

It has been done many times, and they all differ somewhat.

What's clear that it has to be part of training, it's actually more important in training than at inference
(of couse it's fun to use it for inference as well, I'm not against that)

Everything starts simple: have a halting neuron that predicts the probability of halting.

Other parts are a bit different for different implementations:
- Should the training be sampling these probabilites for halting or adding them up?
- Do those probabilites mean ,,probability of halting at this stage relative to survival''
  or absolute pribability of halting at that stage?
- What pondering cost is added?

Sampling is interestingly not done in ACT and not in TRM.
- ACT: probabilities are absolute, added together until they reach 1-epsilon. Halt when they get over that
  Pondering cost: const per each step + remainder (1-p s)
  losses are multiplied by halting probabilities
- TRM:
    q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"],
                                                seq_is_correct.to(outputs["q_halt_logits"].dtype),          
                                                reduction="sum")
    predicts exact match and uses that for halting loss, it's the same as just stopping on exact match,
    not really incentivizing the network to optimize for early stopping
- Increase multiplyer slowly: starts with max depth of 1, increases it if there's an exact match. Works a bit, but
  very crude

The questions is how to incentivize the network to just halt if it's too hard problem to solve.
q_halt_loss is just a simple accuracy predictor.

It doesn't incentivize the network to stop in itself for a hard problem, it still goes to the end.

ACT is more interesting:
- Each step predicts a halting probability p for that step. They are added together, so it can predict the same at each step, and loss can be mutiplied with this prediction. This will incentivize p to be bigger for more correct steps and also loss to count as 1 for 1 item. The main disadvantage is that the network maybe won't learn as much from early steps, but it may be the right thing to do.
- Const per each step is interesting as it should aready just improve as we don't need to learn as much when there already is exact match. I just tried it, it works, speeds up training 2x. It will be even better with early stopping.