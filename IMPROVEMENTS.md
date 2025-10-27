*** Improvements compared to TRM architecture ***

This document explains what improvements were made compared to the TRM architecture to improve solution accuracy.

** Increasing steps at inference time, using adaptive stopping **

Sudoku Extreme exact accuracy goes up from 87% to 96% with the original TRM model when inferencing with at most 1024 steps
instead of 16 steps. To make this inferencing efficient, q_logit is used for stopping it when a task is solved. It makes
the average number of steps 84 instead of 1024. 

What's also interesting is that q_halt_accuracy is about 98%, which should be much closer to 100% as it's trivial to check
if a Sudoku problem's solution is correct or not.

This result also means that TRM is quite good at solving Sudoku, and the battleneck seems to be the verification.

Increasing steps at inference time doesn't improve ARC-AGI test accuracy at all, and the reason may be becasuse q_halt_accuracy (the accuracy of q_logit > 0 meaning )
is 0.75.

I started experiments of improving the q head to solve the trivial mistakes it makes for Sudoku Extreme, and hope that it improves
its results in a way that can be translated to ARC-AGI task as well.

** Adding output.detach() to q_head input **

q_head is an extremely small neural network, although I don't see any reason why it couldn't be similar size to other parts of the TRM.

TRM's documentation shows that adding more hidden layers don't help, but after checking the output, I see that the q_head is not always able to find
if a number is multiple times in the same row. I believe a 2 layer MLP should be able to find this, but I though that first just adding output.detach() to the linear
head should be a good start to see if I'm going in the right direction.


