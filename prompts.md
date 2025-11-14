"Now I want you to make another variant of the eval precond comprehensive

It should take in two models:
One model is the base pretrained model
One model is the preconditioned model

During inference/eval, this is how it should work:
1. First we run inference on the base pretrained model to get the forecasts in the actual space

2. Next run inference on the preconditioned model to get the forecasts in the preconditioned space 

3. Now recover the "hybrid" forecast by reversing the preconditioning of the preconditioned model, but using the first model's outputs. 

In particular, if the preconditioned model predicts some delta, we will use the first model's previous outputs, and add it to that delta to get the actual forecast. 

This new "hybrid" forecast should then be evaluated according to the ground truth. 
