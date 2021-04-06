# Questions MIDA 2

*A series of answered questions from past exams of the course MIDA 2 as taught in Politecnico di Milano by Savaresi*

[TOC]

<div style="page-break-after: always;"></div> 
### 1 - Briefly explain the concept of minimum variance control for a given plant described by an ARX model, when can this method be used and when not?

General Idea:
Given an input $u$ and an output $y$, we aim to retroactively intervein on the input $u$ in such a way that the output $y$ takes values always closer to a certain desired output $y^0$. In order to do so we suppose that the continuous input gets sampled and put into the system discretized, then reconverted in continuous domain at the output.  

HERE I WRITE THE REQUIREMENTS FOR MVC...

Let's get into a specific ARX process:

Let's suppose we have the following process $(e(t)\sim WN(0,1))$:
$$
y(t)=\frac{1}{2}y(t-1)+u(t-1)+2u(t-2)+2e(t)
$$
Let's write in operational representation:
$$
y(t)=\frac{1-2z^{-1}}{1-\frac{1}{2}z^{-1}}z^{-1}u(t)+\frac{z}{1-\frac{1}{2}z^{-1}}e(t)
$$
$B(z)$ (coefficient of $u(t)$) it's not in minimum phase. I could not use the minimum variance. If I'd do it, in fact, I would obtain the following transfer function from the desired input to $u$:
$$
f.d.t_{y^0\to u}=\frac{1-\frac{1}{2}z^{-1}}{1+2z^{-1}}
$$
This function is unstable. this means that, at small changes of the desired output the system generates an internally unstable solution

HERE I TALK ABOUT HOW TO TACKLE THIS PROBLEM -> GENERALIZED MVC

SO IN ARX PROCESSES I BEHAVE DIFFERENTLY IN 2 CASES:

$B$ Minimum Phase? Normal MVC

otherwise GMVC.

<div style="page-break-after: always;"></div> 
### 2 - How can we estimate both the state x(t) and the unknown parameter a of the system $\begin{cases}x(t+1)=ax(t)+v_1(t) \\ y(t)=x(t)+v_2(t) \end{cases}$ by means of the extended Kalman filter



### 3 - Explain the meaning of model reference control



