![image-20200618201511388](C:\Users\Willi\AppData\Roaming\Typora\typora-user-images\image-20200618201511388.png)







![image-20200618203148615](C:\Users\Willi\AppData\Roaming\Typora\typora-user-images\image-20200618203148615.png)













Normalization equations: Consider the intermediate activations (x) of a mini-batch of size \(m\): We can compute the mean and variance of the batch $\({\mu_B} = \frac{1}{m} \sum_{i=1}^{m} {x_i}\) \({\sigma_B^2} = \frac{1}{m} \sum_{i=1}^{m} ({x_i} - {\mu_B})^2\)$ and then compute a normalized \(x\), including a small factor $\({\epsilon}\)$ for numerical stability. $\(\hat{x_i} = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}\) $And finally $(\hat{x})$ is linearly transformed by $({\gamma}\)$ and $\({\beta}\),$ which are learned parameters: $\({y_i} = {\gamma * \hat{x_i} + \beta}$