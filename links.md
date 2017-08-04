# Training - Books, Lectures, Tutorials, etc

- [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning) by Andrew Trask. Use our exclusive discount code traskud17 for 40% off. This provides a very gentle introduction to Deep Learning and covers the intuition more than the theory.
- [Neural Networks And Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen. This book is more rigorous than Grokking Deep Learning and includes a lot of fun, interactive visualizations to play with.
- [The Deep Learning Textbook](http://www.deeplearningbook.org/) from Ian Goodfellow, Yoshua Bengio, and Aaron Courville. This online book has lot of material and is the most rigorous of the three books suggested.

## Papers
- Reducing overfitting - "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" - https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

## Videos
- [Visualizing and Understanding Deep Neural Networks by Matt Zeiler\(https://www.youtube.com/watch?v=ghEmQSxT6tw)

## Training / Tutorials
- Andrew Ng - https://www.coursera.org/learn/machine-learning
- Machine Learning Tutorials - https://github.com/aymericdamien/Machine-Learning-Tutorials
- TensorFlow Examples - https://github.com/aymericdamien/TensorFlow-Examples

# Deep Learning Projects
- A list of popular github projects related to deep learning - https://github.com/aymericdamien/TopDeepLearning
- Image augmentation for machine learning experiments - https://github.com/aleju/imgaug

# People
- Andrew Ng - http://www.andrewng.org

# Applying DL

[Fast style transfer](https://github.com/lengstrom/fast-style-transfer)
[Deep Traffic](http://selfdrivingcars.mit.edu/deeptrafficjs/)
[Flappy Bird](https://github.com/yenchenlin/DeepLearningFlappyBird)

# Concepts

## CNN - Convolutional Neural Networks
- [Andrej Karpathy's CS231n Stanford course on Convolutional Neural Networks](http://cs231n.github.io)
- Paper "Visualizing and Understanding Convolutional Networks" - http://www.matthewzeiler.com/pubs/arxive2013/eccv2014.pdf
https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
- ConvNet Max Pooling - https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
- http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/
- https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/
- http://cs231n.github.io/convolutional-networks/
- http://deeplearning.net/tutorial/lenet.html
- https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
- http://neuralnetworksanddeeplearning.com/chap6.html
- http://xrds.acm.org/blog/2016/06/convolutional-neural-networks-cnns-illustrated-explanation/
- http://andrew.gibiansky.com/blog/machine-learning/convolutional-neural-networks/
- https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721#.l6i57z8f2

## RNN - Recurrent Neural Networks
- [Andrej Karpathy's lecture on RNNs and LSTMs from CS231n](https://www.youtube.com/watch?v=iX5V1WpxxkY)
- [A great blog post by Christopher Olah on how LSTMs work](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Building an RNN from the ground up, this is a little more advanced, but has an implementation in TensorFlow](http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html)
- http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- [https://github.com/karpathy/char-rnn](Implementation in Torch)
- http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
- https://github.com/sherjilozair/char-rnn-tensorflow
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [LSTM Networks for Sentiment Analysis](http://deeplearning.net/tutorial/lstm.html)
- [A Beginner's Guide to Recurrent Networks and LSTMs](https://deeplearning4j.org/lstm.html)
- [TensorFlow's Recurrent Neural Network Tutorial](https://www.tensorflow.org/tutorials/recurrent)
- [Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras](http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
- [Demystifying LSTM neural networks](https://blog.terminal.com/demistifying-long-short-term-memory-lstm-recurrent-neural-networks/)

### LSTM Vs GRU
- "These results clearly indicate the advantages of the gating units over the more traditional recurrent units. Convergence is often faster, and the final solutions tend to be better. However, our results are not conclusive in comparing the LSTM and the GRU, which suggests that the choice of the type of gated recurrent unit may depend heavily on the dataset and corresponding task."
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555) by Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio

- "The GRU outperformed the LSTM on all tasks with the exception of language modelling"
- [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf) by Rafal Jozefowicz, Wojciech Zaremba, Ilya Sutskever

- "Our consistent finding is that depth of at least two is beneficial. However, between two and three layers our results are mixed. Additionally, the results are mixed between the LSTM and the GRU, but both significantly outperform the RNN."
- [Visualizing and Understanding Recurrent Networks](https://arxiv.org/abs/1506.02078) by Andrej Karpathy, Justin Johnson, Li Fei-Fei

- "Which of these variants is best? Do the differences matter? [Greff, et al. (2015)](https://arxiv.org/pdf/1503.04069.pdf) do a nice comparison of popular variants, finding that they’re all about the same. [Jozefowicz, et al. (2015)](http://proceedings.mlr.press/v37/jozefowicz15.pdf) tested more than ten thousand RNN architectures, finding some that worked better than LSTMs on certain tasks."
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Chris Olah

- "In our [Neural Machine Translation] experiments, LSTM cells consistently outperformed GRU cells. Since the computational bottleneck in our architecture is the softmax operation we did not observe large difference in training speed between LSTM and GRU cells. Somewhat to our surprise, we found that the vanilla decoder is unable to learn nearly as well as the gated variant."
- [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/abs/1703.03906v2) by Denny Britz, Anna Goldie, Minh-Thang Luong, Quoc Le

### Example RNN Architectures

| Application | Cell | Layers | Size | Vocabulary | Embedding Size | Learning Rate | Links |
| --- | --- | --- | --- | --- | --- | --- | --- |
|Speech Recognition (large vocabulary)|LSTM|5, 7|600, 1000|82K, 500K|--|--|[paper](https://arxiv.org/abs/1610.09975)|
|Speech Recognition|LSTM|1, 3, 5|250|--|--|0.001|[paper](https://arxiv.org/abs/1303.5778)|
|Machine Translation (eq2seq)|LSTM|4|1000|Source: 160K, Target: 80K|1,000|--|[paper](https://arxiv.org/abs/1409.3215)|
|Image Captioning|LSTM|--|512|--|512|(fixed)|[paper](https://arxiv.org/abs/1411.4555)|
|Image Generation|LSTM|--|256, 400, 800|--|--|--|[paper](https://arxiv.org/abs/1502.04623)|
|Question Answering|LSTM|2|500|--|300|--|[pdf](http://www.aclweb.org/anthology/P15-2116)|
|Text Summarization|GRU||200|Source: 119K, Target: 68K|100|0.001|[pdf](https://pdfs.semanticscholar.org/3fbc/45152f20403266b02c4c2adab26fb367522d.pdf)|

## Hyperparameters

If you want to learn more about hyperparameters, these are some great resources on the topic:
 
- [Practical recommendations for gradient-based training of deep architectures](https://arxiv.org/abs/1206.5533) by Yoshua Bengio
- [Deep Learning book - chapter 11.4: Selecting Hyperparameters](http://www.deeplearningbook.org/contents/guidelines.html) by Ian Goodfellow, Yoshua Bengio, Aaron Courville
- [Neural Networks and Deep Learning book - Chapter 3: How to choose a neural network's hyper-parameters?](http://neuralnetworksanddeeplearning.com/chap3.html#how_to_choose_a_neural_network's_hyper-parameters) by Michael Nielsen
- [Efficient BackProp (pdf)](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) by Yann LeCun

*More specialized sources:*

- ]How to Generate a Good Word Embedding?](https://arxiv.org/abs/1507.05523) by Siwei Lai, Kang Liu, Liheng Xu, Jun Zhao
- ]Systematic evaluation of CNN advances on the ImageNet](https://arxiv.org/abs/1606.02228) by Dmytro Mishkin, Nikolay Sergievskiy, Jiri Matas
- ]Visualizing and Understanding Recurrent Networks](https://arxiv.org/abs/1506.02078) by Andrej Karpathy, Justin Johnson, Li Fei-Fei

### Weight Initialization
- [Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852v1.pdf)
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167v2.pdf)

### Learning Rate
- [Exponential Decay in TensorFlow](https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay)
- Adaptive Learning Optimizers
  - [AdamOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)
  - [AdagradOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer)

### Batch size
- [Systematic evaluation of CNN advances on the ImageNet by Dmytro Mishkin, Nikolay Sergievskiy, Jiri Matas](https://arxiv.org/abs/1606.02228)

### Number Of Hidden Units Layers
- Andrej Karpathy in https://cs231n.github.io/neural-networks-1/
- [Model's capacity appears in the Deep Learning book, chapter 5.2 (pages 110-120)](http://www.deeplearningbook.org/contents/ml.html)

## Sentiment Analysis
- http://deeplearning.net/tutorial/lstm.html
- https://www.quora.com/How-is-deep-learning-used-in-sentiment-analysis
- https://gab41.lab41.org/deep-learning-sentiment-one-character-at-a-t-i-m-e-6cd96e4f780d#.nme2qmtll
- http://k8si.github.io/2016/01/28/lstm-networks-for-sentiment-analysis-on-tweets.html
- https://www.kaggle.com/c/word2vec-nlp-tutorial
- Lexical Based Approach (predicting using a prebuilt Lexicon)
  - https://www.aclweb.org/anthology/J/J11/J11-2001.pdf
  - http://ceur-ws.org/Vol-1314/paper-06.pdf


## Backpropagation
- From Andrej Karpathy: [Yes, you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.vt3ax2kg9)
- Andrej Karpathy, [a lecture on Backpropagation from Stanford's CS231n course](https://www.youtube.com/watch?v=59Hbtz7XgjM)

## Data Preparation

- Loading and manipulating data with Pandas in our [Intro to Data Analysis course](https://www.udacity.com/course/intro-to-data-analysis--ud170)
- Cleaning data with Python in our [Data Wrangling course](https://www.udacity.com/course/data-wrangling-with-mongodb--ud032)
- Feature Scaling and Principle Component Analysis (PCA) in [Intro to Machine Learning](https://www.udacity.com/course/intro-to-machine-learning--ud120) featuring our own Sebastian Thrun.

## Math

- Matrix Multiplication - Dot product - https://en.wikipedia.org/wiki/Dot_product
- Linear Algebra cheatsheet: http://www.souravsengupta.com/cds2016/lectures/Savov_Notes.pdf
- Calculus cheatsheet: http://tutorial.math.lamar.edu/pdf/Calculus_Cheat_Sheet_All.pdf
- Statistics cheatsheet: http://web.mit.edu/~csvoss/Public/usabo/stats_handout.pdf

More Learning Resources: 
- https://people.ucsc.edu/~praman1/static/pub/math-for-ml.pdf
- http://www.vision.jhu.edu/tutorials/ICCV15-Tutorial-Math-Deep-Learning-Intro-Rene-Joan.pdf
- http://datascience.ibm.com/blog/the-mathematics-of-machine-learning/

[How To Do Linear Regression Live Stream](https://www.youtube.com/watch?v=XdM6ER7zTLk)

## Other concepts
[An overview of gradient descent algorithms](http://sebastianruder.com/optimizing-gradient-descent/index.html#momentum)
https://en.wikipedia.org/wiki/Perceptron
https://en.wikipedia.org/wiki/Topological_sorting
https://en.wikipedia.org/wiki/Sigmoid_function
https://en.wikipedia.org/wiki/Reinforcement_learning


# Datasets

THE MNIST DATABASE of handwritten digits - http://yann.lecun.com/exdb/mnist/
