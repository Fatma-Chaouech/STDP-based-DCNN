# **Summary of STDP-based spiking deep convolutional neural networks for object recognition** 

## Introduction

The human brain's exceptional ability to process visual information has captivated researchers for decades. Inspired by the hierarchical feed-forward processing observed in the mammalian visual cortex, researchers have been developing artificial neural networks that closely mimic these biological processes. In the STDP-based Spiking DCNN paper, the author proposes an innovative unsupervised training approach for a neural network that combines Spike-Timing-Dependent Plasticity (STDP) and spike-time neural coding. The trained neural network processes images, and the output is fed into a classifier for further analysis. This approach strives to emulate the way the brain processes visual information, presenting a more biologically plausible model for image processing and feature learning.

## Methodology
The proposed architecture comprises a Difference of Gaussians (DoG) filtering layer, a temporal encoding step, two convolutional layers, a local pooling layer, and a global pooling layer, which directly feeds into a classifier. This structure is designed to closely resemble the hierarchical organization and processing observed in the mammalian visual cortex, paving the way for a more accurate and efficient image processing system.

| Architecture proposed in *STDP-based spiking deep convolutional neural networks for object recognition*  |
|---|
| ![Architecture proposed in *STDP-based spiking deep convolutional neural networks for object recognition*](./docs/architecture.jpg)  |

| Tensor shapes of the layers | Details of the architecture |
| :---: | :---: |
| ![Tensor shapes of the layers](./docs/shapes.jpg) | ![Details of the architecture](./docs/details.jpg) |


#### Image Preprocessing

The first step in the proposed approach is the application of Difference of Gaussians (DoG) filters to the input image. These filters detect contrast by approximating the center-surround properties of the ganglion cells of the retina. The goal is to develop V1-like edge detectors, similar to the primary visual cortex's neurons. The higher the contrast, the higher the output value, and the earlier the neurons fire. Each DoG cell is allowed to fire if its activation is above a certain threshold. There are two types of DoG cells: ON-center, which detects positive contrast, and OFF-center, which detects negative contrast. This ensures that only one cell at each location will fire, creating a sparse representation of the input image.

#### Convolutional Layers

The convolutional layers play a crucial role in processing the input image and extracting features. The neurons in these layers utilize non-leaky integrate-and-fire mechanisms, and learning (STDP) occurs only within these layers. Each feature map within the convolutional layer represents a group of neurons that share the same kernel, or set of weights, detecting the same feature at different locations within the input image. The first neuron to fire in each feature map triggers a series of events, including the STDP weight update process, a lateral inhibition, as well as a Global Intra-map competition mechanisms.

##### STDP-based Learning
After the input image is processed by the DoG filters and the convolutional layers, the network utilizes Spike-Timing-Dependent Plasticity (STDP) to learn the features in an unsupervised manner. STDP is a learning rule based on the relative timing of pre- and post-synaptic spikes, which adjusts the synaptic weights according to the time difference between these spikes. The primary input to STDP is the spike timing of both pre- and post-synaptic neurons, while the output is the updated synaptic weights. The equation governing the STDP weight update is as follows:

$$\Delta\omega_{ij}=\left\{ \begin{array}{ll} a^+\omega_{ij}(1-\omega_{ij}) , \quad \quad if \quad t_j-t_i \leqslant 0 \\ \\ a^- \omega_{ij}(1-\omega_{ij}), \quad \quad if \quad t_j - t_i \gt 0\end{array} \right.$$

This equation indicates that if the presynaptic neuron fires before the postsynaptic one ($$t_j-t_i \leq 0$$), the synaptic weight will be potentiated (strengthened), while if the presynaptic neuron fires after the postsynaptic one ($$t_j-t_i > 0$$), the synaptic weight will be depressed (weakened). By incorporating STDP, the network can learn to represent different features in an unsupervised manner, adapting its weights according to the spiking activity observed during processing. 

Choosing the initial values of the learning rates is crucial for the network's performance. If the learning rates are too large, the network's learning memory will decrease, reducing the network's ability to learn and generalize from the input data. Conversely, too small learning rates will slow down the learning process, making it challenging for the network to converge to a solution. It is essential to choose learning rates that allow neurons to fire and learn one pattern without firing for all of them, balancing the trade-off between learning speed and generalization ability.

The STDP assumes that all presynaptic spikes fall close to the postsynaptic spike time, and time lags are negligible. This assumption simplifies the learning process and ensures that the network can efficiently update the synaptic weights based on the relative timing of pre- and post-synaptic spikes.

The learning process is focused exclusively on the convolutional layers and progresses layer by layer. A learning convergence metric is defined to measure the extent to which a layer has learned. This metric is calculated using the following equation:  $$\quad C_l = \sum_i\sum_jw_{i,j}(1-w_{i,j})/n_w$$ 
Here, $$n_w$$ represents the total number of synaptic weights within that layer. A layer will not be processed until the preceding layer has achieved a learning convergence of $$C_l \lt 0.01$$. This approach ensures that each layer effectively learns to represent the input data before the learning process advances to the subsequent layer.
##### Lateral Inhibition Mechanism

The network incorporates several competition mechanisms to ensure that different features are learned by different neurons. The first mechanism is the Lateral Inhibition mechanism, which presents a local inter-map competition. When the first neuron in a feature map fires, it inhibits other neurons within a small neighborhood around its location belonging to other neuronal maps. This mechanism resets the potentials of inhibited neurons to zero, preventing them from firing until the next image is shown.

##### Global Intra-Map Competition Mechanism
In addition to the Lateral Inhibition mechanism, the network incorporates a Global Intra-map competition mechanism. This competition occurs among neurons within the same feature map. The winner in this competition triggers the STDP, updates the weights, and prevents the rest of the neurons in that map from firing, imposing the update of its synaptic weights onto them. This competition is essential for selecting the most prominent neuron within a map. Neurons are not allowed to fire more than once per image, which results in sparse but highly informative coding.

| Global intra-map competition and Local inter-map competition  |
|---|
| ![Global intra-map competition and Local inter-map competition](./docs/competition.jpg)  |

#### Pooling Layer
The pooling layer serves a critical function in condensing and summarizing the visual information extracted by the convolutional layers. The architecture includes a local pooling layer and a global pooling layer. 

##### Local Pooling Layer
The local pooling layer is a max-pooling layer that is responsible for reducing spatial dimensions while preserving the essential features identified in the convolutional layers. This reduction in dimensionality helps minimize the computational complexity and control overfitting. 

##### Global Pooling Layer
The global pooling layer, on the other hand, performs a global max pooling operation for each neuronal map. As a result, the output of this layer consists of a single value for each map, representing the presence (1) or absence (0) of a specific visual feature. The output generated by the global pooling layer is then utilized to train a classifier, such as an SVM classifier, for further analysis and decision-making based on the processed visual information.

## Conclusion
The proposed STDP-based Spiking DCNN offers several advantages compared to its counterparts. Firstly, it represents the first spiking neural network with more than one learnable layer capable of processing large-scale natural object images. Secondly, the network employs an efficient temporal coding scheme that encodes visual information based on the timing of the first spikes, allowing for low spike count processing and rapid processing time steps. Lastly, the network leverages the bio-inspired and entirely unsupervised STDP learning rule, enabling it to learn diagnostic object features while disregarding irrelevant backgrounds. These advantages make the proposed SDNN a powerful and innovative approach to processing and learning from visual data.                 $a^+ , a^- :$  learning rates
