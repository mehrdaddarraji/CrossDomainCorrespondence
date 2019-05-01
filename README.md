#Analysis of Different Networks for Sparse Cross-Domain Correspondence
By: Mehrdad Darraji, Jorge Garcia, Adriana Ixba, and Carlos Valdez

#I. Description of Problem
For this project, we have gained inspiration from the “Neural Best Buddies: Sparse Cross-Domain
Correspondence” paper. We will use different pre-trained networks and datasets to find crossdomain correspondences, and use it for morphing between pets and their owners. The paper
defines sparse cross-domain correspondence as “a more general and challenging version of the
sparse correspondence problem, where the object of interest in the two input images can differ
more drastically in their shape and appearance, such as objects belonging to different semantic
categories (domains).”

#II. Project Datasets and Software Packages
We are not training any data, we'll be using pre-trained classifiers and testing their performance.
For testing the image correspondence, we will compare the Labeled Faces and dogs In the Wild
dataset by the University of Massachusetts, Stanford's Dogs dataset, Cifar-100, and a small
custom dataset to morph dogs to their owner’s face. The authors of the paper use PyTorch to
implement their algorithm. We will also use the PyTorch library to implement the different
networks because it has faster deep learning training.

#III. Implementation
The paper uses VGG19 pre-trained network to find correspondences between images in the
Pascal 3D+ dataset. We will implement the algorithms of the paper on VGG19, ResNet50, and
InceptionV3.

#IV. Set of Experiments
Run cross-domain correspondences on the general dataset (cifar-100). Run cross-domain
correspondences on faces & dogs datasets. Create morphing using images of owners and their
dogs from our custom dataset.

#V. Measuring Success
We plan to compare and contrast different pre-trained networks to the VGG19 one used on the
paper to see which would give the best results for cross-domain correspondence problem and has
the ability to morph images the best. We will base our success measures off of how the paper
evaluates their results, “visually compare NBB-based key point selection” with other NBB-based
key point selections of other networks, we will evaluate our methods on different objects as well
as objects of similar categories. We will analyze which network performs the best cross-domain
correspondences.