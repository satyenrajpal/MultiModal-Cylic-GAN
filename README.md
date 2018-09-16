# Multimodal Cyclic StackGAN
Text To Image Synthesis using stacked Generative Adversarial Networks.

## Code is messy. Please proceed with caution

Requirements-
torchvision <br>
pytorch(python 3.x) <br>
PIL <br>
torchfile <br>

Download these files-
Glove- http://nlp.stanford.edu/data/glove.6B.zip
MSCOCO Validation images-http://images.cocodataset.org/zips/val2014.zip
MSCOCO Train/Val Captions-http://images.cocodataset.org/annotations/annotations_trainval2014.zip

We experimented with a novel architecture-
![](/docs/Architecture_t2i.png)
<br>
The idea behind this architecture is that the captioning model provides additional feedback to the GAN. A captioning model maps low dimensional data (text) to high dimensional data (images), which is generally easier than the reverse. We exploit this by providing a cyclic feedback mechanism across different modalities. Specifically, the caption when mapped back from the image space should have the same feature representation. <br>
The results were not satisfactory. We suspect that the cosine embedding loss might not be right choice, instead an L2-loss should suffice. In addition, we enforce the captioning model bias on the generative model. A better approach would be to train the captioning model simultaneously with the GAN. This would allow for a common manifold to be learnt shared across both the generative and captioning model.   

# TODO 
 - [ ] Replace GAN loss with Wasserstein gradient penalty
 - [ ] Train on Flowers dataset
 - [ ] Force input caption and output caption to be in same space of images. Share weights between input caption encoder and caption predicition.
 - [ ] Train captioning model as well

### I'm currently tied up with other projects. However, I do aim to bring this to fruition sometime soon. 


