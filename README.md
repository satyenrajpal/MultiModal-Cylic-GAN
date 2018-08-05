# BaLSaKGAN-
Text To Image Synthesis usgin Generative Adversarial Networks...
Download these files-
Glove- http://nlp.stanford.edu/data/glove.6B.zip
MSCOCO Validation images-http://images.cocodataset.org/zips/val2014.zip
MSCOCO Train/Val Captions-http://images.cocodataset.org/annotations/annotations_trainval2014.zip

Prerequisites-
torchvision <br>
pytorch(python 3.x) <br>
PIL <br>
torchfile <br>

# TODO (In order!)
 - [ ] Replace GAN loss with Wasserstein gradient penalty
 - [ ] Train on Flowers dataset
 - [ ] Force input caption and output caption to be in same space of images. Share weights between input caption encoder and caption predicition.
 - [ ] Train captioning model as well? <- Not sure about this


