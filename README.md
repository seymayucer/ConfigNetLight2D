# CONFIG: Controllable Face Image Generation for Race-Related Facial Phenotype

## Abstract

Achieving fine-grained control over the generative process of 2D face images while preserving their realism and identity is a challenging task due to the high complexity of the 2D image pixel space. Despite significant efforts to exert fine-tuned control over the generative process, the results only become satisfactory for certain specific attribute control including illumination, head pose, smiling etc. In this study, we propose a 2D-aware framework based on ConfigNet and StyleGAN2 to enable control over individual aspects of the output images that are relevant to facial and racial phenotype. Our framework factorises the latent space into elements that correspond to the inputs of race-related facial phenotype representations, thereby separating aspects such as skin colour, hair colour, nose shape, and mouth shape, which are difficult to annotate in real data. Unlike ConfigNet, our framework does not rely on 3D data or 3D-aware GANs and achieves state-of-the-art individual control over attributes in the output images while improving their photo-realism.


## 3rd Party Libraries
This implementation uses [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) to detect facial landmarks that are required to align the face images. When using our code and pre-trained models make sure to follow the constraints of the OpenFace license as well as [the licences of the datasets used in its training](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Datasets).

The pre-trained models were trained using data from the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) and CelebHQ and their use is thus constrained by the licence of FFHQ and CelebHQ.


## Citation
If you use this code, models or method in your research, please cite papers as follows.
```
@inproceedings{KowalskiECCV2020,
    author = {Kowalski, Marek and Garbin, Stephan J. and Estellers, Virginia and Baltrušaitis, Tadas and Johnson, Matthew and Shotton, Jamie},
    title = {CONFIG: Controllable Neural Face Image Generation},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2020}
}
```