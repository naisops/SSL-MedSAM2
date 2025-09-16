# SSL-MedSAM2
The code of the method SSL-MedSAM2, a SAM2 based semi-supervised learning method published in the MICCAI challenge CARE 2025.

## Instruction
### MedSAM2 installnation
Install MedSAM2 following the instructions on https://github.com/bowang-lab/MedSAM2/tree/main 
### Initial pseudo label generation
1. Copy the .py files to medsam2 folder.
2. Put the labelled and unlabelled data in the corresponding folder.
3. Run generate_logits_3d.py
### Pseudo label refinement via nnUNet
1. Install nnUNetV2 following the instructions on https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md
2. Based on the whole unlabelled and labelled data along with GT masks and pseudo masks to train nnUNetV2, then refine the pseudo masks
3. Repeat step2 untill the pseudo masks converge 
### Inference with nnUNet
Use the final trained nnUNet for model inference.
