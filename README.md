# cspine-2022-challenge-inference-collection

This repository contains inference scripts for the winning solutions of the [RSNA 2022 Cervical Spine Fracture Detection Challenge](https://www.kaggle.com/c/rsna-2022-cervical-spine-fracture-detection).

Please note that the 2nd place solution is not included, as it is not publicly available. We attempted to replicate it, but the performance did not match the original model.

The 7th place solution was successfully replicated with similar performance to the original model. All other model weights and codes are provided by the original authors.

## Models

The table below summarizes the models, their weights, references, and performance on the Kaggle competition's private test set, and external datasets from Toronto St Michael's Hospital (non-contrast and contrast CT):

| **Models** | **Weights**                                                                               | **References**                                                                                                  | **AUC - Private test** | **AUC - External Non-Contrast** | **AUC - External Contrast** |
| ---------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ---------------------- | ------------------------------- | --------------------------- |
| 1.Qishen   | [Download](https://www.kaggle.com/models/zixuanh/cspine-1st-place-solution-model-weights) | [Solutions](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/365115) | 0.97                   | 0.91                            | 0.94                        |
| 2.RAWE    | Download | [Solutions](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362619) | 0.96                   | --                            | --                       |
| 3.Darragh  | [Download](https://www.kaggle.com/models/zixuanh/cspine-3rd-place-solution-model-weights) | [Solutions](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362643) | 0.97                   | 0.90                            | 0.92                        |
| 4.Selim    | [Download](https://www.kaggle.com/models/zixuanh/cspine-4th-place-solution-model-weights) | [Solutions](https://github.com/selimsef/rsna_cervical_fracture/)                                               | 0.95                   | 0.91                            | 0.92                        |
| 5.Speedrun | [Download](https://www.kaggle.com/models/zixuanh/cspine-5th-place-solution-model-weights) | [Solutions](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/363232) | 0.95                   | 0.79                            | 0.76                        |
| 6.Skecherz  | [Download](https://www.kaggle.com/models/zixuanh/cspine-6th-place-solution-model-weights) | [Solutions](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362651) | 0.95                   | 0.87                            | 0.87                        |
| 7.QWER  | [Download](https://www.kaggle.com/models/zixuanh/cspine-7th-place-solution-model-weights) | [Solutions](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/364848) | 0.96                   | 0.92                            | 0.91                        |
| 8.Harshit  | [Download](https://www.kaggle.com/models/zixuanh/cspine-8th-place-solution-model-weights) | [Solutions](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362669) | 0.95                   | 0.90                            | 0.87                        |

## Usage

To run the inference script on your dataset:

1. Download the model weights from the links provided in the table above.
2. Extract the weights to the corresponding solution folder.
3. Set the input path at the beginning of the inference script:

```python
PROJECT_FOLDER = "YOUR_PROJECT_FOLDER"  # Parent folder of the input images
IMAGE_DATA_FOLDER = PROJECT_FOLDER + "images/"  # Folder containing the input images
INPUT_TEST_CSV_FILE = "YOUR_TEST_FILE"  # CSV file listing the test case paths (DICOM)
OUTPUT_FOLDER = "YOUR_OUTPUT_FOLDER"  # Folder to save the output images
OUTPUT_CSV_FILE = "YOUR_OUTPUT_CSV_FILE"  # CSV file to save output predictions
```

- `INPUT_TEST_CSV_FILE` should contain the following columns:
  - `StudyInstanceUID`: A unique identifier for each study.
  - `image_folder`: A folder containing the DICOM images for each study (one exam per folder).
<br/><br/>

4. Run the inference script.
5. The script will output predictions in a CSV file.

## Grad-CAM Visualization

Grad-CAM visualizations are available for the 1st, 3rd, 4th, and 7th place solutions. These visualizations were generated using the [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam) library.

To generate the Grad-CAM visualizations, run the corresponding Grad-CAM script. The Grad-CAM images will be saved in the `OUTPUT_FOLDER` specified in the inference script.

## Citation

If you find this repository helpful, please consider citing our [paper](https://pubs.rsna.org/doi/10.1148/ryai.230550) and the original authors of the models.

```
@article{hu2024assessing, 
  title     = {Assessing the Performance of Models from the 2022 RSNA Cervical Spine Fracture Detection Competition at a Level I Trauma Center},
  author    = {Hu, Zixuan and Patel, Markand and Ball, Ball L. and Lin, Hui Ming and Prevedello, Luciano M. and Naseri, Mitra and Mathur, Shobhit and Moreland, Robert and Wilson, Jefferson and Witiw, Christopher and Yeom, Kristen W. and Ha, Qishen and Hanley, Darragh and Seferbekov, Selim and Chen, Hao and Singer, Philipp and Henkel, Christof and Pfeiffer, Pascal and Pan, Ian and Sheoran, Harshit and Li, Wuqi and Flanders, Adam E. and Kitamura, Felipe C. and Richards, Tyler and Talbott, Jason and Sejdić, Ervin and Colak, Errol},
  journal   = {Radiology: Artificial Intelligence},
  year      = {2024},
  doi       = {10.1148/ryai.230550},
  url       = {https://pubs.rsna.org/doi/10.1148/ryai.230550}
}
```

## Acknowledgments

We would like to extend our sincere thanks to the RSNA for organizing the 2022 Cervical Spine Fracture Detection Challenge, and to all the model authors for sharing their solutions. A special thank you to our colleagues at Toronto St. Michael’s Hospital, Dr. Colak, Huiming, and Robyn, for their invaluable support and collaboration throughout this project.