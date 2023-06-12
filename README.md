### Contents:
- [Problem Stament](#Background)
- [File Structure](#Data-Import-and-Cleaning)
- [Semantic Segmentation Data](#Semantic-Segmentation-Data)
- [U-Net Model](#U-Net-Model)
- [Model Analysis](#Model-Analysis)
- [Conclusion](#Conclusion)

# Problem Statement

One of the highest costs in package delivery is refered to as 'last-mile delivery'. This referes to the local package delivery once a product has arrived at a distribution center. To address this need, a new start-up plans to use drone delivery for packages. 

As a data scientist for this start-up, I have been tasking with helping to create the model that will determine the zones were the drone can descend and unload the packages. 

I will be using the drone image dataset from the Institute of Computer Graphics and Vision ([source](https://www.tugraz.at/index.php?id=22387)) to train a model on image semantic segmentation. This will allow the drone to distinquish which surfaces to land on through its cameras. In the evaluation of the model, I will focus on its score for the following surfaces:
- Paved Area
- Gravel
- Dirt
- Grass


# File Structure

```
├── presentation.pdf
├── README.md
├── module                                          # Contains scripts to use custom functions
├── notebooks                                       # Contains notebooks for modeling workflow
    ├──EDA.ipynb
    ├──Modeling.ipynb
    └──Analysis.ipynb
├── imgs                                            # Collection of visualizations from modeling
└── semantic_drone_dataset                          # Includes original and processed images and masks
```

# Semantic Segmentation Data

# U-Net Model

# Model Analysis

# Conclusion


