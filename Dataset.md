## Initial Dataset (30-Second Samples):

  - The original dataset contained 30-second samples for each track, providing a comprehensive representation of each song. Each track is categorized by genre and analyzed for spectral properties that are characteristic of different musical styles.

## Modified Dataset (3-Second Segments):

  - To enhance the dataset, each 30-second song sample was split into ten 3-second segments. This transformation:

    - Increased the dataset size, offering more training data.
   
    - Allowed the model to focus on shorter patterns within each genre, making it more adaptable and capable of generalizing across different musical samples.
Each 3-second segment was then used as an individual data point for training. The segmentation also enables the model to learn short-term characteristics, which are often more genre-specific than longer, blended soundscapes.

## Access:

The dataset can be downloaded from [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

Or instead of downloading, the dataset can directly be accessed in python. Refer to [this site](https://datasets.activeloop.ai/docs/ml/datasets/gtzan-genre-dataset/)




