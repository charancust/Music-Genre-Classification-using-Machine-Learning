# Music-Genre-Classification-using-Machine-Learning
This project involves training various Machine Learning models to classify the genre of music using python.

## Abstract:

This project explores the classification of music genres using machine learning techniques in Python. We leverage various learning models, including K-Nearest Neighbors, Support Vector Machines and Digital Signal Processing techniques to classify music into genres based on audio features. The project demonstrates the effectiveness of these models on a widely-used dataset, showcasing accuracy, precision, recall, and F1-score metrics. Through a detailed analysis of the dataset and performance metrics, we provide insights into the strengths and limitations of each model.

## Pre-processing:

Data preprocessing involved the following steps:

1. Normalization and Scaling: To ensure consistency in feature values, all features were normalized and scaled to have zero mean and unit variance.

2. Feature Extraction: Critical features were extracted to represent the spectral, rhythmic, and harmonic characteristics of each audio segment. This included:

   - Short-Time Fourier Transform (STFT):

     - The STFT is a method for analyzing how the frequency content of a signal changes over time. It divides the audio signal into smaller chunks (windows) and applies the Fourier transform to each one. The mathematical representation of STFT is:
    
         - $$X(m, k) = \sum_{n = 0}^{N - 1} x(n)\cdot w(n - m)\cdot e^{-j2\pi kn/N}$$
    
       - $$x(n)$$ is the original audio signal.
      
       - $$w(n - m)$$ is the windowing function (eg. Hamming window).
      
       - $$N$$ is the number of samples in each window.
      
       - $$m$$ and $$k$$ are the time and frequency indices respectively.
      
      - The result is a time-frequency representation of the signal, enabling the identification of frequency patterns within each 3-second segment. STFT is especially useful for capturing the harmonic structure and texture of audio, which are essential for distinguishing genres.
  
    - Mel-Frequency Cepstral Coefficients (MFCC):
  
      - MFCCs are among the most commonly used features in audio classification. They compress audio data by representing sound on the Mel scale, which approximates human auditory perception. MFCCs are calculated as follows:

        - Fourier Transform: Converts the time-domain signal into the frequency domain.
       
        - Mel Scaling: Maps the frequency components to the Mel scale.
       
        - Log Transformation: Applies a logarithmic scale to the Mel-scaled data.
       
        - Discrete Cosine Transform (DCT): Reduces the dimensionality by focusing on the most relevant coefficients.
       
        - The MFCC calculation formula is:
       
          - $$C_{i} = \sum_{k = 1}^{K}log(S_{k})cos(\tfrac{\pi i(k - 0.5)}{K})$$
         
          - $$C_{i}$$ is the $$i^{th}$$ MFCC.
         
          - $$S_{k}$$ are the Mel-scaled fourier coefficients.
         
          - $$K$$ is the total number of Mel filters.
         
      - MFCCs provide a compact and effective representation of the timbral qualities of audio, enabling the model to distinguish between different tonal patterns characteristic of each genre.
     
    - Chroma frequencies:
  
      - Chroma frequencies capture pitch classes across an octave. They are effective in representing harmonic content, especially for genres where harmony is a key distinguishing feature.

    - Spectral contrast:
  
      - Spectral contrast measures the difference in amplitude between peaks and valleys in an audio spectrum. Genres with different instrumentation or dynamic ranges can exhibit distinctive spectral contrasts, aiding in classification.
     
  3. Model training:

     - Various machine learning models were evaluated, including:
    
       - Random Forest Classifier: An ensemble model based on decision trees, which provides robust results by reducing overfitting.
      
       - Support Vector Machine (SVM): Effective for high-dimensional data, it uses hyperplanes to classify segments based on extracted features.
      
       - K-Nearest Neighbors (KNN): A simple yet effective classifier that categorizes samples based on similarity to nearest neighbors.
      
       - Naive Bayes: A probabilistic classifier that assumes feature independence, often used as a baseline.
      
       - Gradient Boosting: An ensemble model that iteratively improves performance by combining weak learners.
      

        
  4. Evaluation metrics:

     - Accuracy: Overall proportion of correctly classified samples.
    
     - Precision, Recall, F1-Score: Key metrics for imbalanced classes, indicating classification quality.
    
     - ROC-AUC Score: Measures the area under the receiver operating characteristic curve, providing insight into overall model performance.
    
     - Random Forest Classifier's feature importance was used to identify the dominant features.
     
     - Cosine similarity: Cosine similarity is a commonly used metric in text and audio classification tasks to measure similarity between feature vectors, regardless of their size. In this project, cosine similarity likely serves as a way to assess how closely a segment of audio resembles another by quantifying the angle between their feature vectors in a multi-dimensional space. The cosine similarity between two vectors AAA and BBB is calculated as follows:
         - $$cosinesimilarity = \frac{A\cdot B}{||A||\cdot ||B||} = \tfrac{\sum_{i = 1}^{n}A_{i} \times B_{i}}{\sqrt{\sum_{i = 1}^{n}A_{i}^{2}}\times\sqrt{\sum_{i = 1}^{n}B_{i}^{2}}}$$
        
         - $$A\cdot B$$ is the dot product of the vectors $$A$$ and $$B$$.
        
         - $$||A||$$ and $$||B||$$ are the magnitudes (norms) of $$A$$ and $$B$$.
        
         - $$n$$ is the dimensionality of the vectors.
      
  5. Results analysis:

     - Model performance metrics:
    
       | Model                      | Accuracy | Precision | Recall | F1-Score |
       | -------------------------- | -------- | --------- | ------ | -------- |
       | KNN                        | 89%      | 90%       | 89%    | 89%      |
       | SVM                        | 81%      | 81%       | 82%    | 81%      |
       | Random Forest Classifier   | 81%      | 81%       | 81%    | 81%      |       
       | Decision Tree              | 50%      | 53%       | 21%    | 51%      |
       | Basic Perceptron           | 61%      | 62%       | 61%    | 61%      |

6. Conclusion:

   This project demonstrated that segmenting audio into shorter samples can effectively improve genre classification performance. Feature extraction techniques like MFCC and STFT were pivotal in capturing the unique spectral and harmonic properties of each genre, which facilitated model learning. The increased data size due to 3-second segments allowed models to capture finer distinctions between genres, leading to more robust and accurate classifications.
Further improvements could involve experimenting with deep learning architectures, such as Convolutional Neural Networks (CNNs), which have been successful in audio classification tasks.

      

    






   
    
    
  

