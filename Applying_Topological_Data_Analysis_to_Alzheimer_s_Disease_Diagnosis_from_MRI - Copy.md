Applying Topological Data Analysis to Alzheimer’s Disease Diagnosis from MRI

Hugo Jal, Ravi Shah, Parth Parik

June 2024

1  Abstract

Alzheimer’s Disease (AD) represents a significant challenge in cognitive health, affecting millions of Americans. This study explores the application of Topo- logical Data Analysis (TDA) in conjunction with various deep learning mod- els for AD diagnosis using MRI data from the Alzheimer’s Disease Neu- roimaging Initiative (ADNI). We implemented and compared five models: a Random Forest Classifier, a Long Short-Term Memory (LSTM) network, a Linear TDA model using the Gudhi library, a Lasso regression model, and a Stochastic TDA model with persistence image. Our results demonstrate the potential of TDA in enhancing AD diagnosis accuracy. The study highlights the effectiveness of persistence images in improving model performance and the value of combining TDA with traditional machine learning approaches for more accurate and interpretable AD diagnosis.

2  Introduction

As the fifth-leading cause of death in adults over the age of 65 in the United States, Alzheimer’s Disease (AD) represents one of the most severe and detri- mental illnesses affecting cognitive and mental health [(National Center for Chronic Disease Prevention and Health Promotion, 2020). ](#_page11_x73.62_y643.74)Characterized by progressive cognitive decline, memory loss, and personality changes, AD is primarily driven by the abnormal accumulation of beta-amyloid plaques and tau protein tangles, alongside neuroinflammation and oxidative stress, lead- ing to synaptic dysfunction and brain atrophy [(Breijyeh & Karaman, 2020). ](#_page11_x73.62_y253.70)With projections estimating an increase in the number of individuals affected by AD from 5.8 million Americans to 14 million by 2060, it is imperative to invest in advanced imaging and organizational systems to accurately clas- sify the brain’s morphological changes and facilitate early mitigative action [(Matthews et al., 2019).](#_page11_x73.62_y484.83)

![](Aspose.Words.727933d5-d86f-4f00-82f3-c5460e9f9ad3.001.jpeg)

Figure 1: Age-Adjusted Death Rates for Alzheimer Disease Among Adults Aged ≥65 Years, by Sex — National Vital Statistics System, United States, 1999–2019

The relationship between cognitive impairment and the pathological le- sions observed in AD is complex, often overlapping with normal aging pro- cesses [(Wyss-Coray,](#_page12_x73.62_y357.43) [2016).](#_page12_x73.62_y357.43) Structural markers such as atrophy of medial temporal and hippocampal regions, identifiablevia magnetic resonance imag- ing (MRI), are crucial in the progression of AD [(Rao et al., 2022). ](#_page12_x73.62_y184.08)Ground- breaking research in recognizing age-related degenerative diseases has laid the foundations for continually advanced modes of identification and qualification to improve patient outcomes and therapeutic interventions. Despite exist- ing methods to identify and categorize neurodegenerative age-related dis- eases, high-dimensional MRI analysis combined with topological data analy- sis (TDA) remains an underexplored area in neuroscientific research [(Chazal](#_page11_x73.62_y311.48)

- [Michel,](#_page11_x73.62_y311.48) [2021;](#_page11_x73.62_y311.48)[ Singh et al.,](#_page12_x73.62_y241.86) [2023).](#_page12_x73.62_y241.86)

TDA is a mathematical framework that extracts geometric and topolog- ical features from complex datasets, allowing researchers to discern granular changes in brain morphology associated with AD progression. This approach captures both global and local features simultaneously, complementing tra- ditional methods like Convolutional Neural Networks (CNNs). TDA holds promise for earlier and more accurate diagnosis, potentially slowing disease progression and improving patient outcomes through timely interventions tailored to individual needs.

To investigate the potential of TDA in enhancing the categorization of high-dimensional MRIs in AD patients, we have developed several deep- learning algorithms, including a Random Forest classifier,a Long Short-Term Memory network, a linear TDA model, and a stochastic TDA model. This research paper will examine our findings and explore the role TDA can play in advancing Alzheimer’s Disease MRI categorization.

3  Background

Since the introduction of deep learning algorithms in medicine in the early 2000s, such algorithms have grown in use in Alzheimer’s Disease research in a variety of facets, including prognosis, predictive analysis, and identification of biomarkers. Deep learning algorithms, including artificialneural networks, convolutional neural networks, and recurrent neural networks, are exceptional at extracting information from high-dimensional data such as MRIs, thus making such tools crucial for the development of research in Alzheimer’s Disease.

Our introduction to the possibilities that TDA was able to offer at the time of analyzing highly dimensional medical imaging data was brought to us by seeing how such algorithms were able to detect biomarkers for basal cell carcinoma (i.e., presence of telangiectasia within the skin lesions) and achieve an accuracy of 97.4% [(Maurya et al., 2024). ](#_page11_x73.62_y585.95)However, to this day, very little to almost no research has been published on the potential of using topological data analysis for Alzheimer’s Disease identification. During the course of this research project, Harshitha Bingi and T. Sobha Rani published \*Identification of Onset and Progression of Alzheimer’s Disease Using Topo- logical Data Analysis\*, which they were kind enough to share a copy with us to further our understanding of the subject.

4  Methods

In this study, we conducted a comprehensive comparative analysis of five distinct deep learning models to gain deeper insights into their respective performances. The built models encompassed a Random Forest Classifier, a Long Short-Term Memory (LSTM) network, a Linear Topological Data Analysis (TDA) model implemented with Gudhi’s Python library, and a Stochastic TDA model incorporating persistence image.

The dataset utilized for our investigation was sourced from the Alzheimer’s Disease Neuroimaging Initiative (ADNI), specifically the ADNI1: Complete 1Yr 1.5T dataset. The ADNI dataset is renowned for its comprehensive and longitudinal neuroimaging data, making it an ideal choice for studying Alzheimer’s disease progression. Said dataset contains subjects who have screening, 6 and 12-month scans.

1. Linear Models

The linear model utilizes Lasso regression, a form of linear regression that is regularized with an L1 penalty, for predictive analysis.

wˆ = argmin (MSE(w) + ∥w∥1)

w

Lasso regression happens to be particularly useful in machine learning so as to handle high dimensional data since it allows for a facilitated automatic feature selection. Specifically, the model undergoes training and evaluation with two separate datasets: one created from persistence diagrams converted into persistence images, and the other containing original input data.

![](Aspose.Words.727933d5-d86f-4f00-82f3-c5460e9f9ad3.002.png)

Figure 2: Persistence images for the first four samples

At first, the dataset is separated into training and testing sets with a 70-30 split ratio and a specified random state to guarantee reproducibility. Afterward, the training data is used to instantiate and fitthe Lasso regression model. Lasso regression penalizes the absolute magnitude of regression coef- ficients, encouraging sparsity in the model and potentially enhancing inter- pretability by choosing important features. Moreover, the model undergoes training with defined hyperparameters such as the regularization parameter (λ) and the maximum iteration limit for optimization.

Ultimately, the model is assessed for performance using the mean squared error and the coefficient of determination. The results yielded a 93.2% accu- racy.

![](Aspose.Words.727933d5-d86f-4f00-82f3-c5460e9f9ad3.003.jpeg)

Figure 3: Graph showcasing the Lasso Regression model with persistence homol-

ogy features inputted.

In contrast, the model trained without persistence homology features per- formed significantly worse; yielding a 69% accuracy rate.

2. Linear TDA with Gudhi Library

Linear Topological Data Analysis (TDA) has emerged as a powerful tool in the fieldof medical imaging analysis, particularly in the context of Alzheimer’s disease diagnosis. In this study, we implemented a model using the Gudhi li- brary to conduct linear TDA on MRI images for Alzheimer’s diagnosis. This methodology was chosen for its ability to extract topological features from complex data structures, such as MRI images, and provide valuable insights into the underlying geometric and topological properties of the data.

By leveraging the Gudhi library, we built simplicial complexes from the MRI data, allowing us to capture the intrinsic geometric and topological information embedded in the images [(Bingi & Rani, 2024). ](#_page11_x73.62_y210.36)The latter is a crucial step for identifying key topological features, such as persistent homol- ogy, which are indicative of structural changes associated with Alzheimer’s disease progression.

![](Aspose.Words.727933d5-d86f-4f00-82f3-c5460e9f9ad3.004.png)

Figure 4: Illustration of the first few types of simplicial complexes.

Regarding the data split, we again opted for a 70-30 split and applied a Lasso Regression model to the MRI images preprocessed with persistence images. The accuracy score yielded was 83.9%.

3. Random Forest Classifier

Random forest classifiers are a technique for classification and regression that involve systematically choosing subsets of features from the feature vec- tor to construct trees in random subspaces: {h(x,k),k = 1,...} where the {θk} are independent identically distributed random vectors and each tree casts a unit vote for the most popular class at input x (Ho,[ 1998; ](#_page11_x73.62_y441.50)[Rammal et al.,](#_page12_x73.62_y126.30) [2022).](#_page12_x73.62_y126.30) Given that these types of classifiers consist of a collection of decision trees that are trained with different data subsets and then averaged, they are capable of being tolerant of the problem of overfitting (V[elazquez, Lee, & Alzheimer’s Disease Neuroimaging Initiative, 2021).](#_page12_x73.62_y299.65)

he utilized pipeline incorporates persistence diagrams to address the topo- logical complexity of analyzing biomedical data, like MRI scans. Every data point in the dataset is processed to generate persistence diagrams; the birth and death values in the diagrams are adjusted to a specific range for data standardization. Afterward, the persistence diagrams are split into two cat- egories depending on their class labels, then the persistence diagrams within each group are converted into persistence images, which act as characteristic attributes of the data.

Persistence images are used as the primary data for classification pur- poses. The data is split into training and testing sets in the following man- ner: 70-30. The random forest is set with predetermined parameters such as a maximum depth of 2 and a constant random state of 0. The confusion matrix was implemented, and the accuracy was comprised as 100%.

![](Aspose.Words.727933d5-d86f-4f00-82f3-c5460e9f9ad3.005.png)

Figure 5: Random Forest Classifier Results

4. LSTM

The implementation of an LSTM (Long Short-Term Memory) model rep- resents a powerful approach for Alzheimer’s diagnosis from MRI images. LSTM, a type of recurrent neural network (RNN), is particularly well-suited for processing sequential data, such as the temporal information present in MRI scans.

The model we implemented is defined with a single hidden layer of 128 units. The input shape is set to match the length of the input sequence and a single feature dimension. The model is then compiled with the mean squared error loss function and the Adam optimizer, which is a popular choice for its adaptive learning rate and momentum-based updates.

During the training process, the model is fit to the training data using a batch size of 20 and a maximum of 100 epochs. To prevent overfitting, an EarlyStopping callback is used, which monitors the validation loss and stops the training if the validation loss does not improve for 50 consecutive epochs.

The results show that the LSTM model achieves excellent performance, with low MSE and high R2 score: 99.6%.

5. Stochastic TDA with Persistence Image

We applied Stochastic Topological Data Analysis with the aim of seeing how topological features in the MRI scans would behave across visits and whether

TDA can discriminate between mild cognitive impairment and Alzheimer’s.

Principal Components Analysis (PCA) is applied to TDA so as to find a lower-dimensional subspace onto which to project the data and minimize information loss [(Abdesselam, 2021).](#_page11_x73.62_y152.58) This is done via an unsupervised approach where a set of linearly uncorrelated variables (eigenvectors) and their corresponding values (eigenvalues) are computed by using an orthog- onal transformation (Singular Value Decomposition) [(Garcia, 2019). In](#_page11_x73.62_y354.82) our Stochastic TDA with Persistence Image, we applied PCA with 3 components.

Moreover, we applied t-SNE in order to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data. We opted for a perplexity value of 20 and 2 components. Following t-SNE, Uniform Manifold Approximation and Projec- tion (UMAP) was used. UMAP is another powerful dimensionality reduction technique that projects the data into two dimensions, thus preserving more of the global structure compared to t-SNE.

At this point, KepplerMapper reveals the shape and connectivity of the data just before a Random Forest is trained on the persistence images. As a result, we obtained a 94.85% accuracy score.

5  Results & Discussion

The Random Forest classifiermodel yielded an accuracy of 100% when evalu- ated on the test set. This model utilized persistence diagrams converted into persistence images to assist in the classification. The high accuracy indicates that the model effectively captures the complex topological features present in the MRI data. However, an 100% accuracy is not common and could be a sign of overfitting or a bad test set quality. Through our calculations, we determined that our model most likely did not overfit and something else may have occurred.

The Lasso regression model trained on persistence images achieved an accuracy of 93.2%, whereas the model trained without persistence homology features only reached 69% accuracy. This difference shows the value of incor- porating topological features derived from TDA and persistence homology in enhancing our specific model performance.

The Linear TDA model, implemented using the Gudhi library, achieved

an accuracy of 83.9%. On the other hand, the LSTM model exhibited an excellent performance with an accuracy of 99.6%. We used the temporal information present in the MRI scans and the LSTM’s job of handling se- quential data to accurately diagnose AD.

The Stochastic TDA model yielded an accuracy of 94.85%. The use of this model allowed for the effective visualization and analysis of the data. It is clear that Stochastic TDA performed the greatest out of the other topological models. The LSTM model performed the greatest without considering other problems. The Random Forest classifier achieved perfect accuracy, however it is unclear if any problems occurred.

Persistence images proved to be highly effective in improving the accuracy of different models. This shows that it is important to transform topological data into a format more compatible with different algorithms. Moreover, the LSTM model’s high accuracy also shows the value of using temporal infor- mation in MRI data. AD is a multifaceted disease, and LSTM’s ability to use sequential data makes it suited for the diagnosis of Alzheimers.

The application of PCA and t-SNE in the Stochastic TDA model aided in visualization and analysis of high-dimensional data. In this way, they re- duced the complexity of the data yet preserved essential topological features for more intuitive interpretations and insights.

Additionally, comparative analysis indicates that though TDA increases the benefit for all models, the choice of the model can be tailored according to specific research needs and characteristics of data. For instance, Random Forest and LSTM models would be preferred for their high accuracy, while the linear models with TDA offer valuable insights into the importance of features and model interpretability. Combining TDA with other data types, such as genetic or clinical data, may lead to a better accuracy and a greater understanding of AD. The development of topological diagnostic tools com- bined with other forms of machine learning could help in the early detection of AD progression. The exploration of further techniques in topological data analysis and the use with other machine learning algorithms can unlock new possibilities in the diagnosis of AD and potentially other diseases.

6  Acknowledgements

Data collection and sharing for this project were funded by the Alzheimer’s Disease Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904) and DOD ADNI (Department of Defense award number W81XWH-12-2-0012). ADNI is funded by the National Institute on Ag- ing, the National Institute of Biomedical Imaging and Bioengineering, and through generous contributions from the following: AbbVie, Alzheimer’s As- sociation; Alzheimer’s Drug Discovery Foundation; Araclon Biotech; BioClin- ica, Inc.; Biogen; Bristol-Myers Squibb Company; CereSpir, Inc.; Cogstate; Eisai Inc.; Elan Pharmaceuticals, Inc.; Eli Lilly and Company; EuroImmun; F. Hoffmann-La Roche Ltd and its affiliated company Genentech, Inc.; Fu- jirebio; GE Healthcare; IXICO Ltd.; Janssen Alzheimer Immunotherapy Re- search & Development, LLC.; Johnson & Johnson Pharmaceutical Research

- Development LLC.; Lumosity; Lundbeck; Merck & Co., Inc.; Meso Scale Diagnostics, LLC.; NeuroRx Research; Neurotrack Technologies; Novartis Pharmaceuticals Corporation; Pfizer Inc.; Piramal Imaging; Servier; Takeda Pharmaceutical Company; and Transition Therapeutics. The Canadian In- stitutes of Health Research are providing funds to support ADNI clinical sites in Canada. Private sector contributions are facilitated by the Foundation for the National Institutes of Health (www.fnih.org). The grantee organization is the Northern California Institute for Research and Education, and the study is coordinated by the Alzheimer’s Therapeutic Research Institute at the University of Southern California. ADNI data are disseminated by the Laboratory for Neuro Imaging at the University of Southern California.

  The authors would like to thank Dr. Matthew Leming for his guidance on choosing an appropriate methodology and for his assistance in accessing the ADNI dataset.

  References

  <a name="_page11_x73.62_y152.58"></a>Abdesselam, R. (2021). A topological approach of principal component

analysis. International Journal of Data Science and Analysis, 7(2), 20–31. Retrieved from [https://hal.science/hal-03205861/ ](https://hal.science/hal-03205861/)doi: 10.11648/j.ijdsa.20210702.11

<a name="_page11_x73.62_y210.36"></a>Bingi, H., & Rani, T. (2024, 01). Identification of onset and progression

of alzheimer’s disease using topological data analysis. In (p. 193-205). doi: 10.1007/978-3-031-50583-6\_13

<a name="_page11_x73.62_y253.70"></a>Breijyeh, Z., & Karaman, R. (2020). Comprehensive review on alzheimer’s

disease: Causes and treatment. Molecules, 25(24), 5789. Retrieved from[ https://doi.org/10.3390/molecules25245789 ](https://doi.org/10.3390/molecules25245789)doi: 10.3390/ molecules25245789

<a name="_page11_x73.62_y311.48"></a>Chazal, F., & Michel, B. (2021). An introduction to topological data analysis:

Fundamental and practical aspects for data scientists. Frontiers in

Artificial Intelligence , 4, 667963. doi: 10.3389/frai.2021.667963
<a name="_page11_x73.62_y354.82"></a>Garcia, A. C. (2019). Study of brain imaging correlates of mild cognitive

impairment (mci) and alzheimer’s disease (ad) with machine learning

(Master’s thesis, Universitat Politècnica de Catalunya). Retrieved

from [https://upcommons.upc.edu/bitstream/handle/2117/](https://upcommons.upc.edu/bitstream/handle/2117/133055/anna_canal_garcia_UPC.pdf?sequence=1&isAllowed=y)

[133055/anna_canal_garcia_UPC.pdf?sequence=1&isAllowed=y](https://upcommons.upc.edu/bitstream/handle/2117/133055/anna_canal_garcia_UPC.pdf?sequence=1&isAllowed=y)

(Retrieved February 27, 2024)

<a name="_page11_x73.62_y441.50"></a>Ho, T. K. (1998). The random subspace method for constructing decision

forests. IEEE Transactions on Pattern Analysis and Machine Intelli- gence, 20(8), 832–844. doi: 10.1109/34.709601

<a name="_page11_x73.62_y484.83"></a>Matthews, K. A., Xu, W., Gaglioti, A. H., Holt, J. B., Croft, J. B.,

Mack, D., & McGuire, L. C. (2019). Racial and ethnic estimates of alzheimer’s disease and related dementias in the united states (2015–2060) in adults aged 65 years. Alzheimer’s & Dementia , 15(1), 17-24. Retrieved from [https://alz-journals.onlinelibrary.wiley .com/doi/abs/10.1016/j.jalz.2018.06.3063 ](https://alz-journals.onlinelibrary.wiley.com/doi/abs/10.1016/j.jalz.2018.06.3063)doi: https://doi.org/ 10.1016/j.jalz.2018.06.3063

<a name="_page11_x73.62_y585.95"></a>Maurya, A., Stanley, R. J., Lama, N., Nambisan, A. K., Patel, G., Saeed, D.,

... Stoecker, W. V. (2024). Hybrid topological data analysis and deep

learning for basal cell carcinoma diagnosis. Journal of Imaging Infor-

matics in Medicine , 37(1), 92–106. doi: 10.1007/s10278-023-00924-8 <a name="_page11_x73.62_y643.74"></a>National Center for Chronic Disease Prevention and Health Promotion.

(2020). Alzheimer’s disease and related dementias. [https://www.cdc](https://www.cdc.gov/aging/aginginfo/alzheimers.htm)

[.gov/aging/aginginfo/alzheimers.htm. (Last ](https://www.cdc.gov/aging/aginginfo/alzheimers.htm)reviewed: October

26, 2020. Source: Division of Population Health, National Center for

Chronic Disease Prevention and Health Promotion)

<a name="_page12_x73.62_y126.30"></a>Rammal, A., Assaf, R., Goupil, A., Boudaoud, S., El Rafei, M., & El Hajj,

A. (2022). Machine learning techniques on homological persistence features for prostate cancer diagnosis. BMC Bioinformatics, 23, 476. doi: 10.1186/s12859-022-04992-5

<a name="_page12_x73.62_y184.08"></a>Rao, Y. L., Ganaraja, B., Murlimanju, B. V., Joy, T., Krishnamurthy, A., &

Agrawal, A. (2022). Hippocampus and its involvement in alzheimer’s disease: a review. 3 Biotech, 12(2), 55. doi: 10.1007/s13205-022-03123 -4

<a name="_page12_x73.62_y241.86"></a>Singh, Y., Farrelly, C. M., Hathaway, Q. A., Leiner, T., Jagtap, J., Carlsson,

G. E., & Erickson, B. J. (2023). Topological data analysis in medical imaging: current state of the art. Insights into Imaging , 14(1), 58. doi: 10.1186/s13244-023-01413-w

<a name="_page12_x73.62_y299.65"></a>Velazquez, M., Lee, Y., & Alzheimer’s Disease Neuroimaging Initiative.

(2021). Random forest model for feature-based Alzheimer’s disease

conversion prediction from early mild cognitive impairment subjects.

PLoS ONE , 16(4), e0244773. doi: 10.1371/journal.pone.0244773 <a name="_page12_x73.62_y357.43"></a>Wyss-Coray, T. (2016). Ageing, neurodegeneration and brain rejuvenation.

Nature , 539(7628), 180–186. doi: 10.1038/nature20411
13
