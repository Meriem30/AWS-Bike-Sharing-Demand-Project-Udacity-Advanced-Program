# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### ARBAOUI MERIEM
>Here is the notebook page of all the steps accomplished though this project [index](https://meriem30.github.io/AWS-Bike-Sharing-Demand-Project-Udacity-Advanced-Program/index.html). 

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
The first submission was accepted on the first try because there were no negative values in the predictions (although I had added code to clip them just in case). However, the submission process failed when I attempted it directly from the SageMaker notebook. Instead, I successfully submitted the predictions manually through the Kaggle platform.

### What was the top ranked model that performed?
The top-ranked model was WeightedEnsemble_L3. This is an ensemble of several base models trained in earlier layers. AutoGluon automatically created and tuned this ensemble, and it consistently outperformed individual models in terms of root mean squared error (RMSE).

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
During the exploratory data analysis (EDA), I observed significant variations in bike rental demand based on both weather conditions. While the datetime field initially showed strong influence on demand, its format made it difficult to interpret directly. To address this, I extracted the hour from the datetime column and introduced it as a new feature. This proved highly effective, as bike rental patterns (expected to) clearly follow a daily cycle.
In addition, I converted the season and weather columns from numerical values to categorical types to help the model treat them as nominal variables, not ordinal numbers.

### How much better did your model preform after adding additional features and why do you think that is?
The RMSE improved significantly after adding the hour feature and categoricals:

* Initial model score: 1.79478
* After feature engineering: 0.62064

This represents a 65.4% improvement in RMSE. The improvement is largely due to the addition of the hour feature, which captured daily demand cycles more effectively.

Treating season and weather as categorical variables also allowed the model to interpret these features more accurately, improving its ability to generalize and make better predictions.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
Hyperparameter tuning with num_trials=10 and targeted model selection using a specific set of algorithms—LightGBM (GBM), Random Forest (RF), Extra Trees (XT), and Neural Network with Torch (NN_TORCH), resulted in further performance gains.

* Score after feature engineering: 0.62064
* Score after hyperparameter tuning: 0.47362

This reflects a 23.7% improvement over the feature-engineered model. The improvement likely stems from focusing on strong model families (GBM, RF, XT, NN_TORCH), and avoiding wasting resources on weaker models and improved ensemble quality. Additionally, hyperparameter tuning with num_trials=10 allowed AutoGluon to optimize key settings, resulting in better generalization and a lower RMSE.

### If you were given more time with this dataset, where do you think you would spend more time?
* I would experiment more with time-based features, like is_weekend, rush_hour, dayofweek, or also month.
* I would also try XGBoost and CatBoost with advanced custom hyperparameters, and run more extensive HPO with more number of trails

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|none|none|none|1.79478|
|add_features|hour added|categoricals|none|0.62064|
|hpo|num_trials=10|search=bayesopt|GBM+RF+XT+NN|0.47362|

### Create a line plot showing the top model score for the three (or more) training runs during the project.



![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.



![model_test_score.png](img/model_test_score.png)

## Summary
This project demonstrated the power of AutoGluon in rapidly training high-performance tabular models. The most important boost in performance came from feature engineering, especially extracting the hour from datetime and converting variables to categorical types. Hyperparameter tuning gave good improvements but not as good as the feature engineering. Overall, AutoGluon's default ensembling, stacking, and model selection proved very effective out-of-the-box.

For further improvement, I would explore richer features and spend more time on tuning individual models using hyperparameters directly.
