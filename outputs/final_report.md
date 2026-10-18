# Executive Experiment Report: Fake News Detection Pipeline
Generated: 2026-10-18 10:28:53 UTC

## 1. Dataset Partitioning
- Total Deduplicated Articles: 39,103
- Training Split (70%): 10,500
- Validation Split (15%): 2,250
- Held-Out Test Split (15%): 2,250

## 2. Held-Out Test Evaluation Summary
| Model                     |   Accuracy |   Balanced_Accuracy |   Precision |   Recall |   F1_Score |   Macro_F1 |   ROC_AUC |   PR_AUC |     MCC |   Latency_Per_1000_Queries_Sec |
|:--------------------------|-----------:|--------------------:|------------:|---------:|-----------:|-----------:|----------:|---------:|--------:|-------------------------------:|
| Linear_SVC                |     0.996  |              0.9959 |      0.9951 |   0.9975 |     0.9963 |     0.996  |    0.9996 |   0.9997 |  0.9919 |                         0.0007 |
| Tuned_Logistic_Regression |     0.9924 |              0.9922 |      0.991  |   0.9951 |     0.993  |     0.9924 |    0.9993 |   0.9994 |  0.9848 |                         0.0006 |
| Random_Forest             |     0.9907 |              0.9902 |      0.987  |   0.9959 |     0.9914 |     0.9906 |    0.9994 |   0.9995 |  0.9812 |                         0.0238 |
| Logistic_Regression       |     0.9884 |              0.9881 |      0.9862 |   0.9926 |     0.9894 |     0.9884 |    0.9985 |   0.9987 |  0.9767 |                         0.0006 |
| Multinomial_NB            |     0.9636 |              0.9634 |      0.9679 |   0.9648 |     0.9663 |     0.9633 |    0.9927 |   0.9937 |  0.9266 |                         0.001  |
| Dummy_Stratified          |     0.476  |              0.4728 |      0.517  |   0.5107 |     0.5138 |     0.4728 |    0.4728 |   0.5293 | -0.0543 |                         0.0002 |

## 3. Persistent Artifacts & Deliverables
- Model Pipeline: `artifacts/models/fake_news_logistic_pipeline.joblib`
- Data Quality Report: `outputs/data_quality_report.csv`
- Model Comparison: `outputs/model_comparison.csv`
- Misclassified Examples: `outputs/misclassified_examples.csv`
- SQLite Audit Database: `outputs/fake_news.db`
