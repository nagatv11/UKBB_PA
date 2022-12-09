# Predicting successful Physical activity behaviour change from fMRI: 
# Take rs-fMRI time series to compute a connectivity matrix
# for each participant. Use the connectivity matrix values as the input
# features to predict whether the participant meets the the WHO guidelines of
# 150 minutes of moderate activity per week or 75 minutes of vigorous activity.

from nilearn import datasets
from nilearn.maskers import NiftiMasker
from nilearn.image import load_img
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
from nilearn.maskers import NiftiMapsMasker

import numpy as np

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.dummy import DummyClassifier

def load_func_data_and_pa():
    """
    Load functional data and participants' PA activity (i.e their MVPA group) at follow-up
    
    Returns X, a list with an array of shape (n_subjects, n_rois) per participant, and y,
    an array of length n_participants containing integers indicating whether a person
    met the WHO guidelines of 150 minutes of moderate activity per week or 75 minutes of vigorous activity
    """
    atlas = datasets.fetch_atlas_schaefer_2018()
    
    # Loading atlas image stored in 'maps'
    atlas_filename = atlas['maps']
    
    # Loading atlas data stored in 'labels'
    labels = atlas['labels']
    
    # Load the functional datasets
    func_data = load_img("~/scratch/ukbb_cvd_dir/49*_filtered_func_data_clean.nii.gz")
    
    # Extract time series
    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,
                         memory='nilearn_cache', verbose=5)
   
    masker.fit(func_data)
    X = masker.transform(func_data)

    # Load physical activity behaviour change (whether or not they meet above the MVPA recommendations)
    df_PA = pd.read_csv("~/scratch/UKBB_PA_FU.csv")
    y = LabelEncoder().fit_transform(df_PA["MVPA_group"])
    return X, y

def prepare_pipelines():
    """Prepare scikit-learn pipelines for fmri classification with connectivity.

    Returns a dictionary where each value is a scikit-learn estimator (a
    `Pipeline`) and the corresponding key is a descriptive string for that
    estimator.

    """
    connectivity = ConnectivityMeasure(
        kind="correlation", vectorize=True, discard_diagonal=True
    )
    scaling = StandardScaler()
    logreg = LogisticRegressionCV(solver="liblinear", cv=3, Cs=3)
    logreg = LogisticRegression(C=10)
    logistic_reg = make_pipeline(
        clone(connectivity), clone(scaling), clone(logreg)
    )

    pca_logistic_reg = make_pipeline(
        clone(connectivity),
        clone(scaling),
        PCA(n_components=20),
        clone(logreg),
    )
    dummy = make_pipeline(clone(connectivity), DummyClassifier())
    return {
        "Logistic no PCA": logistic_reg,
        "Logistic with PCA": pca_logistic_reg,
        "Dummy": dummy,
    }

def compute_cv_scores(models, X, y):
    """Compute cross-validation scores for all models

    `models` is a dictionary of the form `{"model_name": estimator}`, where `estimator` is a
    scikit-learn estimator.

    `X` and `y` are the design matrix and the outputs to predict.

    Returns a `pd.DataFrame` with one row for each model and cross-validation
    fold. Columns include `test_score` and `fit_time`.

    """
    all_scores = []
    for model_name, model in models.items():
        print(f"Computing scores for model: '{model_name}'")
        model_scores = pd.DataFrame(cross_validate(model, X, y, return_train_score=True))
        model_scores["model"] = model_name
        all_scores.append(model_scores)
    all_scores = pd.concat(all_scores)
    return all_scores

if __name__ == "__main__":
    X, y = load_func_data_and_pa()
    models = prepare_pipelines()
    all_scores = compute_cv_scores(models, X, y)
    print(all_scores.groupby("model").mean())
    sns.stripplot(data=all_scores, x="train_score", y="model")
    plt.tight_layout()
    plt.show()