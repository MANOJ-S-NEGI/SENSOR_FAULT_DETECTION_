from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from sensor_main_dir.entity.artifect_entity_dataclass import ClassificationMetricArtifact
from sensor_main_dir.logger_dir import logging


def calculate_metric(model, x, y):
    """
    model: estimator
    x: input feature
    y: output feature
    """
    logging.info("calling Classification_Metric_Artifact")
    y_pred = model.predict(x)
    Classification_Metric_Artifact = ClassificationMetricArtifact(
        f1_score=f1_score(y, y_pred),
        precision_score=precision_score(y, y_pred),
        recall_score=recall_score(y, y_pred))
    return Classification_Metric_Artifact


def total_cost(y_true, y_pred):
    """
    This function takes y_ture, y_predicted, and prints Total cost due to misclassification
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  # .ravel() provide flatten 1D
    cost = 10 * fp + 500 * fn
    return cost





