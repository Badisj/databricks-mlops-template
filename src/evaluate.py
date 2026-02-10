import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession
import argparse
import logging

# -------------------
# Configure logging
# -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("evaluate")

# -------------------
# Evaluate function
# -------------------
def evaluate(args):
    """
    Evaluate the latest version of a registered MLflow model on a test dataset
    and log the evaluation metrics in MLflow.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments containing catalog, schema, test table, and MLflow model name.

        Attributes:
        - catalog : str, Unity Catalog name
        - schema : str, schema name
        - test_table : str, test dataset table name
        - model_name : str, MLflow model registry name

    Returns
    -------
    None
    """
    logger.info("Starting model evaluation...")

    spark = SparkSession.builder.getOrCreate()
    logger.info("SparkSession created")

    test_table_full = f"{args.catalog}.{args.schema}.{args.test_table}"
    logger.info(f"Loading test table: {test_table_full}")
    test_df = spark.table(test_table_full).toPandas()

    logger.info(f"Fetching latest model version for '{args.model_name}'")
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{args.model_name}'")
    if not versions:
        logger.error(f"No versions found for model '{args.model_name}'")
        raise ValueError(f"No versions found for model '{args.model_name}'")

    latest = max(versions, key=lambda v: int(v.version))
    logger.info(f"Latest model version: {latest.version}")

    model_uri = f"models:/{args.model_name}/{latest.version}"
    logger.info(f"Loading model from URI: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    X_test, y_test = test_df[["x"]], test_df["y"]
    score = model.score(X_test, y_test)
    logger.info(f"Test R^2 score: {score:.4f}")

    with mlflow.start_run() as run:
        logger.info(f"Logging test metric to MLflow run {run.info.run_id}")
        mlflow.log_metric("test_r2", score)
        mlflow.set_tag("evaluated_model_version", latest.version)

    logger.info("Model evaluation completed successfully")

# -------------------
# Script Entry Point
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MLflow model on test dataset")

    # ===== DS INPUT =====
    parser.add_argument("--catalog", required=True, help="Unity Catalog name")
    parser.add_argument("--schema", required=True, help="Schema name")
    parser.add_argument("--test_table", required=True, help="Test table name")
    parser.add_argument("--model_name", required=True, help="MLflow model registry name")
    # ===================

    args = parser.parse_args()
    evaluate(args)
