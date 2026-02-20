import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession
from sklearn.linear_model import LinearRegression
import argparse
import logging


# -------------------
# Configure logging
# -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("train")


# -------------------
# Train function
# -------------------
def train(args):
    """
    Train a linear regression model on the training dataset, validate it on
    the validation dataset, log metrics to MLflow, and register the model.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments containing catalog, schema, train/validation table names, and model name.

        Attributes:
        - catalog : str, Unity Catalog name
        - schema : str, schema name
        - train_table : str, training table name
        - val_table : str, validation table name
        - model_name : str, MLflow model registry name

    Returns
    -------
    None
    """
    logger.info("Starting model training...")

    spark = SparkSession.builder.getOrCreate()
    logger.info("SparkSession created")

    train_table_full = f"{args.catalog}.{args.schema}.{args.train_table}"
    val_table_full = f"{args.catalog}.{args.schema}.{args.val_table}"

    logger.info(f"Loading train table: {train_table_full}")
    train_df = spark.table(train_table_full).toPandas()

    logger.info(f"Loading validation table: {val_table_full}")
    val_df = spark.table(val_table_full).toPandas()

    X_train, y_train = train_df[["x"]], train_df["y"]
    X_val, y_val = val_df[["x"]], val_df["y"]

    model = LinearRegression()
    logger.info("LinearRegression model instantiated")

    with mlflow.start_run() as run:
        logger.info(f"MLflow run started: {run.info.run_id}")

        logger.info("Fitting model on training data")
        model.fit(X_train, y_train)

        val_score = model.score(X_val, y_val)
        logger.info(f"Validation R^2 score: {val_score:.4f}")

        mlflow.log_param("algorithm", "linear_regression")
        mlflow.log_metric("val_r2", val_score)
        logger.info("Logging model to MLflow")
        mlflow.sklearn.log_model(model, "model")

        logger.info(f"Registering model as '{args.model_name}' in MLflow registry")
        mlflow.register_model(f"runs:/{run.info.run_id}/model", args.model_name)

    logger.info("Model training and registration completed successfully")


# -------------------
# Script Entry Point
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a linear regression model and register in MLflow")

    # ===== DS INPUT =====
    parser.add_argument("--catalog", required=True, help="Unity Catalog name")
    parser.add_argument("--schema", required=True, help="Schema name")
    parser.add_argument("--train_table", required=True, help="Training table name")
    parser.add_argument("--val_table", required=True, help="Validation table name")
    parser.add_argument("--model_name", required=True, help="MLflow model registry name")
    # ===================

    args = parser.parse_args()
    train(args)
