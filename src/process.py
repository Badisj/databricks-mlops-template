from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
import argparse
import logging

# -------------------
# Configure logging
# -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("process")

# -------------------
# Process function
# -------------------
def process(args):
    """
    Splits an input table into train, validation, and test datasets and writes them
    back to the specified tables in the given catalog and schema.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments containing catalog, schema, input table, and output table names.

        Attributes:
        - catalog : str, Unity Catalog name
        - schema : str, target schema name
        - input_table : str, source table name
        - train_table : str, train output table
        - val_table : str, validation output table
        - test_table : str, test output table

    Returns
    -------
    None
    """
    logger.info("Starting data processing...")

    spark = SparkSession.builder.getOrCreate()
    logger.info("SparkSession created")

    input_table = f"{args.catalog}.{args.schema}.{args.input_table}"
    train_table = f"{args.catalog}.{args.schema}.{args.train_table}"
    val_table = f"{args.catalog}.{args.schema}.{args.val_table}"
    test_table = f"{args.catalog}.{args.schema}.{args.test_table}"

    logger.info(f"Reading input table: {input_table}")
    df = spark.table(input_table)

    logger.info("Splitting dataset into train/validation/test sets")
    df = df.withColumn("split_rand", rand(seed=42))

    train_df = df.filter("split_rand < 0.7").drop("split_rand")
    val_df = df.filter("split_rand >= 0.7 AND split_rand < 0.85").drop("split_rand")
    test_df = df.filter("split_rand >= 0.85").drop("split_rand")

    # Ensure schema exists
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {args.catalog}.{args.schema}")

    logger.info(f"Writing train table: {train_table}")
    train_df.write.format("delta").mode("overwrite").saveAsTable(train_table)

    logger.info(f"Writing validation table: {val_table}")
    val_df.write.format("delta").mode("overwrite").saveAsTable(val_table)

    logger.info(f"Writing test table: {test_table}")
    test_df.write.format("delta").mode("overwrite").saveAsTable(test_table)

    logger.info("Data processing completed successfully")

# -------------------
# Script Entry Point
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split an input table into train, validation, and test tables"
    )

    # ===== DS INPUT =====
    parser.add_argument("--catalog", required=True, help="Unity Catalog name")
    parser.add_argument("--schema", required=True, help="Schema name")
    parser.add_argument("--input_table", required=True, help="Input table name")
    parser.add_argument("--train_table", required=True, help="Train output table")
    parser.add_argument("--val_table", required=True, help="Validation output table")
    parser.add_argument("--test_table", required=True, help="Test output table")
    # ===================

    args = parser.parse_args()
    process(args)
