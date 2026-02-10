import argparse
import logging
from mlflow.deployments import get_deploy_client

# -------------------
# Configure logging
# -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("serve")

def deploy_model(
    model_name: str,
    endpoint_name: str,
    strategy: str = "first",
    traffic: int = None,
    enable_data_capture: bool = True
):
    """
    Deploy a registered MLflow model to a Databricks endpoint with deployment strategies.
    """
    client = get_deploy_client("databricks")

    # Check existing deployment
    try:
        existing = client.get_deployment(endpoint_name)
        endpoint_exists = True
        logger.info(f"Existing deployment found for endpoint '{endpoint_name}'")
    except Exception:
        existing = None
        endpoint_exists = False
        logger.info(f"No existing deployment found for endpoint '{endpoint_name}'")

    strategy = strategy.lower()

    # -------------------
    # First deployment
    # -------------------
    if strategy == "first":
        if not endpoint_exists:
            logger.info("Deploying first model to new endpoint...")
            client.create_deployment(
                name=endpoint_name,
                model_uri=f"models:/{model_name}/Staging",
                mode="online",
                traffic=100
            )
        else:
            logger.error("Endpoint already exists. Use shadow/canary/AB/bluegreen strategy")
            raise ValueError("Endpoint already exists. Use another strategy.")

    # -------------------
    # Shadow deployment
    # -------------------
    elif strategy == "shadow":
        if endpoint_exists:
            logger.info("Deploying new model in shadow mode (mirror traffic, no impact to production)")
            client.create_deployment(
                name=endpoint_name,
                model_uri=f"models:/{model_name}/Staging",
                mode="shadow"
            )
        else:
            logger.info("No existing endpoint. Deploying as first online deployment with 0 traffic")
            client.create_deployment(
                name=endpoint_name,
                model_uri=f"models:/{model_name}/Staging",
                mode="online",
                traffic=0
            )

    # -------------------
    # Canary / AB deployment
    # -------------------
    elif strategy in ["canary", "ab"]:
        if not endpoint_exists:
            logger.error(f"Cannot deploy {strategy} model because no existing endpoint found")
            raise ValueError(f"Cannot deploy {strategy} model because no existing endpoint found")

        if traffic is None:
            traffic = 10 if strategy == "canary" else 50

        logger.info(f"Updating endpoint to add new model with {traffic}% traffic while keeping existing model(s)")
        client.update_deployment(
            name=endpoint_name,
            model_uri=f"models:/{model_name}/Staging",
            traffic=traffic
        )

    # -------------------
    # Blue-Green deployment
    # -------------------
    elif strategy == "bluegreen":
        if endpoint_exists:
            logger.info("Switching all traffic to new model (blue-green promotion)")
            client.update_deployment(
                name=endpoint_name,
                model_uri=f"models:/{model_name}/Staging",
                traffic=100
            )
        else:
            logger.info("No existing endpoint. Deploying new model as full online deployment")
            client.create_deployment(
                name=endpoint_name,
                model_uri=f"models:/{model_name}/Staging",
                mode="online",
                traffic=100
            )

    else:
        logger.error(f"Unknown deployment strategy: {strategy}")
        raise ValueError(f"Unknown deployment strategy: {strategy}")

    # -------------------
    # Enable data capture
    # -------------------
    if enable_data_capture:
        logger.info("Enabling input/output data capture for monitoring and drift detection")
        client.update_endpoint(
            name=endpoint_name,
            served_models=[{
                "model_name": model_name,
                "model_version": "latest",
                "enable_capture": True
            }]
        )

    logger.info(f"Deployment of '{model_name}' with strategy '{strategy}' completed.")


# -------------------
# Script Entry Point
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy MLflow model to Databricks endpoint with strategy")
    parser.add_argument("--model_name", required=True, help="MLflow model name in registry")
    parser.add_argument("--endpoint_name", required=True, help="Databricks serving endpoint name")
    parser.add_argument("--strategy", default="first",
                        help="Deployment strategy: first/shadow/canary/ab/bluegreen")
    parser.add_argument("--traffic", type=int, help="Traffic percentage for canary/AB")
    parser.add_argument("--enable_data_capture", type=bool, default=True,
                        help="Enable input/output data capture")

    args = parser.parse_args()

    deploy_model(
        model_name=args.model_name,
        endpoint_name=args.endpoint_name,
        strategy=args.strategy,
        traffic=args.traffic,
        enable_data_capture=args.enable_data_capture
    )
