import io

import pandas as pd
from flask import Flask, render_template, request
from loguru import logger

from app.model_service import ModelService
from src.config.features import FEATURE_COLUMNS
from src.config.logging_config import setup_logger


def create_routes(app: Flask) -> None:
    """Create all routes for the application."""

    @app.route("/")
    def index() -> str:
        """Serve the HTML upload interface."""
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload() -> str:
        """Handle CSV file upload, validate features, and return predictions."""
        file = request.files["file"]
        if not file.filename or not file.filename.endswith(".csv"):
            return render_template("index.html", error="Please upload a CSV file")

        try:
            # Read CSV content
            content = file.read().decode("utf-8")
            features = pd.read_csv(io.StringIO(content))

            # Validate column names against the canonical feature contract
            missing_cols = [col for col in FEATURE_COLUMNS if col not in features.columns]
            if missing_cols:
                return render_template(
                    "index.html",
                    error=f"Missing required columns: {', '.join(missing_cols)}",
                )
            features = features[FEATURE_COLUMNS]

            # Make predictions
            predictions = app.model_service.predict(features)  # type: ignore[attr-defined]

            # Format predictions for display
            result = predictions.to_string()

            return render_template("index.html", predictions=result)

        except Exception as e:
            logger.exception(f"Error processing file: {e}")
            return render_template(
                "index.html",
                error=f"Error processing file: {e!s}",
            )


# Create and configure Flask app at module level
setup_logger()
app = Flask(__name__)
app.model_service = ModelService()  # type: ignore[attr-defined]
create_routes(app)
logger.info("Application initialized with model service and routes")


def main() -> None:
    """Run the Flask development server."""
    app.run(host="0.0.0.0", port=5001)


if __name__ == "__main__":
    main()
