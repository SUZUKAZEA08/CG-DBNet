"""CG-DBNet package: models, data utilities, evaluation metrics."""
from .model import (
    AdaptiveAffineFusion,
    CGDBNetBackbone,
    CyclicPositionalEncoding,
    GaussianHead,
    ModelConfig,
    PredictionNet,
    QCNet,
    RevIN,
    build_pred_model,
    build_qc_model,
)
from .data_utils import (
    DataConfig,
    DataProcessor,
    MultiFileTimeSeriesDataset,
    REGION_LAT_MAX,
    REGION_LAT_MIN,
    REGION_LON_MAX,
    REGION_LON_MIN,
    SeqDataset,
    build_feature_matrix,
    build_input_cols,
    extract_lat_lon,
    feature_engineering,
    generate_spatial_features,
    load_sheet,
    prepare_datasets,
    save_json,
)
from .metrics import (
    calculate_metrics,
    compute_dynamic_k_by_coverage,
    evaluate_cls,
    gaussian_nll_loss,
)

__all__ = [
    # model
    "AdaptiveAffineFusion", "CGDBNetBackbone", "CyclicPositionalEncoding", "GaussianHead",
    "ModelConfig", "PredictionNet", "QCNet", "RevIN",
    "build_pred_model", "build_qc_model",
    # data_utils
    "DataConfig", "DataProcessor", "MultiFileTimeSeriesDataset", "SeqDataset",
    "REGION_LAT_MAX", "REGION_LAT_MIN", "REGION_LON_MAX", "REGION_LON_MIN",
    "build_feature_matrix", "build_input_cols", "extract_lat_lon", "feature_engineering",
    "generate_spatial_features", "load_sheet", "prepare_datasets", "save_json",
    # metrics
    "calculate_metrics", "compute_dynamic_k_by_coverage", "evaluate_cls", "gaussian_nll_loss",
]
