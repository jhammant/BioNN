"""Model registry."""

from bionn.models.bnn import BNNModel
from bionn.models.mlp import MLPModel
from bionn.models.online import OnlineModel
from bionn.models.snn import SNNModel

MODELS: dict[str, type] = {
    "mlp": MLPModel,
    "bnn": BNNModel,
    "snn": SNNModel,
    "online": OnlineModel,
}
