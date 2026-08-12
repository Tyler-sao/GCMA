from transformers.configuration_utils import PretrainedConfig

class AIGMAEConfig(PretrainedConfig):
    def __init__(
        self,
        cross_hidden_size=3584,
        cross_num_heads=8,
        freeze: bool = False,
        hidden_size=64,
        model_type="AIGMAE",
        num_classes=4,
        num_hidden_layers=4,  # ✅ 默认4层，可按需改
        num_cross_decoder_layers=2,
        num_encoder_layers=7,
        num_graph_layers=3,

        **kwargs,
    ):
        super().__init__(**kwargs)

        # ✅ 保存所有参数到实例
        self.cross_hidden_size = cross_hidden_size
        self.cross_num_heads = cross_num_heads
        self.freeze = freeze
        self.hidden_size = hidden_size
        self.model_type = model_type
        self.num_classes = num_classes
        self.num_hidden_layers = num_hidden_layers  # ✅ 关键行
        self.num_cross_decoder_layers = num_cross_decoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.num_graph_layers = num_graph_layers
