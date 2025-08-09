from transformers import Olmo2ForCausalLM as TOlmo2ForCausalLM
from transformers import Olmo2Model as TOlmo2Model
from transformers import Olmo2PreTrainedModel as TOlmo2PreTrainedModel

from attention_sinks.inject_mixin import InjectAttentionSinksMixin


class Olmo2PreTrainedModel(InjectAttentionSinksMixin, TOlmo2PreTrainedModel):
    pass


class Olmo2Model(Olmo2PreTrainedModel, TOlmo2Model):
    pass


class Olmo2ForCausalLM(Olmo2PreTrainedModel, TOlmo2ForCausalLM):
    pass

