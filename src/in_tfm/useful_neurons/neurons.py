"""Every neuron the walk can pick, as one discriminated union keyed by `kind`.

- `Resid(layer=l)` is the stream entering layer l, so `Resid(layer=n_layers)` is what the unembed reads.
  - In the attn-only model `Resid(layer=0)` is the token embedding.
  - In GPT-Neo it is token embedding + position embedding, and `ResidMid` sits between attention and MLP.
- Attention neurons carry the layer they belong to.
- `pos` is always the token whose activation this is. `Score.src` is the token being attended to.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class NeuronKey(BaseModel):
    model_config = ConfigDict(frozen=True)


class Logit(NeuronKey):
    kind: Literal["logit"] = "logit"
    pos: int
    token: int


class Resid(NeuronKey):
    kind: Literal["resid"] = "resid"
    layer: int
    pos: int
    dim: int


class ResidMid(NeuronKey):
    """The stream after layer l's attention and before its MLP."""

    kind: Literal["resid_mid"] = "resid_mid"
    layer: int
    pos: int
    dim: int


class TokenEmbed(NeuronKey):
    kind: Literal["token"] = "token"
    pos: int
    dim: int


class PosEmbed(NeuronKey):
    kind: Literal["pos"] = "pos"
    pos: int
    dim: int


class OProj(NeuronKey):
    """One head's write to one residual dim."""

    kind: Literal["o_proj"] = "o_proj"
    layer: int
    pos: int
    head: int
    dim: int


class HeadOut(NeuronKey):
    """One dim of a head's attention-weighted value sum, the input to that head's o_proj."""

    kind: Literal["head_out"] = "head_out"
    layer: int
    pos: int
    head: int
    dim: int


class Value(NeuronKey):
    kind: Literal["v"] = "v"
    layer: int
    pos: int
    head: int
    dim: int


class Query(NeuronKey):
    kind: Literal["q"] = "q"
    layer: int
    pos: int
    head: int
    dim: int


class Key(NeuronKey):
    kind: Literal["k"] = "k"
    layer: int
    pos: int
    head: int
    dim: int


class Score(NeuronKey):
    """The pre-softmax attention score from `pos` to `src`."""

    kind: Literal["score"] = "score"
    layer: int
    pos: int
    src: int
    head: int


class MlpHidden(NeuronKey):
    """One MLP hidden neuron, after its activation function."""

    kind: Literal["mlp_hidden"] = "mlp_hidden"
    layer: int
    pos: int
    dim: int


class MlpDown(NeuronKey):
    """The MLP's write to one residual dim."""

    kind: Literal["mlp_down"] = "mlp_down"
    layer: int
    pos: int
    dim: int


Neuron = Annotated[
    Logit
    | Resid
    | ResidMid
    | TokenEmbed
    | PosEmbed
    | OProj
    | HeadOut
    | Value
    | Query
    | Key
    | Score
    | MlpHidden
    | MlpDown,
    Field(discriminator="kind"),
]
