from collections.abc import Sequence

import torch

from .base import ModelBatch, SampleId


class TextSource:
    """A list of strings, tokenized and embedded for a decoder LM.

    Unlike the image path this must hand back embeddings rather than the model's natural input:
    `input_ids` are integer indices, so there is no gradient to take with respect to them. The
    embedding lookup is the first differentiable point, which makes `inputs_embeds` the leaf and
    means the source needs the model's embedding table.
    """

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        embed: torch.nn.Module,
        max_length: int = 128,
        prepend_bos: bool = False,
    ) -> None:
        self.texts = list(texts)
        self.prepend_bos = prepend_bos
        self.tokenizer = tokenizer
        self.embed = embed
        self.max_length = max_length

        # The padding side is left as the tokenizer's own - left, for Gemma. It used to be
        # forced to "right", because a token index read off a batched capture had to still be
        # valid when the sample was later re-encoded alone for attribution, and left padding
        # shifts every real token by however much padding that particular batch needed.
        #
        # Padding every sequence to max_length removes that reason: a sample occupies the same
        # columns in a batch of eight and in a batch of one, whichever side the padding is on.
        # See `_encode`, and `activations.cat_captures` for the invariant it enforces.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def sample_ids(self) -> Sequence[SampleId]:
        return [str(i) for i in range(len(self.texts))]

    def text_for(self, sample_id: SampleId) -> str:
        return self.texts[int(sample_id)]

    def to_model_batch(self, ids: Sequence[SampleId]) -> ModelBatch:
        encoded = self._encode(ids)
        mask = encoded["attention_mask"]
        # the embedding table already sits on the accelerator by this point, while the
        # tokenizer always returns cpu ids - index_select needs both on the same device
        with torch.no_grad():
            embeds = self.embed(encoded["input_ids"].to(self._embed_device()))
        return ModelBatch(
            kwargs={"inputs_embeds": embeds, "attention_mask": mask},
            grad_leaf_key="inputs_embeds",
            valid_mask=mask.bool(),
            display_ids=encoded["input_ids"],
        )

    def _embed_device(self) -> torch.device:
        return next(self.embed.parameters()).device

    def _encode(self, ids: Sequence[SampleId]):
        """Every sequence comes back exactly `max_length` wide - padded up, truncated down.

        `padding=True` would pad to the longest sequence in *this* batch, which makes a
        sample's columns depend on what it was batched with. Capture runs in batches over the
        whole corpus while attribution re-runs one sample at a time, so a batch-dependent
        width means a hit's token index does not survive the trip between them. A fixed width
        is what lets both passes speak the same coordinates.

        The cost is attention over `max_length` columns even for a short text. Cheap when the
        corpus is filtered to long samples (see wikitext.load_texts), wasteful
        when it is not - `max_length` is the knob.
        """
        return self.tokenizer(
            [self._with_bos(self.text_for(i)) for i in ids],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

    def _with_bos(self, text: str) -> str:
        """Some tokenizers (GPT-NeoX) never add BOS themselves, but parse its string as the
        special token, which is how TransformerLens prepends it."""
        return self.tokenizer.bos_token + text if self.prepend_bos else text
