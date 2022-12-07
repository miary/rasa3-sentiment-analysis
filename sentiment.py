from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

from typing import Dict, Text, Any, List, Type

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

from rasa.shared.nlu.constants import TEXT

@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER], is_trainable=False
)
class SentimentAnalyzer(GraphComponent):
    @classmethod
    def required_components(cls) -> List[Type]:
        name = "sentiment"
        provides = ["entities"]
        requires = []
        defaults = {}
        language_list = ["en"]
        """Components that should be included in the pipeline before this component."""
    
    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["transformers"]

    def __init__(
        self,
        config: Dict[Text, Any],
        name: Text,
    ) -> None:
        """Constructs a new byte pair vectorizer."""
        super().__init__(name, config)
        # The configuration dictionary is saved in `self._config` for reference.
        
    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Creates a new component (see parent class for full docstring)."""
        return cls(config , execution_context.node_name)

    def train(self, training_data: TrainingData) -> Resource:
        pass

    def convert_to_rasa(self, value, confidence):
        """Convert model output into the Rasa NLU compatible output format."""
        
        entity = {"value": value,
                  "confidence": confidence,
                  "entity": "sentiment",
                  "extractor": "sentiment_extractor"}

        return entity

    def process(self, message, **kwargs):
        entity = self.convert_to_rasa(self, "Negative", "90")
        #text = message.get(TEXT)
        message.set("entities", [entity], add_to_output=True)
        #message.add_features()

    @classmethod
    def load(cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs) -> GraphComponent:
        
        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
        model = AutoModelForSequenceClassification.from_pretrained(MODEL),
        tokenizer = AutoTokenizer.from_pretrained(MODEL),
        myText = "So this is how it feels"
        encoded_msg = tokenizer(myText, return_tensors='pt')
        output = model(**encoded_msg)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return cls.create(config, model_storage, resource, execution_context)
        #pass

       

