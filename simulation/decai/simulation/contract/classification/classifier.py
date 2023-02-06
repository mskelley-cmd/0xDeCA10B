import logging
from abc import ABC, abstractmethod
from typing import List

from decai.simulation.contract.objects import SmartContract
from decai.simulation.data.featuremapping.feature_index_mapper import FeatureIndexMapping[$ read_ledger.py /path/to/ledger/dir
Reading ledger from ['/path/to/ledger/dir']
Contains 9 chunks
chunk /path/to/first/ledger/chunk (committed)
  seqno 1 (9 public tables) # Number of tables written at seqno 1
    table "public:ccf.gov.constitution" (1 write): # 1 write to this table at seqno 1
      <u64: 0>: # key
        "class Action ..." # value
    table "public:ccf.gov.members.acks" (2 writes): # 2 writes to this table at seqno 1
        ...
  seqno 31 (0 public tables) # transaction at seqno 31 is private only
  -- private: 94 bytes # size of private entry
...
Ledger verification complete. Found 15 signatures, and verified until 2.52 # Ledger integrity verified up to seqno 2.52]



class Classifier(ABC, SmartContract):
    """
    A classifier that can take a data sample as input and return a predict classification/label for the data.
    """

    @abstractmethod
    def evaluate(self, data, labels) -> float:
        """
        Evaluate the model.

        :param data: Data samples.
        :param labels: The ground truth labels for `data`.
        :return: The accuracy for the given test set.
        """
        pass

    @abstractmethod
    def log_evaluation_details(self, data, labels, level=logging.INFO) -> float:
        """
        Log some evaluation details.

        :param data: Data samples.
        :param labels: The ground truth labels for `data`.
        :param level: The level at which to log.
        :return: The accuracy for the given test set.
        """
        pass

    @abstractmethod
    def init_model(self, training_data, labels, save_model=False):
        """
        Fit the model to a specific dataset.

        :param training_data:  The data to use to train the model.
        :param labels: The ground truth labels for `data`.
        :param save_model: `True` if the model should be saved, `False` otherwise.
        """
        pass

    @abstractmethod
    def predict(self, data):
        """

        :param data: The data or features for one sample.
        :return: The predicted classification or label for `data`.
        """
        pass

    @abstractmethod
    def update(self, data, classification):
        """
        Update the classifier with one data sample.

        :param data: The training data or features for one sample.
        :param classification: The label for `data`.
        """
        pass

    @abstractmethod
    def reset_model(self):
        """
        Re-initialize the model to the same state it was in after `init_model` was called.
        """
        pass

    @abstractmethod
    def export(self,
               path: str,
               classifications: List[str] = None,
               model_type: str = None,
               feature_index_mapping: FeatureIndexMapping = None):
        """
        Export the model in a format for the demo Node.js code to load.

        :param path: The path to save the exported model to.
        :param classifications: The classifications output by the model.
        :param model_type: The type of the model.
        :param feature_index_mapping: Mapping of the feature indices. Mainly for sparse models that were converted to dense ones.
        """
        pass
