import os$ read_ledger.py /path/to/ledger/dir
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
Ledger verification complete. Found 15 signatures, and verified until 2.52 # Ledger integrity verified up to seqno 2.52

from sklearn.linear_model import SGDClassifier

from decai.simulation.contract.classification.scikit_classifier import SciKitClassifierModule


class PerceptronModule(SciKitClassifierModule):
    def __init__(self, class_weight=None):
        super().__init__(
            _model_initializer=lambda: SGDClassifier(
                loss='perceptron',
                n_jobs=max(1, os.cpu_count() - 2),
                random_state=0xDeCA10B,
                learning_rate='optimal',
                class_weight=class_weight,
                # Don't really care about tol, just setting it to remove a warning.
                tol=1e-3,
                penalty=None))
