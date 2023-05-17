import sys
import pickle
from dataclasses import dataclass


@dataclass
class Snapshot:
    """
    this class is used for taking snapshots from processed dictionary
    and postings list to avoid re-execution of the same tasks multiple
    times. However, it didn't work
    """
    _dictionary: dict
    _docs_tokens: list[list[int]]

    @property
    def dictionary(self):
        return self._dictionary

    @property
    def docs_tokens(self):
        return self._docs_tokens


def take_snapshot():
    """ takes snapshot from the current status of the running program or process """
    global dictionary
    global docs_tokens
    snapshot = Snapshot(_dictionary=dictionary, _docs_tokens=docs_tokens)
    snapshot_file = open("snapshot.obj", "wb")
    sys.setrecursionlimit(10000)
    pickle.Pickler(snapshot_file, protocol=pickle.HIGHEST_PROTOCOL).dump(pickle.dumps(snapshot))
    snapshot_file.close()


def restore_from_snapshot():
    """ restore a previously taken snapshot """
    global dictionary
    global docs_tokens
    snapshot_file = open("snapshot.obj", "rb")
    pickled = pickle.load(snapshot_file)
    snapshot: Snapshot = pickle.loads(pickled)
    dictionary = snapshot.dictionary
    docs_tokens = snapshot.docs_tokens
    snapshot_file.close()


