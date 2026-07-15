"""Language-free categorical schema used by the model core."""

from enum import IntEnum


class NodeType(IntEnum):
    PAD = 0
    ACTOR = 1
    NEED = 2
    RESOURCE = 3
    ACTION = 4
    CONSTRAINT = 5
    CHANNEL = 6
    VALUE = 7
    REVENUE = 8
    COST = 9
    OUTCOME = 10
    FEEDBACK = 11


class EdgeType(IntEnum):
    NONE = 0
    HAS_NEED = 1
    USES = 2
    ENABLES = 3
    BLOCKS = 4
    REACHES = 5
    DELIVERS = 6
    PAYS = 7
    COSTS = 8
    PRODUCES = 9
    FEEDS_BACK = 10
    DEPENDS_ON = 11
    SUBSTITUTE_FOR = 12
    TRANSFERS_TO = 13
    AMPLIFIES = 14
    REDUCES = 15


class EditOperation(IntEnum):
    STOP = 0
    ADD_NODE = 1
    CONNECT = 2
    REWIRE = 3
    TRANSFER_ROLE = 4
    REMOVE_CONSTRAINT = 5
    INVERT_RELATION = 6
    SUBSTITUTE = 7
    MERGE = 8


SCORE_NAMES = ("utility", "feasibility", "coherence")
