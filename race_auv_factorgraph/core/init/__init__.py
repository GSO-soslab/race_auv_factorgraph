"""Station initialization (§7): the graph gets ``S`` and the ``T_j`` only once
a reliable first estimate exists.

* ``solver``        -- per-frame PnP hypotheses, voting, batch solve, accept/reject
* ``state_machine`` -- NO_STATION -> CANDIDATE -> TRACKING and the frame buffer
"""
