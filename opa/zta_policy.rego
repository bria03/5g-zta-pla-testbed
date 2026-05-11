package zta.access

import future.keywords.if
import future.keywords.in

# Default deny
default allow = false
default decision = "TSA_DENY"

# Threshold T - configurable
threshold := 0.50

# TSA stage - physical layer check
tsa_pass if {
    input.trust_score >= threshold
}

# UCA stage - all four context attributes must pass
uca_pass if {
    input.supi != ""
    input.access_hour >= 0
    input.access_hour <= 23
    input.requested_slice in {1, 2, 3}
    input.request_frequency <= 100
}

# Full access decision
allow if {
    tsa_pass
    uca_pass
}

# Decision label for logging
decision = "TSA_DENY" if {
    not tsa_pass
}

decision = "UCA_DENY" if {
    tsa_pass
    not uca_pass
}

decision = "UCA_GRANT" if {
    tsa_pass
    uca_pass
}