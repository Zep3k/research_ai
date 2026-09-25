# Case 05 v1 — Asynchronous A2A Isolation

## Historical cutoff

This snapshot represents the state of the research after the isolation idea had been identified, but before a quantitative communication lower bound had been derived from it.

No target asymptotic lower bound is assumed.

The final lower-bound proof is deliberately excluded.

## System model

There are \(n\) parties.

Up to \(f\) parties are statically Byzantine.

For every party, the adversary may corrupt up to:

* \(d\) incoming communication links;
* \(d\) outgoing communication links.

The adversary may choose link corruptions adaptively, subject to these per-party budgets.

In the asynchronous model, once a communication link is corrupted, it cannot later be restored.

Messages sent over uncorrupted links are eventually delivered, but there is no known upper bound on their delivery time. The adversary may therefore delay an honest-link message for an arbitrary finite amount of time.

Assume

$$
n > \max\{3f,\; f+2d\}.
$$

Standard cryptographic tools are available, including:

* PKI and digital signatures;
* threshold signatures;
* VRFs;
* common/random coins.

The adversary is not assumed to break these cryptographic primitives.

## All-to-all task

Every honest party \(P_i\) has an original input \(x_i\).

For every honest sender \(P_i\), every honest party must eventually obtain \(P_i\)'s original input \(x_i\).

Equivalently, the protocol must provide per-sender liveness for every honest sender.

## Communication metric

Communication complexity is measured in point-to-point transmissions.

The quantity of interest is worst-case communication complexity.

## Known historical idea: isolation

Consider a set \(D\) of up to \(f\) honest parties.

Delay all communication from the parties in \(D\).

The remaining active parties cannot distinguish these honest-but-delayed parties from Byzantine parties that simply remain silent.

Therefore, because the protocol must tolerate up to \(f\) Byzantine parties and still make progress, the active parties cannot wait indefinitely for communication from \(D\).

This isolation/indistinguishability observation is already known at the historical cutoff.

## Informal redundancy intuition

There is also an informal intuition that disseminating an honest sender's input through too few independent communication opportunities is dangerous.

The adversary can corrupt communication links subject to the \(d\)-incoming and \(d\)-outgoing budgets, and it may also control Byzantine parties.

Therefore, if information depends on too small a set of relays or links, the adversary may be able to prevent correct dissemination.

The threat that a communication link may be corrupted, or that a relay may be Byzantine, appears likely to force redundancy.

However, at this historical point:

* there is no precise counting argument;
* there is no proved per-input communication lower bound;
* there is no known method for amplifying the isolation observation quantitatively;
* no asymptotic target is assumed.

## Research question

What is the strongest rigorous worst-case communication lower-bound strategy that can be derived from the isolation observation and the adversarial link model?

In particular:

* can asynchronous delay be used in an indistinguishability argument involving possible link corruptions?
* what quantitative redundancy does tolerance of \(d\) faulty incoming/outgoing links force?
* can the isolation construction be repeated or amplified?
* can a charging or counting argument be obtained?
* which parts are rigorous and which remain conjectural?
* what additional lemma, if any, would be needed to complete the argument?
