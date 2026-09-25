# Case 05 v2 — Asynchronous A2A Sparse Dissemination

## Historical cutoff

This snapshot represents the research state after the isolation/indistinguishability observation had been identified, but before a quantitative lower-bound proof had been derived.

The researcher suspects that sparse dissemination may be difficult under the adversarial-link model, but does not yet know the correct asymptotic lower bound or the argument needed to establish one.

No later proof structure or final lower bound is included in this snapshot.

---

## Research direction

Consider the possibility that asynchronous all-to-all communication can be implemented with relatively sparse communication despite Byzantine parties and adversarial link corruptions.

The question is whether every honest sender's input can be disseminated to every honest party using relatively few communication opportunities while satisfying the required liveness guarantees.

No target asymptotic communication complexity is assumed.

---

## System model

There are \(n\) parties.

Up to \(f\) parties are statically Byzantine.

The communication network is complete and directed.

For each party, the adversary may corrupt at most:

* \(d\) incoming links;
* \(d\) outgoing links.

The adversary may select these link corruptions adaptively, subject to the per-party incoming and outgoing budgets.

In the asynchronous model, once a link has been corrupted, it cannot later be restored.

A message sent over a corrupted directed link need not be delivered correctly.

Messages sent over uncorrupted links are eventually delivered, but there is no known upper bound on their delivery time. The adversary may delay such a message for an arbitrary finite amount of time.

The resilience condition is

$$
n > \max\{3f,\; f+2d\}.
$$

---

## Cryptographic model

Standard cryptographic tools are available, including:

* PKI and digital signatures;
* threshold signatures;
* VRFs;
* common or random coins.

The adversary is not assumed to break the cryptographic primitives.

---

## All-to-all task

Each honest party \(P_i\) begins with an arbitrary input \(x_i\), initially known to that sender.

For every honest sender \(P_i\), every honest party must eventually obtain \(P_i\)'s original input \(x_i\).

Thus the protocol must provide per-sender liveness for every honest sender.

---

## Communication metric

Communication complexity is measured as the worst-case number of point-to-point transmissions made by honest protocol participants until the required honest inputs have been disseminated.

Message payload size is unrestricted.

The research question concerns transmission complexity rather than bit complexity.

The desired conclusion, if one exists, should be parameterized in \(n\), \(f\), and \(d\).

Degenerate parameter choices such as \(f=0\) or \(d=0\) do not by themselves rule out a lower bound whose value naturally decreases or vanishes in those regimes.

---

## Known observation: isolation indistinguishability

Consider a set \(D\) containing up to \(f\) honest parties.

Delay all communication from the parties in \(D\).

From the perspective of the remaining active parties, these honest-but-delayed parties may be indistinguishable from Byzantine parties that simply remain silent.

Because the protocol must tolerate up to \(f\) Byzantine parties and still satisfy its progress requirements, the active parties cannot wait indefinitely for communication from \(D\).

This observation is known at the historical cutoff.

It has not yet been converted into a quantitative communication lower bound.

---

## Informal redundancy intuition

There is also an informal intuition that sparse dissemination may be vulnerable to the adversary.

If an honest sender's information depends on too small a collection of relays, links, or other communication opportunities, then:

* some relays may be Byzantine;
* some relevant links may be corrupted;
* asynchronous delays may prevent parties from immediately determining why information has not arrived.

This suggests that fault tolerance may force substantial communication redundancy.

However, at this historical point there is no precise counting argument establishing how much redundancy is necessary.

---

## What is not known at this point

The researcher does not yet know:

* the correct asymptotic lower bound;
* whether the isolation observation yields a strong communication bound;
* how much redundancy the \(d\)-link budgets quantitatively force;
* whether costs for different honest inputs can be combined;
* what adversarial execution gives the strongest lower bound;
* whether the current intuition is sufficient or whether another lemma is needed.

No quantitative conclusion should be assumed in advance.

---

## Open research question

Adversarially examine the possibility of sparse asynchronous A2A dissemination under this model.

Determine whether the model itself creates a concrete obstruction to sparse communication, and identify the strongest rigorous conclusions that can be justified from the currently known observations.

Any claimed lower bound must respect simultaneously:

* the Byzantine fault budget;
* the incoming-link corruption budgets;
* the outgoing-link corruption budgets;
* asynchronous message-delivery semantics;
* per-sender liveness;
* the stated communication metric.

Separate rigorous consequences from conjectural intuition.
