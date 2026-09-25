# Case 02A — Improving the Naive Super-Send Transformation

## Historical cutoff

This snapshot represents the research state after super-send was already known and identified as a way to tolerate adversarial link corruptions, but before the idea of using a small committee had been identified.

The goal at this point is to improve the communication complexity obtained by naively applying super-send to existing agreement protocols.

No later committee construction, committee-sampling argument, or final protocol is included.

---

## Synchronous model

There are \(n\) parties.

Up to \(f\) parties are statically Byzantine.

Communication is authenticated using a PKI and digital signatures.

The adversary cannot forge signatures of honest parties except with negligible probability.

In addition to the Byzantine parties, the adversary may adaptively corrupt up to:

* \(d\) incoming links incident to each party;
* \(d\) outgoing links incident to each party.

Messages sent over uncorrupted links are delivered within the known synchronous delay bound.

The resilience condition is

$$
n>\max\{2f,\;f+2d\}.
$$

---

## Agreement tasks

The existing protocol family contains agreement primitives such as weak agreement, graded agreement, and Byzantine agreement.

The desired Byzantine-agreement properties are:

* consistency;
* persistence/validity;
* termination.

The existing constructions use authenticated messages and certificate-style evidence.

A certificate for a value can be formed from sufficiently many distinct signature shares. In particular, threshold-signature machinery is available: \(n-f\) valid shares can be combined into a compact certificate/signature.

---

## Known primitive: super-send

A reliable communication primitive called super-send is already known.

To super-send a message \(m\) from \(P_i\) to an intended recipient \(P_j\):

1. \(P_i\) signs \(m\) and sends it to all parties.
2. Every party forwards the signed message to \(P_j\).
3. \(P_j\) accepts a valid message signed by \(P_i\).

In the synchronous setting, an honest intended recipient receives the message despite the allowed link corruptions.

A super-send to one recipient takes two synchronous communication rounds.

Its communication cost is

$$
O(n(|m|+\lambda)).
$$

Thus super-send is reliable but expensive.

---

## Existing protocol structure

The baseline agreement protocols contain repeated phases in which parties:

* send signed values or signature shares;
* collect sufficiently many shares supporting a value;
* form compact certificates;
* redistribute certificates or votes;
* use these primitives inside graded/weak agreement and Byzantine agreement.

Without link faults, many of these are ordinary point-to-point or all-to-all communications.

---

## Naive transformation

A straightforward way to adapt an existing protocol to the faulty-link model is:

> replace every logical point-to-point message that must arrive reliably with a super-send.

This appears to preserve communication reliability.

However, one logical super-send to one recipient already costs \(O(n)\) transmissions up to message size.

Therefore, applying super-send mechanically to a protocol containing many logical sends can add an additional factor of \(n\) to its communication cost.

In particular, phases containing \(\Theta(n^2)\) logical sender-recipient interactions can become cubic-scale communication after naive super-send replacement.

This transformation appears too expensive.

---

## Known tools

At this historical point the following tools may be used:

* PKI and ordinary signatures;
* threshold-signature shares;
* compact threshold certificates;
* super-send;
* synchronous rounds;
* randomness if a justified protocol construction requires it.

No particular use of these tools is assumed.

---

## Research state

The current approach is:

1. start from known agreement protocols;
2. use super-send wherever reliable delivery is required;
3. obtain correctness under the Byzantine-party and faulty-link model.

The unresolved problem is communication complexity.

The researcher does not yet know whether the large amount of super-sending is fundamentally necessary or whether the protocol can be reorganized so that substantially fewer expensive reliable communications are needed.

---

## Open research question

Adversarially examine the claim that widespread use of super-send throughout the existing agreement protocol is essentially necessary.

Look for a structural alternative that could retain the correctness guarantees while substantially reducing the number of expensive super-send operations.

Any proposed direction must respect:

* up to \(f\) static Byzantine parties;
* the \(d\)-incoming and \(d\)-outgoing link-corruption budgets;
* the resilience condition \(n>\max\{2f,f+2d\}\);
* authenticated communication;
* agreement correctness requirements.

Do not assume a particular improved protocol architecture in advance.
