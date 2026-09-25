# Case 02B-WA — Developing Committee-Based Weak Agreement

## Historical cutoff

This snapshot represents the research state after the high-level committee idea had been identified, but before the concrete committee-based weak-agreement protocol had been developed.

The researcher knows that a small randomly sampled committee may allow expensive reliable communication to be concentrated among a few parties.

The exact protocol phases, certificate-forwarding mechanism, correctness proof, and final communication analysis are unknown.

---

## Model

There are \(n\) parties.

Up to \(f\) parties are statically Byzantine.

The network is synchronous.

The adversary may adaptively corrupt up to:

* \(d\) incoming links incident to each party;
* \(d\) outgoing links incident to each party.

Messages sent over uncorrupted links are delivered within the synchronous delay bound.

Assume

$$
n>\max\{2f,\;f+2d\}.
$$

Communication is authenticated using PKI and digital signatures.

Honest signatures cannot be forged.

---

## Weak Agreement

Each party \(P_i\) begins with an input \(x_i\).

A weak-agreement protocol must satisfy:

### Weak consistency

If an honest party decides \(y\), then every honest party decides either \(y\) or \(\perp\).

### Persistence

If all honest parties start with the same input \(x\), then every honest party decides \(x\).

### Termination

Every honest party eventually decides.

---

## Known primitive: super-send

To super-send a message \(m\) from \(P_i\) to intended recipient \(P_j\):

1. \(P_i\) signs \(m\) and sends the signed message to all parties.
2. Every party forwards the signed message to \(P_j\).
3. \(P_j\) accepts a valid message signed by \(P_i\).

For an honest intended recipient, delivery is guaranteed despite the allowed faulty links.

A super-send takes two synchronous communication rounds.

Its communication cost is

$$
O(n(|m|+\lambda)).
$$

Super-send is therefore reliable but expensive.

---

## Threshold signatures and certificates

A threshold-signature scheme is available.

Any set of at least

$$
n-f
$$

valid signature shares for the same protocol instance, phase, message tag, and value \(v\) can be combined into a compact threshold signature.

Define a certificate

$$
C(v)=(v,\Sigma(v)),
$$

where \(\Sigma(v)\) combines \(n-f\) valid shares supporting \(v\).

The resulting certificate has size \(O(|v|+\lambda)\).

Honest parties issue shares only according to the protocol rules.

---

## Known committee idea

Let \(S\) be a small committee sampled uniformly from the parties after the static Byzantine set has been fixed.

Because

$$
n>2f,
$$

the Byzantine fraction is below one half.

For a committee of size \(c\),

$$
\Pr[S\text{ contains no honest party}]
=
\frac{\binom{f}{c}}{\binom{n}{c}}
\le
\left(\frac fn\right)^c
<
2^{-c}.
$$

Thus a small random committee contains at least one honest member with high probability.

For this benchmark, reason conditioned on the successful-sampling event:

> the committee contains at least one honest party.

The committee is not assumed to have an honest majority.

Byzantine committee members remain fully Byzantine.

---

## Motivation

Naively super-sending every important logical message between all pairs of parties can produce cubic-scale communication.

The promising direction is instead to use the small committee to perform selected expensive communication, aggregation, or certificate-related roles.

However, no concrete weak-agreement protocol is known yet.

In particular, it is not yet known:

* what ordinary parties should send to committee members;
* what committee members should aggregate;
* what committee members should send back;
* what happens if Byzantine committee members create, suppress, or selectively release valid evidence;
* how different certificates for different values should be handled;
* what an honest party should output;
* why weak consistency holds;
* why persistence holds;
* what the resulting communication complexity is.

---

## Research task

Develop a concrete committee-based weak-agreement protocol from these ingredients.

Use the small committee, super-send, and threshold signatures in a way that substantially improves over naively making all important logical sender-recipient interactions reliable.

The construction must address:

* weak consistency;
* persistence;
* termination;
* Byzantine committee members;
* conflicting valid certificates;
* the fact that only one honest committee member is guaranteed;
* communication complexity.

Do not assume any later protocol construction.

The goal is not merely to criticize the committee idea. Push it toward a concrete protocol and identify the proof obligations required to establish correctness.
