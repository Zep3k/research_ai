# Case 05 — Asynchronous A2A Isolation

## Historical cutoff

This snapshot represents the state of the research after the isolation idea had been identified, but before a quantitative lower bound had been derived from it.

The final lower-bound argument and its asymptotic complexity are deliberately excluded.

## Problem

We study asynchronous all-to-all communication.

There are \(n\) parties.

Up to \(f\) parties are statically Byzantine.

For every party, the adversary may corrupt up to \(d\) incoming links and up to \(d\) outgoing links.

In the asynchronous model:

* messages over honest links may be delayed for any finite amount of time;
* there is no known upper bound on message delay;
* link corruptions may be added adaptively;
* once a link has been corrupted, it cannot later be restored.

The resilience condition is

$$
n > \max\{3f,\; f+2d\}.
$$

Standard cryptographic tools are available, including:

* PKI and digital signatures;
* threshold signatures;
* VRFs;
* common/random coins.

The adversary is not assumed to break these cryptographic primitives.

## Known historical idea

Consider a set \(D\) of up to \(f\) honest parties.

Delay all communication from parties in \(D\).

The remaining active parties cannot distinguish these honest-but-delayed parties from Byzantine parties that simply remain silent.

Therefore, if the protocol must tolerate up to \(f\) Byzantine parties and still make progress, the remaining active parties cannot wait indefinitely for communication from \(D\).

This isolation/indistinguishability observation is known at this historical point.

## Informal redundancy intuition

There is also an informal intuition that using too few independent parties or links to disseminate information may allow the adversary to block dissemination entirely.

The adversary can:

* corrupt up to \(d\) relevant incoming or outgoing links per party;
* control up to \(f\) Byzantine parties.

Therefore, relying on a small set of relays or communication paths appears dangerous.

The threat that a relay might be Byzantine or that a link might be corrupted seems likely to force communication redundancy.

However, at this historical point:

* there is no precise counting argument;
* there is no proved per-input communication lower bound;
* there is no known way yet to amplify the isolation idea quantitatively;
* no target asymptotic lower bound is assumed.

## Research question

What is the strongest communication lower-bound argument that can be derived from the isolation observation and the adversarial link model?

In particular:

* can delay/corruption indistinguishability force quantitative redundancy?
* can the isolation construction be repeated or amplified?
* can a charging or counting argument be constructed?
* which assumptions are actually necessary?
* what remains unproved?
