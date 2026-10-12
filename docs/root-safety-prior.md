# Soft safety preference at the root

Protocol fixed before full-game testing, 2026-10-12.

The hard safe-move filter loses 6–18 in fresh confirmation. Removing strategic
sacrifices and applying that assumption throughout simulated play can hurt.
The next isolated hypothesis changes root ordering while preserving every
legal action and leaving internal search nodes untouched.

Optional `SearchConfig.root_safety_scale` defaults to 1.0 (no change). Test
0.01: when no capture is open and at least eight safe actions remain, multiply
gifting-action root priors by 0.01 and renormalize. Actions remain available
for search to recover strategically useful sacrifices. When fewer safe actions
remain, or captures are open, retain the original priors. Exact root answers
still consider all legal actions. This is an experimental preference, not a
claim that sacrificing is wrong.

Use retained weights on both sides, 12 paired-seat 10×10 games, seed 33001,
1,024 simulations each, constant-0.33 search, the cached selector and common
12-edge aid. Only the candidate enables the root preference. No hard safe
filter, chain pruning, structural aid, proofs, rollouts, reuse or timers are
enabled. Freeze source and use the immutable retained NumPy snapshot.

A score above 50% advances unchanged settings to 40 fresh games at seed
221001, requiring score above 50% and paired-bootstrap lower bound strictly
above 50%. A separate 20-game zero-first-play control uses seed 321001 and
requires at least 50%. Release also requires actual web inference checks and
20 games each on 4×4 and 5×5, seed 222001, with web deadlines (1.75 seconds
internally, 1.9 seconds for the supervised candidate), at least 50% score on
each board and every candidate response below two seconds. Keep the cached
selector disabled in those timed checks. No new GPU training is required.

The 67 relevant tests pass. Coverage includes preserving all root actions,
unchanged internal and open-capture priors, normalization when multiple gifts
exist, unchanged late decisions, reused roots without accumulated penalties,
and identical cached/ordinary-selector results.
