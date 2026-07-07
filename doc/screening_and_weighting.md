# Understanding reaction screening and experiment weighting

This guide explains, in plain language, how Frhodo decides which parts
of your data matter, how it ranks reactions before an optimization, and
how to use those rankings. No code, no derivations — just the ideas and
how to work with them.

---

## 1. What the optimizer is actually doing

An optimization run adjusts the rate constants of the reactions you
selected until the simulated traces agree with your measured traces as
well as possible. "As well as possible" is defined by a single number —
the objective — built from every selected experiment. Everything in this
guide is about how that number is assembled and how to predict, before
spending hours of computer time, which reactions can improve it.

## 2. Weighting inside one experiment

Every experiment carries a weight curve over its trace (the red curve on
the Signal/Sim plot). It answers one question: *where in this trace do
you trust the data and want the fit to try hard?* Early points dominated
by shock-arrival artifacts, or late points that are mostly noise, get
low weight; the clean chemistry window gets high weight.

The weight curve is not cosmetic. Every quantity described below —
the misfit itself, and both screening scores — is computed under it. A
region you set to zero weight is invisible to the optimizer and to
screening alike.

## 3. Weighting across experiments

Experiments enter the objective as a weighted average. Two mechanisms
set those weights:

**Your per-experiment weight.** Set directly in the series viewer. Use
it to reflect confidence in a particular shock.

**Balance weighting.** Campaigns are rarely spread evenly: you might
have ten shocks near one condition and two lonely ones elsewhere. In a
plain average the crowded cluster outvotes the lonely experiments, even
though the lonely ones often carry information nothing else provides.
Balance weighting counters this by sharing the vote: experiments that
say roughly the same thing split one vote between them, while an
experiment with something unique to say keeps a full vote.

Two experiments can be redundant in two different ways, and Frhodo
checks both. The first is condition redundancy: shocks at nearly the
same temperature and pressure. The second is information redundancy:
shocks that constrain the same reactions in the same way — and its
mirror image, a shock that probes something nobody else does. The
second check matters when composition varies: two shocks at identical
temperature and pressure but different mixtures can carry completely
different information, and only the information check can see that. It
is measured once at the start of the run, from the same sensitivity
machinery screening uses, and held fixed for the whole optimization.
If that measurement fails for any reason, Frhodo falls back to the
condition-based balance alone and says so in the log.

Related to balance is **exclusion**: an experiment that no reaction can
influence (its simulation barely responds to any rate change) cannot
steer the optimization at all. It only adds computing time. Frhodo can
identify such experiments and leave them out of the optimization loop —
they are still reported afterward, since "the model already fits this
one" is itself worth knowing.

## 4. Reaction screening

Optimizing every reaction in a large mechanism is neither feasible nor
meaningful — most reactions cannot affect your observable at your
conditions. Screening ranks all reactions before you fit, using the
mechanism as loaded, so you can choose targets with evidence instead of
intuition.

Screening runs by itself in the background whenever the campaign
changes — selecting or deselecting experiments, adding a series, or
loading a mechanism. You never need to trigger it. When it finishes, a
one-line summary appears in the log, and the sort menu next to the
mechanism filter becomes fully available.

It produces two scores per reaction:

**Importance** — *could this reaction visibly move the simulated trace,
in the regions where your data has weight?* This ignores how good or bad
the current fit is; it only asks whether the observable responds to the
reaction at all. Use it when the starting fit is so poor that "where the
disagreement is" carries little meaning.

**Leverage** — *would changing this reaction actually reduce the current
disagreement?* A reaction can be influential yet useless: if it moves
the trace only where the fit is already good, or pushes in a direction
that doesn't match the misfit, adjusting it buys nothing. Leverage
combines the reaction's influence with the shape of the current
disagreement, under all the weights described above. This is the score
to sort by when choosing what to optimize.

Both scores assume any reaction is allowed to change — you do not need
to set uncertainties before screening is useful.

**The suggested set.** The log summary reports how many reactions are
"suggested." A reaction is suggested when, assuming it could change by
about a factor of two, its best-case effect on the objective exceeds
one percent of the current misfit. This is an absolute test, not a
ranking: on a campaign where nothing can help, nothing is suggested.
That result is meaningful — it says this data cannot improve this
mechanism, no matter which reactions you pick.

**Reading the rest of the summary line.** The *effective rank* estimates
how many independent "knobs" your campaign can actually set. If you plan
to optimize twelve reactions but the effective rank is five, the data
cannot pin twelve values — several fitted rates will trade off against
each other, and their individual values should not be over-interpreted.
A *kink* in the score spectrum, when present, marks a natural boundary
between reactions that matter and reactions that don't; a smooth
spectrum (no kink) means the campaign has no clean cutoff.

## 5. A practical workflow

1. Load the mechanism, load your experiments, select the shocks to fit,
   and set the weight curves so they cover the trustworthy part of each
   trace. Screening runs on its own.
2. Sort the reaction list by **Leverage**. The top of the list is where
   the objective can be improved.
3. Choose your targets from the leaders, set their rate uncertainties
   (these bound how far the optimizer may move each rate), and run the
   optimization.
4. After the run, use the Optimization-tab views to see what moved,
   which reactions sit at their bounds, and which experiments improved.

## 6. Honest limitations

- Screening evaluates the mechanism *as loaded*. After a large
  optimization the landscape changes; re-screening afterward can
  reshuffle the lower ranks. The leaders are usually stable.
- The scores are first-order estimates — excellent for ranking, not
  exact predictions of achievable improvement.
- Screening cannot see reactions missing from the mechanism. A perfect
  fit to a wrong mechanism is still wrong, and no ranking can flag
  chemistry that isn't there.
