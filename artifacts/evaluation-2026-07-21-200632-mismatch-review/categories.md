# Mismatch mechanism review

Source scorecard: `evaluation-2026-07-21-200632-post-contact-direction-report.json`

The numbered raw and annotated clips in this directory were reviewed together
with the confirmed-bounce tracks. Categories describe the first visible or
recorded mechanism that explains the mismatch; they are not detector threshold
recommendations.

| Review | Expected | Predicted | Category | Evidence |
| ---: | --- | --- | --- | --- |
| 1 / #15 | miss | hit | tracker handoff | Terminal shadow contact at the far edge becomes the selected hit. |
| 2 / #22 | miss | hit | own-side-first bounce | The far-table turn is retained without ordered earlier-contact history. |
| 3 / #34 | miss | missing | missed launch | No detector attempt is aligned to the labeled machine cycle. |
| 4 / #39 | miss | hit | own-side-first bounce | The recorded path has a valid near-side turn at frame 3280 before the stronger far-side turn at 3285. |
| 5 / #46 | miss | hit | own-side-first bounce | Only the later far-table turn is retained for the attempt. |
| 6 / #47 | miss | hit | tracker handoff | The selected approach walks backward twice, then jumps forward at the apparent contact. |
| 7 / #52 | miss | hit | own-side-first bounce | Only the later far-table turn is retained for the attempt. |
| 8 | none | miss | extra launch | Cadence emits a slot around the known 1:17 floating-ball glitch; the user confirmed it is not a launch. |
| 9 / #66 | hit | missing | missed launch | No detector attempt is aligned to the labeled machine cycle. |
| 10 / #77 | miss | hit | own-side-first bounce | Only the later far-table turn is retained for the attempt. |
| 11 / #93 | hit | miss | missed return | A labeled far-side landing has no accepted hit event. |
| 12 / #122 | miss | hit | own-side-first bounce | The recorded path has a valid near-side turn at frame 10007 before the stronger far-side turn at 10010. |
| 13 / #123 | hit | miss | delayed/overlapping ball | The adjacent #122/#123 outcomes are assigned to one active-attempt sequence. |
| 14 / #143 | hit | miss | missed return | A labeled far-side landing has no accepted hit event. |
| 15 / #159 | miss | hit | own-side-first bounce | Only the later far-table turn is retained for the attempt. |
| 16 / #176 | miss | hit | own-side-first bounce | Only the later far-table turn is retained for the attempt. |
| 17 / #178 | miss | hit | own-side-first bounce | Only the later far-table turn is retained for the attempt. |
| 18 / #189 | miss | hit | tracker handoff | A terminal shadow contact becomes the selected hit. |
| 19 / #197 | hit | miss | missed return | A labeled far-side landing has no accepted hit event. |
| 20 / #219 | miss | hit | own-side-first bounce | Only the later far-table turn is retained for the attempt. |
| 21 / #232 | miss | hit | tracker handoff | A terminal shadow contact at the far edge becomes the selected hit. |

## Counts

- Own-side-first bounce: 10
- Tracker handoff: 4
- Missed return: 3
- Missed launch: 2
- Delayed/overlapping ball: 1
- Extra launch: 1
- Net interaction: 0
- Off-table return: 0

## Tested fixes

The largest class motivated an ordered-contact experiment: keep the earliest
accepted trajectory turn instead of the strongest one. It corrected #39 and
#122 but introduced seven false misses, reducing accuracy to 88.9%. The
experiment was rejected and reverted; weak trajectory turns are not reliable
enough to serve as contact history by themselves.

The review also exposed a narrower, directly observable tracker-handoff case at
#47. Requiring forward continuity across the three-frame approach rejects that
path. It changes only predicted attempt #46 from hit to miss, raises full-session
accuracy from 91.1% to 91.5%, and preserves all three stable video regressions.
