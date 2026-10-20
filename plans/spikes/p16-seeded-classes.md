# Spike: sources for adv-seeded-v0's fault classes (task P16.8)

Date: 2026-10-20. Question: which published practice do the seeded fault classes of
tests/fixtures/adversary/seeded/ come from, and are the citations right? Method: four web searches; only the
search results' bibliographic records and summaries were read, never the papers' full texts.

## Searches and what they returned

1. "Offutt Lee Rothermel Untch Zapf An experimental determination of sufficient mutant operators ACM TOSEM 1996".
   A BibTeX record (cs.unm.edu) gives ACM Transactions on Software Engineering and Methodology 5(2):99-118, 1996,
   with the five authors; a 1994 George Mason technical report (ISSE-TR-94-00) carries the same title as a draft.
   Citing papers (Just et al., non-redundant mutants, ICST 2012 and JSTVR 2014; a 2011 paper) say the study
   evaluated Mothra's 22 operators and found 5 sufficient, and name AOR, ROR, and LCR among them, with ABS and UOI
   discussed beside them. Not confirmed here: the exact list of five from the paper itself.
   Sources: https://www.cs.unm.edu/~eschulte/dissertation/refs/offutt1996experimental,
   https://cs.gmu.edu/media/techreports/ISSE-TR-94-00.pdf,
   https://homes.cs.washington.edu/~rjust/publ/non_redundant_mutants_icst_2012.pdf,
   https://homes.cs.washington.edu/~rjust/publ/non_redundant_mutants_jstvr_2014.pdf.
2. "DeMillo Lipton Sayward Hints on Test Data Selection IEEE Computer 1978". The CERIAS record and a DBLP-derived
   citation give IEEE Computer 11(4):34-41, April 1978, DOI 10.1109/C-M.1978.218136, authors Richard A. DeMillo,
   Richard J. Lipton, and Frederick G. Sayward; secondary summaries credit it with introducing mutation testing.
   Sources: https://www.cerias.purdue.edu/apps/reports_and_papers/view/919, https://dblp.org/pid/10/3823.
3. "McKeeman Differential Testing for Software Digital Technical Journal 1998". DBLP and a reference list give
   Digital Technical Journal 10(1):100-107, 1998; a 1998 GCC mailing-list post gives the variant title
   "Differential Testing of Software". Sources: https://dblp.org/pid/40/4856,
   https://inbox.sourceware.org/gcc/24514.915075998@hurl.cygnus.com/T.
4. "Laguna Varity quantifying floating-point variations HPC randomized testing IPDPS 2020". The LLNL Varity
   repository cites Ignacio Laguna, "Varity: Quantifying Floating-point Variations in HPC Systems Through
   Randomized Testing", IPDPS 2020, pp. 622-633, and describes randomized differential testing of floating-point
   programs across compilers. Source: https://github.com/LLNL/Varity.

## Result

The boundary and arithmetic classes follow relational and arithmetic operator replacement (ROR, AOR) as citing
papers report Offutt et al.'s sufficient set; the sign class follows the absolute value and unary operator
operators (ABS, UOI) discussed beside it; the precision class follows the floating-point variation that randomized
differential testing looks for (Laguna 2020; McKeeman 1998 for differential testing). The narrowing and
special-value classes cite no source: they are this task's choice of numeric faults a C++ translation can make.
The class list is a choice, known cases, not complete.
