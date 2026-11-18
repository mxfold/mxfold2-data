# Experiment scripts and tests

This directory holds the scripts for the eFold, EternaFold, implicit-MLE,
RNAgym, and default-model experiments, along with their regression tests. The
directory structure follows the former `scripts/` and research `tests/` in
MXfold2. The snapshot includes experimental work from the MXfold2 working
tree at commit `02fd1b1f8`.

Use [the paper reproduction workflow](../README.md) for the study and the
[default-model recipe](../default-model/README.md) for the 0.3.0 training run.
Those workflows carry their prepared inputs and separate frozen sources.
The scripts here cover individual studies, diagnostics, and TSUBAME jobs,
including older configurations that require the original cluster tools,
datasets, and checkpoints. Keep experiment inputs and job outputs in a local
work directory. Run experiment commands from this directory so that
`scripts/` and `tests/` resolve to the files here.

## Run experiment regression tests

Use Python with the requirements from the [repository root](../requirements.txt)
and a built MXfold2 native extension. With the frozen source built by the paper
workflow, run from `experiments/`:

```sh
PYTHONPATH="$PWD:$PWD/../source" python -m unittest discover -s tests
```

Some checks need historical job artifacts or auxiliary executables, including
CD-HIT-EST. The EternaFold numerical suite runs separately with the patched
`contrafold` executable:

```sh
EF_TEST_BINARY=/path/to/patched/contrafold python tests/test_eternafold_workflow.py
```

The EternaFold resume fixtures are four BPSEQ records from the public Rivas
TrainSetA release. The eFold registry test includes the small metrics module it
uses to freeze the test registry.
