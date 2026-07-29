# TF NLI Analysis

- Total TF questions: 120

## Label distribution: `TF_NLI_Label_Pre`

| Label              |   Count |   Frac |
|:-------------------|--------:|-------:|
| SUPPORTED          |      66 |   0.55 |
| HARD_CONTRADICTION |       0 |   0    |
| SOFT_CONTRADICTION |       0 |   0    |
| NOT_ENOUGH_INFO    |      24 |   0.2  |
|                    |       0 |   0    |
| CONTRADICTED       |      30 |   0.25 |


## Label distribution: `TF_NLI_Label_Hop1`

| Label              |   Count |       Frac |
|:-------------------|--------:|-----------:|
| SUPPORTED          |       0 | 0          |
| HARD_CONTRADICTION |       0 | 0          |
| SOFT_CONTRADICTION |       0 | 0          |
| NOT_ENOUGH_INFO    |      23 | 0.191667   |
|                    |      96 | 0.8        |
| CONTRADICTED       |       1 | 0.00833333 |


## Label distribution: `TF_NLI_Label_Hop2`

| Label              |   Count |      Frac |
|:-------------------|--------:|----------:|
| SUPPORTED          |       2 | 0.0166667 |
| HARD_CONTRADICTION |       0 | 0         |
| SOFT_CONTRADICTION |       0 | 0         |
| NOT_ENOUGH_INFO    |      21 | 0.175     |
|                    |      97 | 0.808333  |


## Label distribution: `TF_Final_NLI_Label_Used`

| Label              |   Count |     Frac |
|:-------------------|--------:|---------:|
| SUPPORTED          |      68 | 0.566667 |
| HARD_CONTRADICTION |       0 | 0        |
| SOFT_CONTRADICTION |       0 | 0        |
| NOT_ENOUGH_INFO    |      21 | 0.175    |
|                    |       0 | 0        |
| CONTRADICTED       |      31 | 0.258333 |


## Flip matrix: pre → final

| pre_label          |   SUPPORTED |   HARD_CONTRADICTION |   SOFT_CONTRADICTION |   NOT_ENOUGH_INFO |    |
|:-------------------|------------:|---------------------:|---------------------:|------------------:|---:|
| SUPPORTED          |          66 |                    0 |                    0 |                 0 |  0 |
| HARD_CONTRADICTION |           0 |                    0 |                    0 |                 0 |  0 |
| SOFT_CONTRADICTION |           0 |                    0 |                    0 |                 0 |  0 |
| NOT_ENOUGH_INFO    |           2 |                    0 |                    0 |                21 |  0 |
|                    |           0 |                    0 |                    0 |                 0 |  0 |


- Flip rate (pre != final): 0.025

## TF policy simulation (deterministic mapping)

- Simulated acc using pre-label only: 0.892

- Simulated acc using final-label used: 0.908

- Simulated gain (final - pre): 0.017


### Among Completion_Triggered=True

- N=24 pre-only acc=0.708 final acc=0.792 gain=0.083


### Among Hop2_Triggered=True

- N=23 pre-only acc=0.696 final acc=0.783 gain=0.087
