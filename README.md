# AI.tester
quick test 9 different AI methods to see which method is the best for your data set

test 9 different learning/AI methods to identify which method may be most appropriate for your dataset.

Outputs a 3x3 plot of ROC plots along with JSON of the data and JSON data for precision-recall plots.
```
usage: ai.classifier.py [-h] --file FILE --output_stem OUTPUT_STEM --target TARGET
                        [--categorical CATEGORICAL [CATEGORICAL ...]] [--drop DROP [DROP ...]]
                        [--suptitle SUPTITLE]

Pass a file name.

required arguments:
  --file FILE
  --output_stem OUTPUT_STEM
  --target TARGET

optional arguments:
  --categorical CATEGORICAL [CATEGORICAL ...]
  --drop DROP [DROP ...]
  --suptitle SUPTITLE
```
