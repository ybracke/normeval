from normeval import Evaluator
import json

# Metadata
dname = 'Example dataset1'
pipeline = 'X>Y>Z'


pred = ['AN',  'diesem',  'fünften',  'Stucke',  'des',  'puchs',  'Soll',
'wir',  'sagen',  'von']

gold = ['An',  'diesem',  'fünften',  'Stück',  'des',  'Buchs',  'sollen',
'wir',  'sagen',  'von']

ev = Evaluator(pred, gold)
metrics = ev.evaluate(methods="all")

output = []

output.append({
    'dataset': dname, 
    'normalizer': pipeline,
    'sample-size': len(pred),
    'metrics': metrics
    })

# store results
with open("testout.json", "w") as fp:
    json.dump(output, fp, indent=4)