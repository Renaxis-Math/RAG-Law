import pandas as pd

test_cases = [
    {"input": "Define insurable interest under California law.", 
     "expected_source": "VectorDB"},
    {"input": "When does a newly issued insurance policy become effective in California?", 
     "expected_source": "VectorDB"},
    {"input": "What duties does the insured have upon discovering a loss according to California insurance statutes?", 
     "expected_source": "VectorDB"},
    {"input": "What recent California Supreme Court case clarified insurer bad faith standards?", 
     "expected_source": "ChitterChatterWebSearch"},
    {"input": "How did California Proposition 103 change auto insurance rate regulation?", 
     "expected_source": "ChitterChatterWebSearch"},
    {"input": "Can you recommend a good recipe for chocolate chip cookies?", 
     "expected_source": "ChitterChatterWebSearch"},
]

df = pd.DataFrame(test_cases)
df.to_csv('tests.csv', index=False)
