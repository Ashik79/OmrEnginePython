import json

with open('fake_results.json') as f:
    d = json.load(f)

print('Roll:', d['roll'])
print('SET:', d['set'])

expected = ['A','B','C','D','A','B','C','D','A','B','C','D','A','B','C','D','A','B','C','D']
q_active = [q for q in d['questions'] if q.get('errorType') != 'SKIPPED_INACTIVE']

correct = 0
for i, q in enumerate(q_active):
    exp = expected[i] if i < len(expected) else '?'
    ok = q['detected'] == exp
    if ok:
        correct += 1
    print(f"Q{q['qNum']}: got={q['detected']}  exp={exp}  {'OK' if ok else 'WRONG'}")

print(f"\nAccuracy: {correct}/{len(q_active)} = {correct*100//len(q_active) if q_active else 0}%")
