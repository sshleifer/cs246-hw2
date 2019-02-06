import numpy as np

def lmap(f,x): return list(map(f,x))


shows_path = 'q4/data/shows.txt'
ratings_path = 'q4/data/user-shows.txt'

with open(shows_path) as f:
    shows = [x.strip() for x in f]

with open(ratings_path) as f:
    ratings = [lmap(int, x.strip().split()) for x in f]

R = np.array(ratings)
m, n = R.shape

P = np.diag(R.sum(1))
Q = np.diag(R.sum(0))

assert P.shape == (m, m)
assert Q.shape == (n, n)

QROOT = np.diag(1/np.sqrt(R.sum(0)))

PROOT = np.diag(1/np.sqrt(R.sum(1)))

SU = PROOT.dot(R).dot(R.T).dot(PROOT)
SUR = SU.dot(R)

gamma_item = R.dot(QROOT).dot(R.T).dot(R).dot(QROOT)

ALEX_ID = 499

S = list(range(100))

item_scores = gamma_item[ALEX_ID, S]

user_scores = SUR[ALEX_ID, S]

def postprocess(scores):
    res = []
    scored_recs = sorted(enumerate(scores), key=lambda x: (-x[1], x[0]))
    for show_id, score in scored_recs[:5]:
        show_name = shows[show_id]
        res.append(f'{show_name}: {score: .3f}')
    return '\n'.join(res)

item_out = postprocess(item_scores)
print('ITEM to ITEM')
print(item_out)
print('USER to USER')

user_out = postprocess(user_scores)
print(user_out)
