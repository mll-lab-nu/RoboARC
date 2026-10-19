"""Is a builder checkpoint exactly the state a fresh replay of its rows reaches? (state hash, mechanism welds)"""
import pickle, sys
import numpy as np
import rcb_ikea as R
st = pickle.load(open(sys.argv[1], 'rb')); rows = np.array(st['ex'])
a = R.Session('stenerik'); a.set_state(st); h_ck = a.state_sha()
b = R.Session.__new__(R.Session); b.__dict__.update(a.__dict__); b.reset(); b.run(rows)
print('rows', len(rows), 'ck', h_ck[:16], 'replay', b.state_sha()[:16], 'equal', h_ck == b.state_sha())
