"""Check: in run_synthetic.py the misspecified muY1 constant equals mean compliance (OracleMuA z=1),
not mean outcome E[muY(1,X)] as the paper states. Supports finding misspec-constant-mismatch."""
import sys, os
base=os.path.join(os.path.dirname(__file__),"..","code","CausalML__Adaptive-IV")
sys.path.insert(0,base); os.chdir(base)
import numpy as np
from data import make_synthetic_iv_dgp, OracleMuA, OracleMuY
np.random.seed(1); d=5; beta=np.random.uniform(-1,1,size=d)
def my_f(X,a):
    X=np.asarray(X); single=X.ndim==1
    if single: X=X[None,:]
    val=1+a+X[:,0]+2*a*(X@beta)+0.75*a*X[:,0]**2
    return val[0] if single else val
gen=make_synthetic_iv_dgp(f=my_f,d=d,seed=0)
X_test=np.array([gen()[0] for _ in range(10000)])
ms_const1 = OracleMuA(z=1).predict(X_test).mean()       # what the code uses
mean_muY1 = OracleMuY(z=1,f=my_f).predict(X_test).mean() # what the paper describes
out=f"code_constant(mean_compliance)={ms_const1:.4f}  paper_constant(mean_muY1)={mean_muY1:.4f}\n"
print(out)
open(os.path.join(os.path.dirname(__file__),"out","misspec.txt"),"w").write(out)
